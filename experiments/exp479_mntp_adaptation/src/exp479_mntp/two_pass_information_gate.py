"""Test whether two unchanged causal passes expose useful bidirectional information."""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
from torch.nn.attention import SDPBackend, sdpa_kernel

from exp479_mntp.checkpoint_audit import assert_plan_contract
from exp479_mntp.config import (
    BUDGET_USD,
    EXPERIMENT_TAGS,
    MODEL_ID,
    MODEL_REVISION,
    NUCLEOTIDE_LENGTH,
    SEQUENCE_LENGTH,
    WANDB_PROJECT,
)
from exp479_mntp.data import SequenceCollator, SequencePlanDataset, plan_sha256
from exp479_mntp.gated_lora_mntp import configure_gated_numerics
from exp479_mntp.modeling import ModelBundle, load_model_bundle, model_logits
from exp479_mntp.paired_nucleotide_gate import (
    EXPECTED_VALIDATION_PLAN_SHA256,
    evaluate_readout,
    information_gate,
    paired_comparison,
)
from exp479_mntp.publishing import write_cost_estimate

TWO_PASS_RUN_NAME = "dna-exp479-frozen-two-causal-pass-information-gate"
TWO_PASS_ARTIFACT = "dna-exp479-frozen-two-causal-pass-information-gate"
CALIBRATION_TARGETS = 640
ALPHA_GRID = np.linspace(0.0, 1.0, 1_001, dtype=np.float64)
RC_ALLELE_PERMUTATION = (3, 2, 1, 0)
TWO_PASS_MAX_INSTANCE_HOURS = 1.0


def reverse_complement_token_ids(
    input_ids: torch.Tensor,
    *,
    canonical_token_ids: tuple[int, ...],
    bos_token_id: int,
) -> torch.Tensor:
    """Reverse-complement one-BOS-plus-255-base token batches."""

    if input_ids.ndim != 2 or input_ids.shape[1] != SEQUENCE_LENGTH:
        raise ValueError(f"expected [batch, {SEQUENCE_LENGTH}] token IDs")
    if len(canonical_token_ids) != 4 or len(set(canonical_token_ids)) != 4:
        raise ValueError("canonical token IDs must be distinct A/C/G/T IDs")
    if not torch.all(input_ids[:, 0] == bos_token_id):
        raise ValueError("two-pass inputs must start with BOS")
    reversed_body = input_ids[:, 1:].flip(dims=(1,))
    complemented = reversed_body.clone()
    for source, destination in zip(
        canonical_token_ids,
        reversed(canonical_token_ids),
        strict=True,
    ):
        complemented[reversed_body == source] = destination
    return torch.cat((input_ids[:, :1], complemented), dim=1)


def combine_directional_log_probs(
    left_log_probs: np.ndarray,
    right_log_probs: np.ndarray,
    *,
    log_prior: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Combine left and right conditionals with a tempered prior-corrected product."""

    if left_log_probs.shape != right_log_probs.shape or left_log_probs.shape[-1] != 4:
        raise ValueError("directional log probabilities must have identical [..., 4] shapes")
    if log_prior.shape != (4,):
        raise ValueError("log prior must have shape [4]")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    if alpha == 0:
        return left_log_probs.copy()
    combined = left_log_probs + alpha * (right_log_probs - log_prior)
    return combined - np.logaddexp.reduce(combined, axis=-1, keepdims=True)


def select_alpha(
    frame: pd.DataFrame,
    *,
    log_prior: np.ndarray,
) -> tuple[float, pd.DataFrame]:
    """Choose one scalar on calibration CE only, breaking ties toward the source."""

    left = _log_prob_matrix(frame, prefix="left")
    right = _log_prob_matrix(frame, prefix="right")
    targets = frame["target_nucleotide_class"].to_numpy(dtype=np.int64)
    rows: list[dict[str, float]] = []
    best_index = 0
    best_ce = float("inf")
    sample_rows = np.arange(len(frame))
    for index, alpha in enumerate(ALPHA_GRID):
        combined = combine_directional_log_probs(
            left,
            right,
            log_prior=log_prior,
            alpha=float(alpha),
        )
        ce = float((-combined[sample_rows, targets]).mean())
        accuracy = float((combined.argmax(axis=1) == targets).mean())
        rows.append({"alpha": float(alpha), "nucleotide_ce": ce, "accuracy": accuracy})
        if ce < best_ce:
            best_ce = ce
            best_index = index
    return float(ALPHA_GRID[best_index]), pd.DataFrame(rows)


def _log_prob_matrix(frame: pd.DataFrame, *, prefix: str) -> np.ndarray:
    return frame[[f"{prefix}_logp_{base}" for base in "acgt"]].to_numpy(dtype=np.float64)


def _directional_frame(
    bundle: ModelBundle,
    *,
    plan: Path,
    batch_size: int,
    limit: int | None,
) -> pd.DataFrame:
    dataset = SequencePlanDataset(plan)
    selected_rows = dataset.rows if limit is None else dataset.rows[:limit]
    collator = SequenceCollator(
        tokenizer=bundle.tokenizer,
        objective="mntp",
        canonical_token_ids=bundle.canonical_token_ids,
        mask_token_id=bundle.mask_token_id,
        seed=0,
        validation_mode="single",
    )
    canonical = torch.tensor(bundle.canonical_token_ids, device="cuda", dtype=torch.long)
    rc_permutation = torch.tensor(RC_ALLELE_PERMUTATION, device="cuda", dtype=torch.long)
    records: list[dict[str, Any]] = []
    bundle.model.eval()
    with torch.inference_mode(), sdpa_kernel([SDPBackend.MATH]):
        for start in range(0, len(selected_rows), batch_size):
            batch = collator(selected_rows[start : start + batch_size])
            input_ids = batch["input_ids"].to(device="cuda")
            attention_mask = batch["attention_mask"].to(device="cuda")
            if not torch.all(attention_mask == 1):
                raise RuntimeError("two-pass gate requires fixed unpadded 256-token inputs")
            labels = batch["labels"].to(device="cuda")
            selected = labels != -100
            if not torch.all(selected.sum(dim=1) == 1):
                raise RuntimeError("two-pass gate requires exactly one target per sequence")
            output_positions = selected.to(dtype=torch.int64).argmax(dim=1)
            batch_rows = torch.arange(input_ids.shape[0], device="cuda")
            if bundle.mask_token_id is None or not torch.all(
                input_ids[batch_rows, output_positions + 1] == bundle.mask_token_id
            ):
                raise RuntimeError("forward target mask is not at output position plus one")
            target_ids = labels[batch_rows, output_positions]
            target_matches = target_ids[:, None] == canonical[None, :]
            if not torch.all(target_matches.sum(dim=1) == 1):
                raise RuntimeError("two-pass target is not a canonical nucleotide")
            target_classes = target_matches.to(dtype=torch.int64).argmax(dim=1)

            left_logits = model_logits(
                bundle.model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                attention_mode="causal",
            )[batch_rows, output_positions]
            rc_input_ids = reverse_complement_token_ids(
                input_ids,
                canonical_token_ids=bundle.canonical_token_ids,
                bos_token_id=int(bundle.tokenizer.bos_token_id),
            )
            rc_output_positions = NUCLEOTIDE_LENGTH - 1 - output_positions
            if not torch.all(
                rc_input_ids[batch_rows, rc_output_positions + 1] == bundle.mask_token_id
            ):
                raise RuntimeError("reverse target mask is not at mapped position plus one")
            round_trip = reverse_complement_token_ids(
                rc_input_ids,
                canonical_token_ids=bundle.canonical_token_ids,
                bos_token_id=int(bundle.tokenizer.bos_token_id),
            )
            if not torch.equal(round_trip, input_ids):
                raise RuntimeError("reverse-complement token transform is not an involution")
            right_logits = model_logits(
                bundle.model,
                input_ids=rc_input_ids,
                attention_mask=attention_mask,
                attention_mode="causal",
            )[batch_rows, rc_output_positions]
            left_log_probs = left_logits.index_select(1, canonical).log_softmax(dim=-1)
            right_log_probs = (
                right_logits.index_select(1, canonical)
                .index_select(1, rc_permutation)
                .log_softmax(dim=-1)
            )
            repeat_masked = (
                batch["loss_weights"][torch.arange(input_ids.shape[0]), output_positions.cpu()] < 1
            )
            for row in range(input_ids.shape[0]):
                record: dict[str, Any] = {
                    "sample_id": int(batch["sample_ids"][row]),
                    "component": str(batch["components"][row]),
                    "target_nucleotide_index": int(output_positions[row]),
                    "reverse_target_nucleotide_index": int(rc_output_positions[row]),
                    "target_nucleotide_class": int(target_classes[row]),
                    "repeat_masked_target": bool(repeat_masked[row]),
                }
                for base_index, base in enumerate("acgt"):
                    record[f"left_logp_{base}"] = float(left_log_probs[row, base_index])
                    record[f"right_logp_{base}"] = float(right_log_probs[row, base_index])
                records.append(record)
    return pd.DataFrame(records)


def _score_frame(frame: pd.DataFrame, *, readout: str, log_probs: np.ndarray) -> pd.DataFrame:
    targets = frame["target_nucleotide_class"].to_numpy(dtype=np.int64)
    rows = np.arange(len(frame))
    return frame[
        [
            "sample_id",
            "component",
            "target_nucleotide_index",
            "repeat_masked_target",
        ]
    ].assign(
        readout=readout,
        nucleotide_ce=-log_probs[rows, targets],
        nucleotide_correct=(log_probs.argmax(axis=1) == targets).astype(float),
    )


def _summary(scores: pd.DataFrame) -> pd.DataFrame:
    return (
        scores.groupby("readout", sort=False)
        .agg(
            n_targets=("sample_id", "size"),
            nucleotide_ce=("nucleotide_ce", "mean"),
            nucleotide_accuracy=("nucleotide_correct", "mean"),
        )
        .reset_index()
    )


def _plot(
    calibration: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    selected_alpha: float,
    output_path: Path,
) -> None:
    colors = {"source_left": "#4C78A8", "source_right": "#F58518", "two_pass": "#59A14F"}
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), constrained_layout=True)
    axes[0, 0].plot(calibration["alpha"], calibration["nucleotide_ce"], color="#59A14F")
    axes[0, 0].axvline(selected_alpha, color="black", linestyle="--")
    axes[0, 0].set_ylabel("Calibration four-way CE")
    axes[0, 0].set_xlabel("Right-conditional weight alpha")
    axes[0, 1].plot(calibration["alpha"], calibration["accuracy"], color="#59A14F")
    axes[0, 1].axvline(selected_alpha, color="black", linestyle="--")
    axes[0, 1].set_ylabel("Calibration accuracy")
    axes[0, 1].set_xlabel("Right-conditional weight alpha")

    order = ["source_left", "source_right", "two_pass"]
    indexed = summary.set_index("readout").loc[order]
    x = np.arange(len(order))
    axes[1, 0].bar(x, indexed["nucleotide_ce"], color=[colors[name] for name in order])
    axes[1, 0].set_ylabel("Validation four-way CE")
    axes[1, 1].bar(
        x,
        indexed["nucleotide_accuracy"],
        color=[colors[name] for name in order],
    )
    axes[1, 1].set_ylabel("Validation accuracy")
    axes[1, 1].set_ylim(0, 1)
    for axis in axes[1]:
        axis.set_xticks(x, ("Left causal", "Right causal", "Calibrated two-pass"))
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.suptitle("Frozen source: calibrated left/right causal information gate")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_two_pass_information_gate(
    *,
    artifact_dir: Path,
    output_dir: Path,
    train_plan: Path,
    validation_plan: Path,
    batch_size: int,
    n_bootstrap: int,
) -> None:
    """Calibrate on a train-plan slice and gate on the untouched validation plan."""

    numeric_controls = configure_gated_numerics()
    if not torch.cuda.is_available():
        raise RuntimeError("two-pass information gate requires one CUDA GPU")
    if batch_size <= 0 or n_bootstrap <= 0:
        raise ValueError("batch size and bootstrap count must be positive")
    assert_plan_contract(train_plan, validation_plan)
    validation_hash = plan_sha256(validation_plan)
    if validation_hash != EXPECTED_VALIDATION_PLAN_SHA256:
        raise RuntimeError("two-pass validation plan differs from the registered paired gate")
    prior_cost = float(os.getenv("EXP479_PRIOR_COST_USD", "0"))
    price = float(os.getenv("EXP479_INSTANCE_PRICE_PER_HOUR_USD", "1.006"))
    if prior_cost + TWO_PASS_MAX_INSTANCE_HOURS * price >= BUDGET_USD:
        raise RuntimeError("two-pass projection reaches the issue budget cap")
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    budget_path = output_dir / "prelaunch-budget.json"
    budget_path.write_text(
        json.dumps(
            {
                "prior_cost_usd": prior_cost,
                "maximum_instance_hours": TWO_PASS_MAX_INSTANCE_HOURS,
                "price_per_hour_usd": price,
                "projected_total_usd": prior_cost + TWO_PASS_MAX_INSTANCE_HOURS * price,
                "budget_cap_usd": BUDGET_USD,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    run = wandb.init(
        project=WANDB_PROJECT,
        group="dna-exp479-causal-information-prerequisite",
        name=TWO_PASS_RUN_NAME,
        tags=[*EXPERIMENT_TAGS, "frozen-source", "two-causal-pass", "information-gate"],
        config={
            "base_model": MODEL_ID,
            "base_revision": MODEL_REVISION,
            "calibration_targets": CALIBRATION_TARGETS,
            "calibration_source": "first 640 registered training-plan sequences",
            "alpha_grid": [0.0, 1.0, len(ALPHA_GRID)],
            "combination": "left_logp + alpha * (right_logp - log_empirical_prior)",
            "numeric_controls": numeric_controls,
            "validation_plan_sha256": validation_hash,
        },
    )
    if run is None:
        raise RuntimeError("W&B did not create the two-pass run")

    try:
        loaded = load_model_bundle(
            initialization="transferred",
            add_mask=False,
            attention_implementation="sdpa",
            dtype=torch.bfloat16,
        )
        unk_token_id = loaded.tokenizer.unk_token_id
        if unk_token_id is None:
            raise RuntimeError("source tokenizer lacks UNK for paired masking")
        bundle = ModelBundle(
            model=loaded.model,
            tokenizer=loaded.tokenizer,
            canonical_token_ids=loaded.canonical_token_ids,
            mask_token_id=int(unk_token_id),
            input_output_tied=loaded.input_output_tied,
        )
        bundle.model.to(device="cuda", dtype=torch.bfloat16).eval()
        calibration_frame = _directional_frame(
            bundle,
            plan=train_plan,
            batch_size=batch_size,
            limit=CALIBRATION_TARGETS,
        )
        target_counts = np.bincount(
            calibration_frame["target_nucleotide_class"].to_numpy(dtype=np.int64),
            minlength=4,
        ).astype(np.float64)
        prior = (target_counts + 0.5) / (target_counts.sum() + 2.0)
        log_prior = np.log(prior)
        selected_alpha, calibration_curve = select_alpha(
            calibration_frame,
            log_prior=log_prior,
        )

        validation_frame = _directional_frame(
            bundle,
            plan=validation_plan,
            batch_size=batch_size,
            limit=None,
        )
        if len(validation_frame) != 640:
            raise RuntimeError(f"expected 640 validation targets, found {len(validation_frame)}")
        left = _log_prob_matrix(validation_frame, prefix="left")
        right = _log_prob_matrix(validation_frame, prefix="right")
        combined = combine_directional_log_probs(
            left,
            right,
            log_prior=log_prior,
            alpha=selected_alpha,
        )
        scores = pd.concat(
            [
                _score_frame(validation_frame, readout="source_left", log_probs=left),
                _score_frame(validation_frame, readout="source_right", log_probs=right),
                _score_frame(validation_frame, readout="two_pass", log_probs=combined),
            ],
            ignore_index=True,
        )
        identity = scores.groupby(["sample_id", "target_nucleotide_index"]).size()
        if len(identity) != 640 or not (identity == 3).all():
            raise RuntimeError("two-pass readouts do not share the exact 640 targets")

        with sdpa_kernel([SDPBackend.MATH]):
            canonical_source = evaluate_readout(
                bundle,
                validation_plan=validation_plan,
                batch_size=batch_size,
                readout="canonical_source",
                attention_mode="causal",
            )
        source_left = scores[scores["readout"] == "source_left"].sort_values("sample_id")
        canonical_source = canonical_source.sort_values("sample_id")
        parity = bool(
            np.array_equal(
                source_left["nucleotide_ce"].to_numpy(),
                canonical_source["nucleotide_ce"].to_numpy(),
            )
            and np.array_equal(
                source_left["nucleotide_correct"].to_numpy(),
                canonical_source["nucleotide_correct"].to_numpy(),
            )
        )
        if not parity:
            raise RuntimeError("two-pass alpha-zero source does not match canonical evaluation")

        summary = _summary(scores)
        right_comparison = paired_comparison(
            scores,
            candidate="source_right",
            baseline="source_left",
            n_bootstrap=n_bootstrap,
        )
        comparison = paired_comparison(
            scores,
            candidate="two_pass",
            baseline="source_left",
            n_bootstrap=n_bootstrap,
        )
        noninferiority = information_gate(comparison)
        strict = bool(
            float(comparison["nucleotide_ce_delta_ci95_high"]) < 0
            or float(comparison["nucleotide_accuracy_delta_ci95_low"]) > 0
        )
        gate = {
            "criterion": (
                "calibrated alpha is positive, paired CE/accuracy are confidence-supported "
                "non-inferior to left causal, and at least one metric strictly improves"
            ),
            "selected_alpha": selected_alpha,
            "alpha_positive": selected_alpha > 0,
            "source_noninferiority": noninferiority,
            "confidence_strict_improvement": strict,
            "passed": bool(selected_alpha > 0 and noninferiority["passed"] and strict),
        }

        scores_path = output_dir / "paired-nucleotide-scores.csv"
        summary_path = output_dir / "paired-nucleotide-summary.csv"
        comparisons_path = output_dir / "paired-nucleotide-comparisons.csv"
        gate_path = output_dir / "paired-nucleotide-gate.json"
        calibration_path = output_dir / "calibration-alpha-curve.csv"
        calibration_directional_path = output_dir / "calibration-directional-log-probs.csv"
        validation_directional_path = output_dir / "validation-directional-log-probs.csv"
        manifest_path = output_dir / "manifest.json"
        figure_path = output_dir / "figures" / "two-pass-information-gate"
        scores.to_csv(scores_path, index=False)
        summary.to_csv(summary_path, index=False)
        calibration_curve.to_csv(calibration_path, index=False)
        pd.DataFrame([right_comparison, comparison]).to_csv(comparisons_path, index=False)
        calibration_frame.to_csv(calibration_directional_path, index=False)
        validation_frame.to_csv(validation_directional_path, index=False)
        gate_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
        _plot(
            calibration_curve,
            summary,
            selected_alpha=selected_alpha,
            output_path=figure_path,
        )
        runtime_path = output_dir / "runtime.json"
        runtime_path.write_text(
            json.dumps({"elapsed_seconds": time.time() - started}, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "status": "completed",
            "base_model": MODEL_ID,
            "base_revision": MODEL_REVISION,
            "model_updates": 0,
            "source_alpha_zero_bit_exact": parity,
            "calibration_plan_sha256": plan_sha256(train_plan),
            "calibration_targets": CALIBRATION_TARGETS,
            "validation_plan_sha256": validation_hash,
            "validation_targets": 640,
            "mask_token": "[UNK]",
            "reverse_position_mapping": "i -> 254 - i",
            "reverse_allele_permutation": list(RC_ALLELE_PERMUTATION),
            "empirical_prior_acgt": prior.tolist(),
            "selected_alpha": selected_alpha,
            "numeric_controls": numeric_controls,
            "right_only_comparison": right_comparison,
            "gate": gate,
            "vep_evaluation": "not performed",
            "nucleotide_dependency": "not performed",
            "knowledge_base_update": "not performed",
            "hugging_face_upload": "not performed",
            "checkpoint_deletion": "not performed",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        cost_path = write_cost_estimate(artifact_dir=artifact_dir)

        run.log(
            {
                "two_pass/summary": wandb.Table(dataframe=summary),
                "two_pass/calibration": wandb.Table(dataframe=calibration_curve),
                "two_pass/figure": wandb.Image(str(figure_path.with_suffix(".png"))),
                "two_pass/selected_alpha": selected_alpha,
            }
        )
        run.summary["two_pass/passed"] = bool(gate["passed"])
        result = wandb.Artifact(TWO_PASS_ARTIFACT, type="evaluation")
        for path in (
            scores_path,
            summary_path,
            comparisons_path,
            gate_path,
            calibration_path,
            calibration_directional_path,
            validation_directional_path,
            manifest_path,
            runtime_path,
            budget_path,
            cost_path,
            figure_path.with_suffix(".svg"),
        ):
            result.add_file(str(path))
        logged = run.log_artifact(result, aliases=["paired-gate"])
        logged.wait()
        run.finish(exit_code=0)
    except BaseException:
        run.finish(exit_code=1)
        raise
    finally:
        if "bundle" in locals():
            bundle.model.to(device="cpu")
        gc.collect()
        torch.cuda.empty_cache()
