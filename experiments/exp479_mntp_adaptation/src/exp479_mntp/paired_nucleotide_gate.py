"""Exact paired nucleotide-prediction gate for causal and bidirectional readouts."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
from torch.utils.data import DataLoader

from exp479_mntp.checkpoint_audit import LoadedArm, _loaded_from_hf
from exp479_mntp.config import EXPERIMENT_TAGS, NUCLEOTIDE_LENGTH, WANDB_PROJECT
from exp479_mntp.data import SequenceCollator, SequencePlanDataset, plan_sha256
from exp479_mntp.mntp_dependency import SOURCE_MODEL_ARTIFACT
from exp479_mntp.modeling import ModelBundle, load_model_bundle, model_logits

EXPECTED_VALIDATION_PLAN_SHA256 = "35542611d71102479f3d07dc6565350120d1d89944e5a93f88efb641ece7e3ba"
PAIRED_GATE_RUN_NAME = "dna-exp479-paired-nucleotide-information-gate"
PAIRED_GATE_ARTIFACT = PAIRED_GATE_RUN_NAME


def _loader(
    validation_plan: Path,
    *,
    tokenizer: Any,
    canonical_ids: tuple[int, ...],
    mask_token_id: int,
    batch_size: int,
) -> DataLoader[dict[str, Any]]:
    dataset = SequencePlanDataset(validation_plan)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=SequenceCollator(
            tokenizer=tokenizer,
            objective="mntp",
            canonical_token_ids=canonical_ids,
            mask_token_id=mask_token_id,
            seed=0,
            validation_mode="single",
        ),
    )


def _model_parts(arm: ModelBundle | LoadedArm) -> tuple[Any, Any, tuple[int, ...], int]:
    canonical_ids = arm.canonical_token_ids if isinstance(arm, ModelBundle) else arm.canonical_ids
    if arm.mask_token_id is None:
        raise RuntimeError("paired nucleotide gate requires a registered mask token")
    return arm.model, arm.tokenizer, canonical_ids, arm.mask_token_id


def evaluate_readout(
    arm: ModelBundle | LoadedArm,
    *,
    validation_plan: Path,
    batch_size: int,
    readout: str,
    attention_mode: str,
    replacement_mask_token_id: int | None = None,
) -> pd.DataFrame:
    """Evaluate one model/readout on identical deterministic single-mask targets."""

    model, tokenizer, canonical_ids, mask_token_id = _model_parts(arm)
    rows: list[dict[str, object]] = []
    canonical = torch.tensor(canonical_ids, device="cuda", dtype=torch.long)
    base_names = ("A", "C", "G", "T")
    model.eval()
    with torch.inference_mode():
        for batch in _loader(
            validation_plan,
            tokenizer=tokenizer,
            canonical_ids=canonical_ids,
            mask_token_id=mask_token_id,
            batch_size=batch_size,
        ):
            input_ids = batch["input_ids"].to(device="cuda")
            attention_mask = batch["attention_mask"].to(device="cuda")
            labels = batch["labels"].to(device="cuda")
            selected = labels != -100
            if not torch.all(selected.sum(dim=1) == 1):
                raise RuntimeError("paired gate requires exactly one target per sequence")
            output_positions = selected.to(dtype=torch.int64).argmax(dim=1)
            batch_rows = torch.arange(input_ids.shape[0], device="cuda")
            target_ids = labels[batch_rows, output_positions]
            target_matches = target_ids[:, None] == canonical[None, :]
            if not torch.all(target_matches.sum(dim=1) == 1):
                raise RuntimeError("paired gate target is not a canonical nucleotide")
            target_indices = target_matches.to(dtype=torch.int64).argmax(dim=1)

            if replacement_mask_token_id is not None:
                input_ids = input_ids.clone()
                input_ids[batch_rows, output_positions + 1] = replacement_mask_token_id

            logits = model_logits(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                attention_mode=attention_mode,  # type: ignore[arg-type]
            )
            readout_logits = logits[batch_rows, output_positions]
            nucleotide_logits = readout_logits.index_select(1, canonical)
            nucleotide_log_probs = nucleotide_logits.log_softmax(dim=-1)
            full_log_probs = readout_logits.log_softmax(dim=-1)
            nucleotide_loss = -nucleotide_log_probs[batch_rows, target_indices]
            full_loss = -full_log_probs[batch_rows, target_ids]
            nucleotide_correct = nucleotide_logits.argmax(dim=-1) == target_indices
            full_correct = readout_logits.argmax(dim=-1) == target_ids
            repeat_masked = (
                batch["loss_weights"][torch.arange(input_ids.shape[0]), output_positions.cpu()] < 1
            )

            for index in range(input_ids.shape[0]):
                target_position = int(output_positions[index])
                target_index = int(target_indices[index])
                rows.append(
                    {
                        "readout": readout,
                        "sample_id": int(batch["sample_ids"][index]),
                        "component": str(batch["components"][index]),
                        "target_nucleotide_index": target_position,
                        "left_context_bases": target_position,
                        "right_context_bases": NUCLEOTIDE_LENGTH - target_position - 1,
                        "target_base": base_names[target_index],
                        "repeat_masked_target": bool(repeat_masked[index]),
                        "nucleotide_ce": float(nucleotide_loss[index]),
                        "nucleotide_correct": float(nucleotide_correct[index]),
                        "full_vocab_ce": float(full_loss[index]),
                        "full_vocab_correct": float(full_correct[index]),
                    }
                )
    return pd.DataFrame(rows)


def summarize_readouts(scores: pd.DataFrame) -> pd.DataFrame:
    """Summarize unweighted nucleotide and full-vocabulary metrics."""

    summary = (
        scores.groupby("readout", sort=False)
        .agg(
            n_targets=("sample_id", "size"),
            nucleotide_ce=("nucleotide_ce", "mean"),
            nucleotide_accuracy=("nucleotide_correct", "mean"),
            full_vocab_ce=("full_vocab_ce", "mean"),
            full_vocab_accuracy=("full_vocab_correct", "mean"),
        )
        .reset_index()
    )
    return summary


def paired_comparison(
    scores: pd.DataFrame,
    *,
    candidate: str,
    baseline: str,
    n_bootstrap: int,
) -> dict[str, object]:
    """Return paired candidate-minus-baseline deltas and sequence-bootstrap intervals."""

    columns = ["sample_id", "nucleotide_ce", "nucleotide_correct"]
    candidate_frame = scores[scores["readout"] == candidate][columns]
    baseline_frame = scores[scores["readout"] == baseline][columns]
    paired = candidate_frame.merge(
        baseline_frame,
        on="sample_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_candidate", "_baseline"),
    )
    if len(paired) != len(candidate_frame) or len(paired) != len(baseline_frame):
        raise RuntimeError(f"incomplete paired comparison {candidate} versus {baseline}")
    ce_delta = (paired["nucleotide_ce_candidate"] - paired["nucleotide_ce_baseline"]).to_numpy()
    accuracy_delta = (
        paired["nucleotide_correct_candidate"] - paired["nucleotide_correct_baseline"]
    ).to_numpy()
    rng = np.random.default_rng(0)
    bootstrap_ce = np.empty(n_bootstrap, dtype=np.float64)
    bootstrap_accuracy = np.empty(n_bootstrap, dtype=np.float64)
    for index in range(n_bootstrap):
        sampled = rng.integers(0, len(paired), size=len(paired))
        bootstrap_ce[index] = ce_delta[sampled].mean()
        bootstrap_accuracy[index] = accuracy_delta[sampled].mean()
    ce_interval = np.quantile(bootstrap_ce, (0.025, 0.975))
    accuracy_interval = np.quantile(bootstrap_accuracy, (0.025, 0.975))
    return {
        "candidate": candidate,
        "baseline": baseline,
        "n_targets": len(paired),
        "nucleotide_ce_delta": float(ce_delta.mean()),
        "nucleotide_ce_delta_ci95_low": float(ce_interval[0]),
        "nucleotide_ce_delta_ci95_high": float(ce_interval[1]),
        "nucleotide_accuracy_delta": float(accuracy_delta.mean()),
        "nucleotide_accuracy_delta_ci95_low": float(accuracy_interval[0]),
        "nucleotide_accuracy_delta_ci95_high": float(accuracy_interval[1]),
    }


def information_gate(comparison: dict[str, object]) -> dict[str, object]:
    """Require the bidirectional readout to be no worse on both paired metrics."""

    ce_delta = float(comparison["nucleotide_ce_delta"])
    accuracy_delta = float(comparison["nucleotide_accuracy_delta"])
    ce_ci_high = float(comparison["nucleotide_ce_delta_ci95_high"])
    accuracy_ci_low = float(comparison["nucleotide_accuracy_delta_ci95_low"])
    point_passed = ce_delta <= 0 and accuracy_delta >= 0
    confidence_supported = ce_ci_high <= 0 and accuracy_ci_low >= 0
    return {
        "candidate": comparison["candidate"],
        "baseline": comparison["baseline"],
        "criterion": (
            "paired four-way nucleotide CE candidate <= baseline and paired top-1 "
            "accuracy candidate >= baseline"
        ),
        "point_estimate_passed": point_passed,
        "confidence_supported": confidence_supported,
        "passed": point_passed and confidence_supported,
    }


def plot_paired_gate(scores: pd.DataFrame, summary: pd.DataFrame, output_path: Path) -> None:
    """Plot overall and position-binned paired nucleotide metrics."""

    display = {
        "source_causal": "Source causal",
        "source_full_new_mask": "Source full + new MASK",
        "source_full_unk_mask": "Source full + UNK mask",
        "adapted_causal_step1000": "Adapted causal",
        "adapted_full_step1000": "Adapted full",
    }
    colors = {
        "source_causal": "#4C78A8",
        "source_full_new_mask": "#F58518",
        "source_full_unk_mask": "#ECA82C",
        "adapted_causal_step1000": "#72B7B2",
        "adapted_full_step1000": "#E45756",
    }
    order = [name for name in display if name in set(summary["readout"])]
    indexed = summary.set_index("readout").loc[order]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    x = np.arange(len(order))
    labels = [display[name] for name in order]
    bar_colors = [colors[name] for name in order]
    axes[0, 0].bar(x, indexed["nucleotide_ce"], color=bar_colors)
    axes[0, 0].set_ylabel("Four-way nucleotide cross-entropy")
    axes[0, 0].set_title("Same 640 targets; lower is better")
    axes[0, 1].bar(x, indexed["nucleotide_accuracy"], color=bar_colors)
    axes[0, 1].set_ylabel("Four-way nucleotide top-1 accuracy")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_title("Same 640 targets; higher is better")
    for axis in axes[0]:
        axis.set_xticks(x, labels, rotation=24, ha="right")
        axis.grid(axis="y", alpha=0.25)

    position_edges = np.linspace(0, NUCLEOTIDE_LENGTH, 9, dtype=int)
    scores = scores.copy()
    scores["position_bin"] = pd.cut(
        scores["target_nucleotide_index"],
        bins=position_edges,
        include_lowest=True,
        right=False,
    )
    binned = (
        scores.groupby(["readout", "position_bin"], observed=True)
        .agg(
            target_index=("target_nucleotide_index", "mean"),
            nucleotide_ce=("nucleotide_ce", "mean"),
            nucleotide_accuracy=("nucleotide_correct", "mean"),
        )
        .reset_index()
    )
    positional_order = [
        name
        for name in (
            "source_causal",
            "source_full_new_mask",
            "adapted_causal_step1000",
            "adapted_full_step1000",
        )
        if name in set(binned["readout"])
    ]
    for name in positional_order:
        selected = binned[binned["readout"] == name]
        axes[1, 0].plot(
            selected["target_index"],
            selected["nucleotide_ce"],
            marker="o",
            color=colors[name],
            label=display[name],
        )
        axes[1, 1].plot(
            selected["target_index"],
            selected["nucleotide_accuracy"],
            marker="o",
            color=colors[name],
            label=display[name],
        )
    axes[1, 0].set_ylabel("Four-way nucleotide cross-entropy")
    axes[1, 1].set_ylabel("Four-way nucleotide top-1 accuracy")
    axes[1, 1].set_ylim(0, 1)
    for axis in axes[1]:
        axis.set_xlabel("Target nucleotide index in 255-bp sequence")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Information gate: identical sequences and target positions")
    figure.savefig(output_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def _release(arm: ModelBundle | LoadedArm) -> None:
    del arm
    gc.collect()
    torch.cuda.empty_cache()


def run_paired_nucleotide_gate(
    *,
    artifact_dir: Path,
    output_dir: Path,
    validation_plan: Path,
    batch_size: int,
    n_bootstrap: int,
) -> None:
    """Evaluate exact paired causal/full-attention nucleotide predictions."""

    if not torch.cuda.is_available():
        raise RuntimeError("paired nucleotide gate requires one CUDA GPU")
    if batch_size <= 0 or n_bootstrap <= 0:
        raise ValueError("batch size and bootstrap count must be positive")
    validation_hash = plan_sha256(validation_plan)
    if validation_hash != EXPECTED_VALIDATION_PLAN_SHA256:
        raise RuntimeError(
            f"validation plan hash {validation_hash} differs from registered "
            f"{EXPECTED_VALIDATION_PLAN_SHA256}"
        )

    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=WANDB_PROJECT,
        group="dna-exp479-mntp-longrun-corrected",
        name=PAIRED_GATE_RUN_NAME,
        tags=[*EXPERIMENT_TAGS, "paired-targets", "nucleotide-prediction", "information-gate"],
        config={
            "validation_plan_sha256": validation_hash,
            "validation_targets": 640,
            "target_selection": "one deterministic eligible A/C/G/T target per sequence",
            "primary_distribution": "A/C/G/T-renormalized",
            "repeat_weighting": "none",
            "source_model_artifact": SOURCE_MODEL_ARTIFACT,
            "batch_size": batch_size,
            "n_bootstrap": n_bootstrap,
        },
    )
    if run is None:
        raise RuntimeError("W&B did not create the paired nucleotide gate run")

    try:
        score_frames: list[pd.DataFrame] = []
        source = load_model_bundle(
            initialization="transferred",
            add_mask=True,
            attention_implementation="sdpa",
            dtype=torch.bfloat16,
        )
        source.model.to(device="cuda", dtype=torch.bfloat16).eval()
        score_frames.append(
            evaluate_readout(
                source,
                validation_plan=validation_plan,
                batch_size=batch_size,
                readout="source_causal",
                attention_mode="causal",
            )
        )
        score_frames.append(
            evaluate_readout(
                source,
                validation_plan=validation_plan,
                batch_size=batch_size,
                readout="source_full_new_mask",
                attention_mode="full",
            )
        )
        score_frames.append(
            evaluate_readout(
                source,
                validation_plan=validation_plan,
                batch_size=batch_size,
                readout="source_full_unk_mask",
                attention_mode="full",
                replacement_mask_token_id=int(source.tokenizer.unk_token_id),
            )
        )
        source.model.to(device="cpu")
        del source
        gc.collect()
        torch.cuda.empty_cache()

        source_artifact = run.use_artifact(SOURCE_MODEL_ARTIFACT, type="model")
        checkpoint_root = Path(source_artifact.download(root=artifact_dir / "retained-step-1000"))
        adapted = _loaded_from_hf(checkpoint_root / "hf", "mntp")
        adapted.model.to(device="cuda", dtype=torch.bfloat16).eval()
        score_frames.append(
            evaluate_readout(
                adapted,
                validation_plan=validation_plan,
                batch_size=batch_size,
                readout="adapted_causal_step1000",
                attention_mode="causal",
            )
        )
        score_frames.append(
            evaluate_readout(
                adapted,
                validation_plan=validation_plan,
                batch_size=batch_size,
                readout="adapted_full_step1000",
                attention_mode="full",
            )
        )
        adapted.model.to(device="cpu")
        del adapted
        gc.collect()
        torch.cuda.empty_cache()

        scores = pd.concat(score_frames, ignore_index=True)
        identity_counts = scores.groupby(["sample_id", "target_nucleotide_index"]).size()
        if len(identity_counts) != 640 or not (identity_counts == len(score_frames)).all():
            raise RuntimeError("readouts did not evaluate an identical 640-target panel")
        summary = summarize_readouts(scores)
        comparisons = [
            paired_comparison(
                scores,
                candidate="source_full_new_mask",
                baseline="source_causal",
                n_bootstrap=n_bootstrap,
            ),
            paired_comparison(
                scores,
                candidate="source_full_unk_mask",
                baseline="source_causal",
                n_bootstrap=n_bootstrap,
            ),
            paired_comparison(
                scores,
                candidate="adapted_full_step1000",
                baseline="adapted_causal_step1000",
                n_bootstrap=n_bootstrap,
            ),
            paired_comparison(
                scores,
                candidate="adapted_full_step1000",
                baseline="source_causal",
                n_bootstrap=n_bootstrap,
            ),
        ]
        comparison_frame = pd.DataFrame(comparisons)
        primary = next(
            item
            for item in comparisons
            if item["candidate"] == "adapted_full_step1000" and item["baseline"] == "source_causal"
        )
        gate = information_gate(primary)

        scores_path = output_dir / "paired-nucleotide-scores.csv"
        summary_path = output_dir / "paired-nucleotide-summary.csv"
        comparisons_path = output_dir / "paired-nucleotide-comparisons.csv"
        gate_path = output_dir / "paired-nucleotide-gate.json"
        figure_path = figures / "paired-nucleotide-information-gate"
        scores.to_csv(scores_path, index=False)
        summary.to_csv(summary_path, index=False)
        comparison_frame.to_csv(comparisons_path, index=False)
        gate_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
        plot_paired_gate(scores, summary, figure_path)

        manifest = {
            "status": "completed",
            "validation_plan_sha256": validation_hash,
            "source_model_revision": "a73a5dcfb3d64b8941e7e7596c6e88ef77db3e7a",
            "source_model_artifact": SOURCE_MODEL_ARTIFACT,
            "source_artifact_id": source_artifact.id,
            "target_count": 640,
            "target_pairing": "identical sample and nucleotide index for every readout",
            "primary_metric": "unweighted A/C/G/T-renormalized cross-entropy and top-1 accuracy",
            "secondary_metric": "unweighted full-vocabulary cross-entropy and top-1 accuracy",
            "knowledge_base_update": "not performed",
            "vep_evaluation": "not performed",
            "checkpoint_deletion": "not performed",
            "hugging_face_upload": "not performed",
            "elapsed_seconds": time.time() - started,
            "gate": gate,
        }
        manifest_path = output_dir / "paired-nucleotide-manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        run.log(
            {
                "paired_nucleotide/summary": wandb.Table(dataframe=summary),
                "paired_nucleotide/comparisons": wandb.Table(dataframe=comparison_frame),
                "paired_nucleotide/figure": wandb.Image(str(figure_path.with_suffix(".png"))),
            }
        )
        run.summary["paired_nucleotide/gate_passed"] = bool(gate["passed"])
        run.summary["paired_nucleotide/point_estimate_passed"] = bool(gate["point_estimate_passed"])
        result_artifact = wandb.Artifact(PAIRED_GATE_ARTIFACT, type="evaluation")
        for path in (
            scores_path,
            summary_path,
            comparisons_path,
            gate_path,
            manifest_path,
            figure_path.with_suffix(".svg"),
        ):
            result_artifact.add_file(str(path))
        logged = run.log_artifact(result_artifact, aliases=["step-1000", "paired-gate"])
        logged.wait()
        run.finish(exit_code=0)
    except BaseException:
        run.finish(exit_code=1)
        raise
