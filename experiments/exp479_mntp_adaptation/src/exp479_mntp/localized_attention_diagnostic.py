"""Zero-training paired diagnostic for localized predictor-row attention."""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
from torch.nn.attention import SDPBackend, sdpa_kernel

from exp479_mntp.attention_anneal_diagnostic import _endpoint_delta
from exp479_mntp.config import BUDGET_USD, EXPERIMENT_TAGS, MODEL_REVISION, WANDB_PROJECT
from exp479_mntp.data import plan_sha256
from exp479_mntp.modeling import ModelBundle, load_model_bundle
from exp479_mntp.paired_nucleotide_gate import (
    EXPECTED_VALIDATION_PLAN_SHA256,
    evaluate_readout,
    information_gate,
    paired_comparison,
    summarize_readouts,
)
from exp479_mntp.publishing import assert_budget_reserve, write_cost_estimate

LOCALIZED_ATTENTION_RUN_NAME = "dna-exp479-source-localized-predictor-attention"
LOCALIZED_ATTENTION_ARTIFACT = LOCALIZED_ATTENTION_RUN_NAME
PREDIFF_REFERENCE = "https://arxiv.org/abs/2607.25157"
MAXIMUM_INSTANCE_HOURS = 1.0
ATTENTION_ENCODING_CE_TOLERANCE = 2e-3


def predictor_row_attention_mask(
    attention_mask: torch.Tensor,
    output_positions: torch.Tensor,
    *,
    dtype: torch.dtype,
    open_predictor_row: bool = True,
) -> torch.Tensor:
    """Keep causal attention except optionally opening each selected predictor row."""

    if attention_mask.ndim != 2:
        raise ValueError("token attention mask must have shape [batch, sequence]")
    if output_positions.ndim != 1 or output_positions.shape[0] != attention_mask.shape[0]:
        raise ValueError("output positions must contain one row index per sequence")
    if not dtype.is_floating_point:
        raise ValueError("additive attention mask requires a floating dtype")
    batch_size, sequence_length = attention_mask.shape
    positions = output_positions.to(device=attention_mask.device, dtype=torch.long)
    if torch.any(positions < 0) or torch.any(positions >= sequence_length):
        raise ValueError("output position is outside the token sequence")

    causal = torch.ones(
        (sequence_length, sequence_length),
        dtype=torch.bool,
        device=attention_mask.device,
    ).tril()
    allowed = causal[None, None, :, :].expand(batch_size, 1, -1, -1).clone()
    key_allowed = attention_mask.to(dtype=torch.bool)
    allowed = torch.logical_and(allowed, key_allowed[:, None, None, :])
    if open_predictor_row:
        batch_rows = torch.arange(batch_size, device=attention_mask.device)
        allowed[batch_rows, 0, positions, :] = key_allowed
    additive = torch.zeros(allowed.shape, dtype=dtype, device=attention_mask.device)
    return additive.masked_fill(~allowed, torch.finfo(dtype).min)


def _plot_localized_attention(
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot absolute paired metrics and candidate-minus-causal intervals."""

    display = {
        "source_causal_standard": "Source causal",
        "source_localized_predictor_row": "Localized predictor row",
        "source_full_standard": "Uniform full attention",
    }
    colors = {
        "source_causal_standard": "#4C78A8",
        "source_localized_predictor_row": "#59A14F",
        "source_full_standard": "#E45756",
    }
    order = list(display)
    indexed = summary.set_index("readout").loc[order]
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), constrained_layout=True)
    x = np.arange(len(order))
    labels = [display[name] for name in order]
    bar_colors = [colors[name] for name in order]
    axes[0, 0].bar(x, indexed["nucleotide_ce"], color=bar_colors)
    axes[0, 1].bar(x, indexed["nucleotide_accuracy"], color=bar_colors)
    axes[0, 0].set_ylabel("Four-way nucleotide CE")
    axes[0, 1].set_ylabel("Four-way nucleotide accuracy")
    axes[0, 1].set_ylim(0, 1)
    for axis in axes[0]:
        axis.set_xticks(x, labels, rotation=15, ha="right")
        axis.grid(axis="y", alpha=0.25)

    ordered = comparisons.set_index("candidate").loc[order[1:]].reset_index()
    delta_x = np.arange(len(ordered))
    delta_labels = [display[name] for name in ordered["candidate"]]
    ce = ordered["nucleotide_ce_delta"].to_numpy(dtype=float)
    ce_low = ordered["nucleotide_ce_delta_ci95_low"].to_numpy(dtype=float)
    ce_high = ordered["nucleotide_ce_delta_ci95_high"].to_numpy(dtype=float)
    accuracy = ordered["nucleotide_accuracy_delta"].to_numpy(dtype=float)
    accuracy_low = ordered["nucleotide_accuracy_delta_ci95_low"].to_numpy(dtype=float)
    accuracy_high = ordered["nucleotide_accuracy_delta_ci95_high"].to_numpy(dtype=float)
    axes[1, 0].errorbar(
        delta_x,
        ce,
        yerr=np.maximum(0, np.vstack((ce - ce_low, ce_high - ce))),
        fmt="o",
        color="#333333",
        capsize=4,
    )
    axes[1, 1].errorbar(
        delta_x,
        accuracy,
        yerr=np.maximum(0, np.vstack((accuracy - accuracy_low, accuracy_high - accuracy))),
        fmt="o",
        color="#333333",
        capsize=4,
    )
    axes[1, 0].axhline(0, color="black", linewidth=1)
    axes[1, 1].axhline(0, color="black", linewidth=1)
    axes[1, 0].set_ylabel("Paired CE delta vs causal")
    axes[1, 1].set_ylabel("Paired accuracy delta vs causal")
    for axis in axes[1]:
        axis.set_xticks(delta_x, delta_labels, rotation=15, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Frozen source: localize right-context access to the shifted predictor row")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_localized_attention_diagnostic(
    *,
    artifact_dir: Path,
    output_dir: Path,
    validation_plan: Path,
    batch_size: int,
    n_bootstrap: int,
) -> None:
    """Compare frozen causal, localized-row, and uniform-full paired predictions."""

    if not torch.cuda.is_available():
        raise RuntimeError("localized attention diagnostic requires one CUDA GPU")
    if batch_size <= 0 or n_bootstrap <= 0:
        raise ValueError("batch size and bootstrap count must be positive")
    validation_hash = plan_sha256(validation_plan)
    if validation_hash != EXPECTED_VALIDATION_PLAN_SHA256:
        raise RuntimeError("localized diagnostic validation plan differs from the paired gate")
    assert_budget_reserve()
    prior_cost = float(os.getenv("EXP479_PRIOR_COST_USD", "0"))
    price = float(os.getenv("EXP479_INSTANCE_PRICE_PER_HOUR_USD", "1.29"))
    if prior_cost + MAXIMUM_INSTANCE_HOURS * price >= BUDGET_USD:
        raise RuntimeError("localized attention diagnostic projection reaches the issue budget cap")

    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=WANDB_PROJECT,
        group="dna-exp479-paired-nucleotide-information",
        name=LOCALIZED_ATTENTION_RUN_NAME,
        tags=[*EXPERIMENT_TAGS, "no-training", "localized-attention", "paired-targets"],
        config={
            "base_revision": MODEL_REVISION,
            "mask_token": "[UNK]",
            "validation_plan_sha256": validation_hash,
            "validation_targets": 640,
            "parameter_updates": 0,
            "reference": PREDIFF_REFERENCE,
            "attention_pattern": "causal except selected shifted predictor row",
            "batch_size": batch_size,
            "n_bootstrap": n_bootstrap,
        },
    )
    if run is None:
        raise RuntimeError("W&B did not create the localized attention diagnostic")

    try:
        loaded = load_model_bundle(
            initialization="transferred",
            add_mask=False,
            attention_implementation="sdpa",
            dtype=torch.bfloat16,
        )
        unk_token_id = loaded.tokenizer.unk_token_id
        if unk_token_id is None:
            raise RuntimeError("source tokenizer lacks [UNK]")
        source = ModelBundle(
            model=loaded.model,
            tokenizer=loaded.tokenizer,
            canonical_token_ids=loaded.canonical_token_ids,
            mask_token_id=int(unk_token_id),
            input_output_tied=loaded.input_output_tied,
        )
        source.model.to(device="cuda", dtype=torch.bfloat16).eval()

        def causal_transform(
            token_mask: torch.Tensor,
            sample_ids: torch.Tensor,
            output_positions: torch.Tensor,
        ) -> torch.Tensor:
            del sample_ids
            return predictor_row_attention_mask(
                token_mask,
                output_positions,
                dtype=torch.bfloat16,
                open_predictor_row=False,
            )

        def localized_transform(
            token_mask: torch.Tensor,
            sample_ids: torch.Tensor,
            output_positions: torch.Tensor,
        ) -> torch.Tensor:
            del sample_ids
            return predictor_row_attention_mask(
                token_mask,
                output_positions,
                dtype=torch.bfloat16,
            )

        with sdpa_kernel([SDPBackend.MATH]):
            standard_causal = evaluate_readout(
                source,
                validation_plan=validation_plan,
                batch_size=batch_size,
                readout="source_causal_standard",
                attention_mode="causal",
            )
            additive_causal = evaluate_readout(
                source,
                validation_plan=validation_plan,
                batch_size=batch_size,
                readout="source_causal_additive",
                attention_mode="full",
                attention_mask_transform=causal_transform,
            )
            localized = evaluate_readout(
                source,
                validation_plan=validation_plan,
                batch_size=batch_size,
                readout="source_localized_predictor_row",
                attention_mode="full",
                attention_mask_transform=localized_transform,
            )
            full = evaluate_readout(
                source,
                validation_plan=validation_plan,
                batch_size=batch_size,
                readout="source_full_standard",
                attention_mode="full",
            )

        encoding_check = _endpoint_delta(additive_causal, standard_causal)
        encoding_check["ce_tolerance"] = ATTENTION_ENCODING_CE_TOLERANCE
        encoding_check["passed"] = bool(
            encoding_check["maximum_absolute_nucleotide_ce_delta"]
            <= ATTENTION_ENCODING_CE_TOLERANCE
            and encoding_check["nucleotide_prediction_mismatches"] == 0
        )
        if not encoding_check["passed"]:
            raise RuntimeError(
                f"localized diagnostic causal encoding check failed: {encoding_check}"
            )

        scores = pd.concat(
            [standard_causal, additive_causal, localized, full],
            ignore_index=True,
        )
        summary = summarize_readouts(scores)
        comparisons = pd.DataFrame(
            [
                paired_comparison(
                    scores,
                    candidate=candidate,
                    baseline="source_causal_standard",
                    n_bootstrap=n_bootstrap,
                )
                for candidate in ("source_localized_predictor_row", "source_full_standard")
            ]
        )
        gate = information_gate(comparisons.iloc[0].to_dict())

        scores_path = output_dir / "localized-attention-scores.csv"
        summary_path = output_dir / "localized-attention-summary.csv"
        comparisons_path = output_dir / "localized-attention-comparisons.csv"
        gate_path = output_dir / "localized-attention-gate.json"
        encoding_path = output_dir / "attention-encoding-check.json"
        runtime_path = output_dir / "runtime.json"
        figure_path = output_dir / "figures" / "localized-attention"
        scores.to_csv(scores_path, index=False)
        summary.to_csv(summary_path, index=False)
        comparisons.to_csv(comparisons_path, index=False)
        gate_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
        encoding_path.write_text(json.dumps(encoding_check, indent=2) + "\n", encoding="utf-8")
        _plot_localized_attention(summary, comparisons, figure_path)
        runtime_path.write_text(
            json.dumps({"elapsed_seconds": time.time() - started}, indent=2) + "\n",
            encoding="utf-8",
        )
        cost_path = write_cost_estimate(artifact_dir=artifact_dir)
        manifest = {
            "status": "completed",
            "wandb_url": run.get_url(),
            "base_revision": MODEL_REVISION,
            "validation_plan_sha256": validation_hash,
            "paired_target_count": 640,
            "mask_token": "[UNK]",
            "parameter_updates": 0,
            "attention_backend": "math SDPA for every compared readout",
            "attention_pattern": (
                "causal for every query except the shifted predictor row i-1, which attends "
                "to every non-padding key"
            ),
            "reference": PREDIFF_REFERENCE,
            "reference_boundary": (
                "PreDiff-LM partitions a prompt prefix and target suffix and predicts at masked "
                "positions; this arbitrary-interior shifted-row mask is an exp479 inference"
            ),
            "attention_encoding_check": encoding_check,
            "gate": gate,
            "checkpoint_deletion": "not performed",
            "hugging_face_upload": "not performed",
            "vep_evaluation": "not performed",
            "nucleotide_dependency": "not performed",
            "knowledge_base_update": "not performed",
        }
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        run.log(
            {
                "localized_attention/summary": wandb.Table(dataframe=summary),
                "localized_attention/comparisons": wandb.Table(dataframe=comparisons),
                "localized_attention/figure": wandb.Image(str(figure_path.with_suffix(".png"))),
            }
        )
        run.summary["localized_attention/gate_passed"] = bool(gate["passed"])
        run.summary["localized_attention/encoding_passed"] = bool(encoding_check["passed"])
        result = wandb.Artifact(LOCALIZED_ATTENTION_ARTIFACT, type="evaluation")
        for path in (
            scores_path,
            summary_path,
            comparisons_path,
            gate_path,
            encoding_path,
            runtime_path,
            cost_path,
            manifest_path,
            figure_path.with_suffix(".svg"),
        ):
            result.add_file(str(path))
        logged = run.log_artifact(result, aliases=["zero-training", "predictor-row"])
        logged.wait()
        run.finish(exit_code=0)
        del source, loaded
        gc.collect()
        torch.cuda.empty_cache()
    except BaseException:
        run.finish(exit_code=1)
        raise
