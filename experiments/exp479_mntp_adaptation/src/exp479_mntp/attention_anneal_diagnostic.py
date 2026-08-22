"""Frozen-source nucleotide accuracy while stochastic future attention is opened."""

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

from exp479_mntp.config import BUDGET_USD, EXPERIMENT_TAGS, MODEL_REVISION, WANDB_PROJECT
from exp479_mntp.data import plan_sha256
from exp479_mntp.lora_mntp import annealed_attention_mask
from exp479_mntp.masking import sample_seed
from exp479_mntp.modeling import ModelBundle, load_model_bundle
from exp479_mntp.paired_nucleotide_gate import (
    EXPECTED_VALIDATION_PLAN_SHA256,
    evaluate_readout,
    paired_comparison,
)
from exp479_mntp.publishing import assert_budget_reserve

ATTENTION_PROBABILITIES = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
ATTENTION_MASK_REPLICATES = 5
ATTENTION_ANNEAL_RUN_NAME = "dna-exp479-source-attention-annealing-diagnostic"
ATTENTION_ANNEAL_ARTIFACT = ATTENTION_ANNEAL_RUN_NAME
MAXIMUM_INSTANCE_HOURS = 2.0


def _readout_name(probability_index: int, replicate: int) -> str:
    return f"source_unk_p{probability_index:02d}_mask{replicate}"


def _mean_readout_name(probability_index: int) -> str:
    return f"source_unk_p{probability_index:02d}_mean"


def summarize_annealing(
    scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate stochastic-mask replicates and exact paired target means."""

    required = {
        "future_edge_probability",
        "mask_replicate",
        "sample_id",
        "target_nucleotide_index",
        "nucleotide_ce",
        "nucleotide_correct",
        "full_vocab_ce",
        "full_vocab_correct",
    }
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"annealing scores lack columns: {sorted(missing)}")
    probabilities = sorted(scores["future_edge_probability"].unique())
    replicates = sorted(scores["mask_replicate"].unique())
    identity = scores.groupby(["future_edge_probability", "mask_replicate", "sample_id"]).size()
    if not (identity == 1).all():
        raise RuntimeError("annealing trajectory repeats a probability/replicate/sample")
    counts = scores.groupby(["future_edge_probability", "mask_replicate"]).size()
    if counts.nunique() != 1:
        raise RuntimeError("annealing trajectory has incomplete target panels")

    replicate_summary = (
        scores.groupby(["future_edge_probability", "mask_replicate"], sort=True)
        .agg(
            n_targets=("sample_id", "size"),
            nucleotide_ce=("nucleotide_ce", "mean"),
            nucleotide_accuracy=("nucleotide_correct", "mean"),
            full_vocab_ce=("full_vocab_ce", "mean"),
            full_vocab_accuracy=("full_vocab_correct", "mean"),
        )
        .reset_index()
    )
    trajectory = (
        replicate_summary.groupby("future_edge_probability", sort=True)
        .agg(
            mask_replicates=("mask_replicate", "size"),
            n_targets=("n_targets", "first"),
            nucleotide_ce=("nucleotide_ce", "mean"),
            nucleotide_ce_min=("nucleotide_ce", "min"),
            nucleotide_ce_max=("nucleotide_ce", "max"),
            nucleotide_accuracy=("nucleotide_accuracy", "mean"),
            nucleotide_accuracy_min=("nucleotide_accuracy", "min"),
            nucleotide_accuracy_max=("nucleotide_accuracy", "max"),
            full_vocab_ce=("full_vocab_ce", "mean"),
            full_vocab_accuracy=("full_vocab_accuracy", "mean"),
        )
        .reset_index()
    )
    if trajectory["mask_replicates"].tolist() != [len(replicates)] * len(probabilities):
        raise RuntimeError("annealing trajectory has inconsistent mask replicate counts")

    first = trajectory.iloc[0]
    last = trajectory.iloc[-1]
    ce_span = float(last["nucleotide_ce"] - first["nucleotide_ce"])
    accuracy_span = float(first["nucleotide_accuracy"] - last["nucleotide_accuracy"])
    if ce_span <= 0 or accuracy_span <= 0:
        raise RuntimeError("full attention does not degrade both registered source metrics")
    trajectory["ce_degradation_fraction"] = (
        trajectory["nucleotide_ce"] - float(first["nucleotide_ce"])
    ) / ce_span
    trajectory["accuracy_degradation_fraction"] = (
        float(first["nucleotide_accuracy"]) - trajectory["nucleotide_accuracy"]
    ) / accuracy_span

    metadata = [
        "future_edge_probability",
        "sample_id",
        "component",
        "target_nucleotide_index",
        "left_context_bases",
        "right_context_bases",
        "target_base",
        "repeat_masked_target",
    ]
    target_means = (
        scores.groupby(metadata, sort=True)
        .agg(
            nucleotide_ce=("nucleotide_ce", "mean"),
            nucleotide_correct=("nucleotide_correct", "mean"),
            full_vocab_ce=("full_vocab_ce", "mean"),
            full_vocab_correct=("full_vocab_correct", "mean"),
        )
        .reset_index()
    )
    probability_to_index = {probability: index for index, probability in enumerate(probabilities)}
    target_means["readout"] = target_means["future_edge_probability"].map(
        lambda probability: _mean_readout_name(probability_to_index[probability])
    )
    return replicate_summary, trajectory, target_means


def _plot_annealing(
    replicate_summary: pd.DataFrame,
    trajectory: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot no-training nucleotide metrics over future-edge probability."""

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    for _, frame in replicate_summary.groupby("mask_replicate", sort=True):
        axes[0].plot(
            frame["future_edge_probability"],
            frame["nucleotide_ce"],
            color="#E45756",
            alpha=0.22,
            linewidth=1,
        )
        axes[1].plot(
            frame["future_edge_probability"],
            frame["nucleotide_accuracy"],
            color="#4C78A8",
            alpha=0.22,
            linewidth=1,
        )
    probability = trajectory["future_edge_probability"].to_numpy(dtype=float)
    axes[0].fill_between(
        probability,
        trajectory["nucleotide_ce_min"].to_numpy(dtype=float),
        trajectory["nucleotide_ce_max"].to_numpy(dtype=float),
        color="#E45756",
        alpha=0.15,
        label="range across 5 nested masks",
    )
    axes[0].plot(
        probability,
        trajectory["nucleotide_ce"],
        marker="o",
        color="#E45756",
        linewidth=2,
        label="mean",
    )
    axes[1].fill_between(
        probability,
        trajectory["nucleotide_accuracy_min"].to_numpy(dtype=float),
        trajectory["nucleotide_accuracy_max"].to_numpy(dtype=float),
        color="#4C78A8",
        alpha=0.15,
        label="range across 5 nested masks",
    )
    axes[1].plot(
        probability,
        trajectory["nucleotide_accuracy"],
        marker="o",
        color="#4C78A8",
        linewidth=2,
        label="mean",
    )
    axes[0].set_ylabel("Four-way nucleotide CE")
    axes[1].set_ylabel("Four-way nucleotide accuracy")
    axes[1].set_ylim(0, 1)
    for axis in axes:
        axis.set_xlabel("Probability of opening each future attention edge")
        axis.set_xlim(0, 1)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    axes[0].set_title("Loss without parameter updates")
    axes[1].set_title("Accuracy without parameter updates")
    figure.suptitle("Frozen source with [UNK]: causal-to-full attention trajectory")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def _endpoint_delta(
    custom: pd.DataFrame,
    standard: pd.DataFrame,
) -> dict[str, float | bool]:
    paired = custom.merge(
        standard,
        on=["sample_id", "target_nucleotide_index"],
        how="inner",
        validate="one_to_one",
        suffixes=("_custom", "_standard"),
    )
    if len(paired) != len(custom) or len(paired) != len(standard):
        raise RuntimeError("custom and standard attention endpoints do not share exact targets")
    ce_delta = np.abs(
        paired["nucleotide_ce_custom"].to_numpy() - paired["nucleotide_ce_standard"].to_numpy()
    )
    return {
        "maximum_absolute_nucleotide_ce_delta": float(ce_delta.max()),
        "nucleotide_predictions_identical": bool(
            np.array_equal(
                paired["nucleotide_correct_custom"].to_numpy(),
                paired["nucleotide_correct_standard"].to_numpy(),
            )
        ),
    }


def run_attention_anneal_diagnostic(
    *,
    artifact_dir: Path,
    output_dir: Path,
    validation_plan: Path,
    batch_size: int,
    n_bootstrap: int,
) -> None:
    """Measure source degradation under nested stochastic attention masks."""

    if not torch.cuda.is_available():
        raise RuntimeError("attention annealing diagnostic requires one CUDA GPU")
    if batch_size <= 0 or n_bootstrap <= 0:
        raise ValueError("batch size and bootstrap count must be positive")
    validation_hash = plan_sha256(validation_plan)
    if validation_hash != EXPECTED_VALIDATION_PLAN_SHA256:
        raise RuntimeError("attention diagnostic validation plan differs from the paired gate")
    assert_budget_reserve()
    prior_cost = float(os.getenv("EXP479_PRIOR_COST_USD", "0"))
    price = float(os.getenv("EXP479_INSTANCE_PRICE_PER_HOUR_USD", "1.29"))
    if prior_cost + MAXIMUM_INSTANCE_HOURS * price >= BUDGET_USD:
        raise RuntimeError("attention diagnostic projection reaches the issue budget cap")

    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=WANDB_PROJECT,
        group="dna-exp479-paired-nucleotide-information",
        name=ATTENTION_ANNEAL_RUN_NAME,
        tags=[*EXPERIMENT_TAGS, "no-training", "attention-annealing", "paired-targets"],
        config={
            "base_revision": MODEL_REVISION,
            "mask_token": "[UNK]",
            "validation_plan_sha256": validation_hash,
            "validation_targets": 640,
            "future_edge_probabilities": list(ATTENTION_PROBABILITIES),
            "nested_mask_replicates": ATTENTION_MASK_REPLICATES,
            "batch_size": batch_size,
            "n_bootstrap": n_bootstrap,
            "parameter_updates": 0,
        },
    )
    if run is None:
        raise RuntimeError("W&B did not create the attention annealing diagnostic")

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
        standard_causal = evaluate_readout(
            source,
            validation_plan=validation_plan,
            batch_size=batch_size,
            readout="standard_causal",
            attention_mode="causal",
        )
        standard_full = evaluate_readout(
            source,
            validation_plan=validation_plan,
            batch_size=batch_size,
            readout="standard_full",
            attention_mode="full",
        )

        frames: list[pd.DataFrame] = []
        for probability_index, probability in enumerate(ATTENTION_PROBABILITIES):
            for replicate in range(ATTENTION_MASK_REPLICATES):

                def transform(
                    token_mask: torch.Tensor,
                    sample_ids: torch.Tensor,
                    *,
                    selected_probability: float = probability,
                    selected_replicate: int = replicate,
                ) -> torch.Tensor:
                    return annealed_attention_mask(
                        token_mask,
                        future_edge_probability=selected_probability,
                        seed=sample_seed(
                            selected_replicate,
                            int(sample_ids[0]),
                            stream=4,
                        ),
                        dtype=torch.bfloat16,
                    )

                frame = evaluate_readout(
                    source,
                    validation_plan=validation_plan,
                    batch_size=batch_size,
                    readout=_readout_name(probability_index, replicate),
                    attention_mode="full",
                    attention_mask_transform=transform,
                )
                frame["future_edge_probability"] = probability
                frame["mask_replicate"] = replicate
                frames.append(frame)
        scores = pd.concat(frames, ignore_index=True)
        replicate_summary, trajectory, target_means = summarize_annealing(scores)

        first_custom = scores[
            (scores["future_edge_probability"] == 0.0) & (scores["mask_replicate"] == 0)
        ]
        last_custom = scores[
            (scores["future_edge_probability"] == 1.0) & (scores["mask_replicate"] == 0)
        ]
        endpoint_checks = {
            "causal": _endpoint_delta(first_custom, standard_causal),
            "full": _endpoint_delta(last_custom, standard_full),
        }
        if not all(
            bool(check["nucleotide_predictions_identical"])
            and float(check["maximum_absolute_nucleotide_ce_delta"]) < 0.002
            for check in endpoint_checks.values()
        ):
            raise RuntimeError(f"custom attention endpoints fail parity: {endpoint_checks}")

        probabilities = list(ATTENTION_PROBABILITIES)
        baseline = _mean_readout_name(0)
        comparisons = []
        for probability_index in range(1, len(probabilities)):
            comparison = paired_comparison(
                target_means,
                candidate=_mean_readout_name(probability_index),
                baseline=baseline,
                n_bootstrap=n_bootstrap,
            )
            comparison["future_edge_probability"] = probabilities[probability_index]
            comparisons.append(comparison)
        comparison_frame = pd.DataFrame(comparisons)

        ce_values = trajectory["nucleotide_ce"].to_numpy(dtype=float)
        accuracy_values = trajectory["nucleotide_accuracy"].to_numpy(dtype=float)
        interpretation = {
            "adjacent_ce_increases": int((np.diff(ce_values) >= 0).sum()),
            "adjacent_accuracy_decreases": int((np.diff(accuracy_values) <= 0).sum()),
            "adjacent_intervals": len(trajectory) - 1,
            "causal_nucleotide_ce": float(ce_values[0]),
            "full_nucleotide_ce": float(ce_values[-1]),
            "causal_nucleotide_accuracy": float(accuracy_values[0]),
            "full_nucleotide_accuracy": float(accuracy_values[-1]),
            "endpoint_checks": endpoint_checks,
        }

        scores_path = output_dir / "attention-annealing-scores.csv"
        replicate_path = output_dir / "attention-annealing-replicates.csv"
        trajectory_path = output_dir / "attention-annealing-trajectory.csv"
        target_means_path = output_dir / "attention-annealing-target-means.csv"
        comparisons_path = output_dir / "attention-annealing-comparisons.csv"
        interpretation_path = output_dir / "attention-annealing-interpretation.json"
        figure_path = output_dir / "figures" / "attention-annealing-trajectory"
        scores.to_csv(scores_path, index=False)
        replicate_summary.to_csv(replicate_path, index=False)
        trajectory.to_csv(trajectory_path, index=False)
        target_means.to_csv(target_means_path, index=False)
        comparison_frame.to_csv(comparisons_path, index=False)
        interpretation_path.write_text(
            json.dumps(interpretation, indent=2) + "\n",
            encoding="utf-8",
        )
        _plot_annealing(replicate_summary, trajectory, figure_path)
        manifest = {
            "status": "completed",
            "parameter_updates": 0,
            "base_revision": MODEL_REVISION,
            "mask_token": "[UNK]",
            "validation_plan_sha256": validation_hash,
            "target_count": 640,
            "attention_masks": "nested DiffuLLaMA stochastic future-edge masks",
            "mask_replicates": ATTENTION_MASK_REPLICATES,
            "elapsed_seconds": time.time() - started,
            "vep_evaluation": "not performed",
            "nucleotide_dependency": "not performed",
            "knowledge_base_update": "not performed",
            "hugging_face_upload": "not performed",
            "checkpoint_deletion": "not performed",
        }
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        run.log(
            {
                "attention_annealing/trajectory": wandb.Table(dataframe=trajectory),
                "attention_annealing/comparisons": wandb.Table(dataframe=comparison_frame),
                "attention_annealing/figure": wandb.Image(str(figure_path.with_suffix(".png"))),
            }
        )
        for key, value in interpretation.items():
            if key != "endpoint_checks":
                run.summary[f"attention_annealing/{key}"] = value
        artifact = wandb.Artifact(ATTENTION_ANNEAL_ARTIFACT, type="evaluation")
        for path in (
            scores_path,
            replicate_path,
            trajectory_path,
            target_means_path,
            comparisons_path,
            interpretation_path,
            manifest_path,
            figure_path.with_suffix(".svg"),
        ):
            artifact.add_file(str(path))
        logged = run.log_artifact(artifact, aliases=["no-training", "nested-attention"])
        logged.wait()
        source.model.to(device="cpu")
        del source, loaded
        gc.collect()
        torch.cuda.empty_cache()
        run.finish(exit_code=0)
    except BaseException:
        run.finish(exit_code=1)
        raise
