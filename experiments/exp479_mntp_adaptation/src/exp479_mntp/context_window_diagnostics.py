"""Post-hoc context-ablation and window-shift VEP diagnostics for exp479."""

from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import HfApi

from exp479_mntp.config import MODEL_ID, MODEL_REVISION, NUCLEOTIDE_LENGTH
from exp479_mntp.publishing import assert_budget_reserve
from exp479_mntp.vep import (
    DATASETS,
    DEVELOPMENT_CHROMS,
    REFERENCE_FASTA,
    REFERENCE_REPO,
    REFERENCE_REVISION,
    ArmSpec,
    DatasetSpec,
    attach_reference_windows,
    download_reference,
    load_arm,
    load_variant_frame,
    score_strand,
)
from exp479_mntp.vep_metrics import matched_metrics, sge_metrics


@dataclass(frozen=True)
class DiagnosticCondition:
    """One deterministic context perturbation and its variant position."""

    name: str
    variant_index: int
    ablated_flank: str | None = None


CENTER = NUCLEOTIDE_LENGTH // 2
WINDOW_SHIFT = 64
CONDITIONS = (
    DiagnosticCondition("centered_full", CENTER),
    DiagnosticCondition("left_context_ablated", CENTER, "left"),
    DiagnosticCondition("right_context_ablated", CENTER, "right"),
    DiagnosticCondition("window_shift_upstream_64", CENTER + WINDOW_SHIFT),
    DiagnosticCondition("window_shift_downstream_64", CENTER - WINDOW_SHIFT),
)


def ablate_context(sequence: str, flank: str) -> str:
    """Replace one complete flank with the source tokenizer's unknown-base token."""

    if len(sequence) != NUCLEOTIDE_LENGTH:
        raise ValueError("context ablation requires a 255-base sequence")
    if flank == "left":
        return "N" * CENTER + sequence[CENTER:]
    if flank == "right":
        return sequence[: CENTER + 1] + "N" * (NUCLEOTIDE_LENGTH - CENTER - 1)
    raise ValueError(f"unsupported context flank {flank!r}")


def diagnostic_frame(
    frame: pd.DataFrame,
    reference: Path,
    condition: DiagnosticCondition,
) -> pd.DataFrame:
    """Build one condition while preserving labels and 0-based/half-open reference access."""

    result = attach_reference_windows(
        frame,
        reference,
        variant_index=condition.variant_index,
    )
    if condition.ablated_flank is not None:
        result["sequence"] = [
            ablate_context(sequence, condition.ablated_flank) for sequence in result["sequence"]
        ]
    return result


def stability_rows(
    raw_scores: pd.DataFrame,
    *,
    dataset: str,
) -> list[dict[str, object]]:
    """Compare each diagnostic score vector with the centered full-context baseline."""

    baseline = raw_scores["centered_full"]
    rows: list[dict[str, object]] = []
    for condition in CONDITIONS:
        values = raw_scores[condition.name]
        rows.append(
            {
                "dataset": dataset,
                "condition": condition.name,
                "spearman_vs_centered": float(values.corr(baseline, method="spearman")),
                "pearson_vs_centered": float(values.corr(baseline, method="pearson")),
                "mean_absolute_llr_change": float((values - baseline).abs().mean()),
                "median_absolute_llr_change": float((values - baseline).abs().median()),
                "n_rows": len(values),
            }
        )
    return rows


def _metrics(
    spec: DatasetSpec,
    variants: pd.DataFrame,
    protocol_scores: pd.DataFrame,
    *,
    n_bootstrap: int,
) -> pd.DataFrame:
    if spec.evaluation == "matched":
        return matched_metrics(
            variants,
            protocol_scores,
            n_bootstrap=n_bootstrap,
            seed=0,
        )
    return sge_metrics(
        variants,
        protocol_scores,
        n_bootstrap=n_bootstrap,
        seed=0,
    )


def run_context_window_diagnostics(
    *,
    artifact_dir: Path,
    output_dir: Path,
    hf_repo_id: str,
    batch_size: int,
    n_bootstrap: int = 1_000,
) -> None:
    """Evaluate final transferred MNTP under registered context/window perturbations."""

    if not torch.cuda.is_available():
        raise RuntimeError("exp479 context/window diagnostics require the Lambda GH200")
    if batch_size <= 0:
        raise ValueError("diagnostic batch size must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = download_reference(artifact_dir / "reference")
    source_frames = {spec.name: load_variant_frame(spec) for spec in DATASETS}

    arm_spec = ArmSpec("transferred_mntp", "mntp")
    arm = load_arm(arm_spec, artifact_dir, hf_repo_id)
    arm.model.to(device="cuda", dtype=torch.bfloat16).eval()

    runtime_rows: list[dict[str, object]] = []
    all_stability_rows: list[dict[str, object]] = []
    for spec in DATASETS:
        assert_budget_reserve()
        raw_scores = pd.DataFrame(index=source_frames[spec.name].index)
        centered_frame: pd.DataFrame | None = None
        for condition in CONDITIONS:
            frame = diagnostic_frame(source_frames[spec.name], reference, condition)
            if centered_frame is None:
                centered_frame = frame
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            values = score_strand(
                arm,
                frame,
                objective="mntp",
                strand="fwd",
                batch_size=batch_size,
                variant_index=condition.variant_index,
            )
            elapsed = time.perf_counter() - started
            raw_scores[condition.name] = values
            runtime_rows.append(
                {
                    "dataset": spec.name,
                    "condition": condition.name,
                    "rows": len(frame),
                    "seconds": elapsed,
                    "variants_per_second": len(frame) / elapsed,
                    "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                }
            )
        if centered_frame is None:
            raise RuntimeError(f"{spec.name} produced no diagnostic frame")

        protocol_scores = pd.DataFrame(
            {
                condition.name: (
                    -raw_scores[condition.name]
                    if spec.protocol == "minus_llr"
                    else raw_scores[condition.name].abs()
                )
                for condition in CONDITIONS
            }
        )
        public_columns = [column for column in centered_frame.columns if column != "sequence"]
        pd.concat([centered_frame[public_columns], protocol_scores], axis=1).to_parquet(
            output_dir / f"{spec.name}.scores.parquet",
            index=False,
        )
        _metrics(
            spec,
            centered_frame,
            protocol_scores,
            n_bootstrap=n_bootstrap,
        ).to_parquet(output_dir / f"{spec.name}.metrics.parquet", index=False)
        all_stability_rows.extend(stability_rows(raw_scores, dataset=spec.name))

    del arm
    gc.collect()
    torch.cuda.empty_cache()

    pd.DataFrame(runtime_rows).to_parquet(output_dir / "runtime.parquet", index=False)
    pd.DataFrame(all_stability_rows).to_parquet(output_dir / "stability.parquet", index=False)
    manifest = {
        "source_model": f"{MODEL_ID}@{MODEL_REVISION}",
        "checkpoint": "transferred_mntp/step-1000",
        "reference": f"{REFERENCE_REPO}@{REFERENCE_REVISION}/{REFERENCE_FASTA}",
        "split": "train",
        "allowed_chromosomes": sorted(DEVELOPMENT_CHROMS),
        "conditions": [
            {
                "name": condition.name,
                "variant_index": condition.variant_index,
                "ablated_flank": condition.ablated_flank,
            }
            for condition in CONDITIONS
        ],
        "window_shift_bases": WINDOW_SHIFT,
        "context_ablation_token": "[UNK] via nucleotide N",
        "parameterization": (
            "Registered diagnostic with post-hoc fixed parameterization; "
            "not a model-selection gate."
        ),
        "batch_size": batch_size,
        "n_bootstrap": n_bootstrap,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    HfApi().upload_folder(
        folder_path=output_dir,
        path_in_repo="evaluation/context-window",
        repo_id=hf_repo_id,
        repo_type="model",
        commit_message="Upload exp479 context and window diagnostics",
    )
