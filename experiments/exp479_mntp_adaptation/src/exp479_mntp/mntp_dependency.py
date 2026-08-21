"""Final-checkpoint LDLR dependency map for the corrected MNTP run."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
import wandb
from pyfaidx import Fasta

from exp479_mntp.checkpoint_audit import _loaded_from_hf
from exp479_mntp.config import EXPERIMENT_TAGS, NUCLEOTIDE_LENGTH, WANDB_PROJECT
from exp479_mntp.mntp_longrun import _dependency_summary, plot_mntp_dependency
from exp479_mntp.nucleotide_dependency import LOCI, Locus, locus_window, orientation_dependency
from exp479_mntp.publishing import assert_budget_reserve
from exp479_mntp.vep import download_reference

WANDB_ENTITY = "gonzalobenegas"
LDLR_LOCUS_NAME = "LDLR"
SOURCE_MODEL_ARTIFACT = (
    f"{WANDB_ENTITY}/{WANDB_PROJECT}/dna-exp479-mntp-longrun-corrected-step-1000:v0"
)
LDLR_WANDB_RUN_NAME = "dna-exp479-mntp-longrun-corrected-step-1000-ldlr-dependency"
LDLR_EVALUATION_ARTIFACT = LDLR_WANDB_RUN_NAME


def selected_locus() -> Locus:
    """Return the browser-default LDLR locus with its registered coordinates."""

    locus = next(item for item in LOCI if item.name == LDLR_LOCUS_NAME)
    if locus != Locus("LDLR", "19", 11_089_299, 11_089_425, "+"):
        raise RuntimeError(f"LDLR registration changed unexpectedly: {locus}")
    return locus


def dependency_checks(matrix: np.ndarray) -> dict[str, float | int | bool]:
    """Validate shape, finiteness, diagonal, and bilateral full-attention signal."""

    expected_shape = (NUCLEOTIDE_LENGTH, NUCLEOTIDE_LENGTH)
    if matrix.shape != expected_shape:
        raise RuntimeError(f"LDLR dependency shape {matrix.shape} differs from {expected_shape}")
    finite = bool(np.isfinite(matrix).all())
    diagonal_maximum = float(np.abs(np.diag(matrix)).max())
    past_maximum = float(matrix[np.triu_indices(NUCLEOTIDE_LENGTH, k=1)].max())
    future_maximum = float(matrix[np.tril_indices(NUCLEOTIDE_LENGTH, k=-1)].max())
    passed = finite and diagonal_maximum == 0 and past_maximum > 0 and future_maximum > 0
    return {
        "passed": passed,
        "finite": finite,
        "matrix_rows": int(matrix.shape[0]),
        "matrix_columns": int(matrix.shape[1]),
        "diagonal_absolute_maximum": diagonal_maximum,
        "past_context_maximum": past_maximum,
        "future_context_maximum": future_maximum,
    }


def run_mntp_dependency(
    *,
    artifact_dir: Path,
    output_dir: Path,
    batch_size: int,
) -> None:
    """Load the retained final model and publish a compact LDLR dependency result."""

    if not torch.cuda.is_available():
        raise RuntimeError("final MNTP dependency evaluation requires the Lambda GH200")
    if batch_size <= 1:
        raise ValueError("dependency batch size must leave room for a paired baseline")

    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=WANDB_PROJECT,
        group="dna-exp479-mntp-longrun-corrected",
        name=LDLR_WANDB_RUN_NAME,
        tags=[*EXPERIMENT_TAGS, "ldlr", "nucleotide-dependency", "final-checkpoint"],
        config={
            "source_model_artifact": SOURCE_MODEL_ARTIFACT,
            "locus": LDLR_LOCUS_NAME,
            "batch_size_including_baseline": batch_size,
            "paired_baseline_same_model_call": True,
            "attention_mode": "full",
            "orientation": "reference",
        },
    )
    if run is None:
        raise RuntimeError("W&B did not create an LDLR evaluation run")

    try:
        assert_budget_reserve()
        source_artifact = run.use_artifact(SOURCE_MODEL_ARTIFACT, type="model")
        checkpoint_root = Path(source_artifact.download(root=artifact_dir / "retained-step-1000"))
        loaded = _loaded_from_hf(checkpoint_root / "hf", "mntp")
        if loaded.mask_token_id is None:
            raise RuntimeError("retained step-1,000 MNTP checkpoint lacks a MASK token")
        loaded.model.to(device="cuda", dtype=torch.bfloat16).eval()

        reference = download_reference(artifact_dir / "reference")
        locus = selected_locus()
        with Fasta(reference, as_raw=True, rebuild=False) as genome:
            sequence, context_start = locus_window(genome, locus)
        matrix = orientation_dependency(
            loaded,
            sequence,
            batch_size=batch_size,
            attention_mode="full",
        )
        checks = dependency_checks(matrix)
        if not bool(checks["passed"]):
            raise RuntimeError(f"LDLR dependency invariants failed: {checks}")

        matrix_path = output_dir / "ldlr-nucleotide-dependency.npz"
        summary_path = output_dir / "ldlr-nucleotide-dependency-summary.csv"
        checks_path = output_dir / "ldlr-nucleotide-dependency-checks.json"
        figure_path = figures / "ldlr-nucleotide-dependency"
        np.savez_compressed(matrix_path, directed=matrix)
        summary = _dependency_summary(matrix)
        summary.to_csv(summary_path, index=False)
        checks_path.write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
        plot_mntp_dependency(matrix, figure_path)

        manifest = {
            "status": "completed",
            "source_model_artifact": SOURCE_MODEL_ARTIFACT,
            "source_artifact_id": source_artifact.id,
            "locus": locus.name,
            "chrom": locus.chrom,
            "locus_start_zero_based": locus.start,
            "locus_end_zero_based_exclusive": locus.end,
            "strand": locus.strand,
            "context_start_zero_based": context_start,
            "context_end_zero_based_exclusive": context_start + len(sequence),
            "sequence_length": len(sequence),
            "orientation": "reference",
            "attention_mode": "full",
            "paired_baseline_same_model_call": True,
            "batch_size_including_baseline": batch_size,
            "mask_token_id": loaded.mask_token_id,
            "canonical_token_ids": list(loaded.canonical_ids),
            "checks": checks,
            "elapsed_seconds": time.time() - started,
            "checkpoint_deletion": "not performed",
            "hugging_face_upload": "not performed",
        }
        manifest_path = output_dir / "ldlr-nucleotide-dependency-manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        run.log(
            {
                "ldlr_dependency/summary": wandb.Table(dataframe=summary),
                "ldlr_dependency/map": wandb.Image(str(figure_path.with_suffix(".png"))),
            }
        )
        for key, value in checks.items():
            run.summary[f"ldlr_dependency/{key}"] = value
        result_artifact = wandb.Artifact(LDLR_EVALUATION_ARTIFACT, type="evaluation")
        for path in (
            matrix_path,
            summary_path,
            checks_path,
            manifest_path,
            figure_path.with_suffix(".svg"),
        ):
            result_artifact.add_file(str(path))
        logged = run.log_artifact(result_artifact, aliases=["selected-ldlr", "step-1000"])
        logged.wait()
        run.finish(exit_code=0)
        del loaded
        gc.collect()
        torch.cuda.empty_cache()
    except BaseException:
        run.finish(exit_code=1)
        raise
