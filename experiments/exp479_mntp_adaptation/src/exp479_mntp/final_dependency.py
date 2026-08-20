"""Final-checkpoint nucleotide-dependency comparison for issue 479."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
from huggingface_hub import HfApi
from matplotlib.colors import PowerNorm
from pyfaidx import Fasta

from exp479_mntp.checkpoint_audit import (
    CHECKPOINT_REPOS,
    ModelPoint,
    _repo_files,
    load_point,
)
from exp479_mntp.nucleotide_dependency import LOCI, locus_window, orientation_dependency
from exp479_mntp.publishing import assert_budget_reserve
from exp479_mntp.vep import download_reference

POINTS = (
    ModelPoint(
        "transferred-step1000",
        "transferred_mntp",
        1000,
        "mntp",
        "hf",
        "Transferred MNTP",
    ),
    ModelPoint(
        "scratch-step1000",
        "scratch_mntp",
        1000,
        "mntp",
        "hf",
        "Scratch MNTP",
    ),
    ModelPoint(
        "clm-original-step1000",
        "clm_continuation",
        1000,
        "clm",
        "hf",
        "Continued CLM",
    ),
)
LOCUS_NAME = "tRNA_Arg_TCT"


def _region_values(matrix: np.ndarray, region: str) -> np.ndarray:
    """Return one directed context region without the diagonal."""

    if region == "past_context":
        indices = np.triu_indices(matrix.shape[0], k=1)
    elif region == "future_context":
        indices = np.tril_indices(matrix.shape[0], k=-1)
    else:
        raise ValueError(f"unknown dependency region {region}")
    return matrix[indices]


def _summary_rows(point: ModelPoint, matrix: np.ndarray) -> list[dict[str, object]]:
    """Summarize past- and future-context dependence in natural matrix units."""

    rows: list[dict[str, object]] = []
    for region in ("past_context", "future_context"):
        values = _region_values(matrix, region)
        rows.append(
            {
                "point_id": point.point_id,
                "arm": point.arm,
                "objective": point.objective,
                "step": point.step,
                "region": region,
                "mean_dependency": float(values.mean()),
                "p95_dependency": float(np.quantile(values, 0.95)),
                "maximum_dependency": float(values.max()),
                "nonzero_fraction": float(np.mean(values > 0)),
                "n_position_pairs": len(values),
            }
        )
    return rows


def _plot_maps(
    maps: list[tuple[ModelPoint, np.ndarray]],
    *,
    output_path: Path,
) -> None:
    """Plot the three final directed maps on one shared power-normalized scale."""

    maximum = max(float(matrix.max()) for _, matrix in maps)
    if maximum <= 0:
        raise RuntimeError("all final dependency maps are zero")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    image = None
    for axis, (point, matrix) in zip(axes, maps, strict=True):
        image = axis.imshow(
            matrix,
            origin="lower",
            cmap="viridis",
            norm=PowerNorm(gamma=0.45, vmin=0, vmax=maximum),
            interpolation="nearest",
            rasterized=True,
        )
        axis.plot((0, matrix.shape[0] - 1), (0, matrix.shape[0] - 1), color="white", lw=0.5)
        axis.set_title(point.plot_series)
        axis.set_xlabel("Readout position")
        axis.set_ylabel("Substitution position")
    assert image is not None
    figure.colorbar(image, ax=axes, label="L∞ change in A/C/G/T log probability")
    figure.suptitle(
        "tRNA-Arg-TCT directed nucleotide dependency at final checkpoints\n"
        "Shared scale; MNTP readouts are masked, CLM uses the causal next-token readout"
    )
    figure.savefig(output_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def _publish_wandb(
    output_dir: Path,
    summary: pd.DataFrame,
    checks: pd.DataFrame,
    *,
    batch_size: int,
) -> str | None:
    """Publish compact final-checkpoint dependencies to a direct W&B run."""

    if not os.getenv("WANDB_API_KEY"):
        return None
    run = wandb.init(
        project="marin",
        group="dna-exp479",
        name="dna-exp479-final-checkpoint-dependency",
        tags=["MNTP-479", "issue-479", "nucleotide-dependency", "final-checkpoints"],
        config={
            "locus": LOCUS_NAME,
            "orientation": "reference",
            "batch_size_including_baseline": batch_size,
            "paired_baseline_same_model_call": True,
            "checkpoints": [point.point_id for point in POINTS],
        },
    )
    run.log(
        {
            "dependency_summary": wandb.Table(dataframe=summary),
            "dependency_checks": wandb.Table(dataframe=checks),
            "final_checkpoint_dependency": wandb.Image(
                str(output_dir / "figures" / "final-checkpoint-dependency.png")
            ),
        }
    )
    run.summary["all_checks_passed"] = bool(checks["passed"].all())
    url = run.url
    run.finish(exit_code=0)
    return url


def run_final_checkpoint_dependency(
    *,
    artifact_dir: Path,
    output_dir: Path,
    hf_repo_id: str,
    batch_size: int,
) -> None:
    """Compare one directed dependency map at every trained arm's final checkpoint."""

    if not torch.cuda.is_available():
        raise RuntimeError("final dependency comparison requires the Lambda GH200")
    if hf_repo_id not in CHECKPOINT_REPOS:
        raise ValueError(f"unexpected publication repository {hf_repo_id}")
    if batch_size <= 1:
        raise ValueError("batch size must include one baseline and at least one substitution")

    output_dir.mkdir(parents=True, exist_ok=True)
    map_dir = output_dir / "maps"
    figure_dir = output_dir / "figures"
    map_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = artifact_dir / "final-dependency-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = _repo_files()
    reference = download_reference(artifact_dir / "reference")
    locus = next(locus for locus in LOCI if locus.name == LOCUS_NAME)
    with Fasta(reference, as_raw=True, rebuild=False) as genome:
        sequence, context_start = locus_window(genome, locus)

    maps: list[tuple[ModelPoint, np.ndarray]] = []
    rows: list[dict[str, object]] = []
    checks: list[dict[str, object]] = []
    for point in POINTS:
        assert_budget_reserve()
        print(f"dependency start: {point.point_id}", flush=True)
        arm = load_point(point, cache_dir=cache_dir, files=files)
        arm.model.to(device="cuda", dtype=torch.bfloat16).eval()
        attention_mode = "full" if point.objective == "mntp" else "causal"
        matrix = orientation_dependency(
            arm,
            sequence,
            batch_size=batch_size,
            attention_mode=attention_mode,
        )
        np.savez_compressed(map_dir / f"{point.point_id}.npz", directed=matrix)
        maps.append((point, matrix))
        point_rows = _summary_rows(point, matrix)
        rows.extend(point_rows)
        by_region = {row["region"]: row for row in point_rows}
        future_maximum = float(by_region["future_context"]["maximum_dependency"])
        past_maximum = float(by_region["past_context"]["maximum_dependency"])
        if point.objective == "clm":
            checks.append(
                {
                    "point_id": point.point_id,
                    "check": "causal_future_context_zero",
                    "observed": future_maximum,
                    "threshold": 1e-6,
                    "passed": future_maximum <= 1e-6,
                }
            )
        else:
            checks.extend(
                [
                    {
                        "point_id": point.point_id,
                        "check": "mntp_past_context_nonzero",
                        "observed": past_maximum,
                        "threshold": 0.0,
                        "passed": past_maximum > 0,
                    },
                    {
                        "point_id": point.point_id,
                        "check": "mntp_future_context_nonzero",
                        "observed": future_maximum,
                        "threshold": 0.0,
                        "passed": future_maximum > 0,
                    },
                ]
            )
        print(
            f"dependency complete: {point.point_id} "
            f"past_max={past_maximum:.6g} future_max={future_maximum:.6g}",
            flush=True,
        )
        del arm
        gc.collect()
        torch.cuda.empty_cache()

    summary = pd.DataFrame(rows)
    check_frame = pd.DataFrame(checks)
    summary.to_csv(output_dir / "dependency-summary.csv", index=False)
    check_frame.to_csv(output_dir / "dependency-checks.csv", index=False)
    _plot_maps(maps, output_path=figure_dir / "final-checkpoint-dependency")
    wandb_url = _publish_wandb(
        output_dir,
        summary,
        check_frame,
        batch_size=batch_size,
    )
    manifest = {
        "locus": LOCUS_NAME,
        "chrom": locus.chrom,
        "locus_start_zero_based": locus.start,
        "locus_end_zero_based_exclusive": locus.end,
        "context_start_zero_based": context_start,
        "context_end_zero_based_exclusive": context_start + len(sequence),
        "sequence_length": len(sequence),
        "orientation": "reference",
        "checkpoints": [point.point_id for point in POINTS],
        "paired_baseline_same_model_call": True,
        "batch_size_including_baseline": batch_size,
        "all_checks_passed": bool(check_frame["passed"].all()),
        "wandb_url": wandb_url,
        "publication_path": "evaluation/final-checkpoint-dependency",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    assert_budget_reserve()
    HfApi().upload_folder(
        folder_path=output_dir,
        path_in_repo="evaluation/final-checkpoint-dependency",
        repo_id=hf_repo_id,
        repo_type="model",
        commit_message="Upload exp479 final-checkpoint dependency comparison",
    )
    if not bool(check_frame["passed"].all()):
        failed = check_frame.loc[~check_frame["passed"], "check"].tolist()
        raise RuntimeError(f"final checkpoint dependency checks failed: {failed}")
