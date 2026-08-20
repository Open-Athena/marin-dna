"""Corrected VEP parity and nucleotide-dependency recheck for issue 479."""

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
from pyfaidx import Fasta

from exp479_mntp.checkpoint_audit import (
    CHECKPOINT_REPOS,
    SCORE_PARITY_TOLERANCE,
    ModelPoint,
    _repo_files,
    compare_existing_scores,
    evaluate_point,
    load_point,
    plot_dependency_panel,
    triangle_summary,
)
from exp479_mntp.nucleotide_dependency import (
    LOCI,
    locus_window,
    mean_symmetrize,
    orientation_dependency,
)
from exp479_mntp.publishing import assert_budget_reserve
from exp479_mntp.vep import (
    DATASETS,
    attach_reference_windows,
    download_reference,
    load_variant_frame,
    reverse_complement,
)


def _anchor_points() -> tuple[ModelPoint, ...]:
    return (
        ModelPoint(
            "source-clm-direct-step0000",
            "source_clm",
            0,
            "clm",
            "source",
            "Source CLM direct",
        ),
        ModelPoint(
            "transferred-step0000",
            "full_attention_no_adaptation",
            0,
            "mntp",
            "no_adaptation",
            "Transferred MNTP",
        ),
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
            "Continued CLM original",
        ),
    )


def _plot_causal_control(
    full_attention: np.ndarray,
    causal_attention: np.ndarray,
    output_path: Path,
) -> None:
    maximum = float(max(full_attention.max(), causal_attention.max()))
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    image = None
    for axis, matrix, title in zip(
        axes,
        (full_attention, causal_attention),
        ("Full attention", "Forced causal attention"),
        strict=True,
    ):
        image = axis.imshow(
            matrix,
            origin="lower",
            cmap="viridis",
            vmin=0,
            vmax=maximum,
            interpolation="nearest",
            rasterized=True,
        )
        axis.set_title(title)
        axis.set_xlabel("Readout position")
        axis.set_ylabel("Substitution position")
    assert image is not None
    figure.colorbar(image, ax=axes, label="L∞ change in A/C/G/T log probability")
    figure.suptitle("tRNA-Arg-TCT paired-baseline dependency control")
    figure.savefig(output_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def corrected_dependency_audit(
    *,
    artifact_dir: Path,
    output_dir: Path,
    cache_dir: Path,
    files: dict[str, set[str]],
    batch_size: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Recompute paired-baseline maps and gate causal leakage and batch stability."""

    final = ModelPoint(
        "transferred-final-corrected-nucdep",
        "transferred_mntp",
        1000,
        "mntp",
        "hf",
        "Transferred MNTP",
    )
    arm = load_point(final, cache_dir=cache_dir, files=files)
    arm.model.to(device="cuda", dtype=torch.bfloat16).eval()
    reference = download_reference(artifact_dir / "reference")
    map_dir = output_dir / "nucleotide-dependency"
    figure_dir = output_dir / "figures"
    map_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    maps: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    summaries: list[dict[str, object]] = []
    checks: list[dict[str, object]] = []
    with Fasta(reference, as_raw=True, rebuild=False) as genome:
        for locus in LOCI:
            assert_budget_reserve()
            sequence, _ = locus_window(genome, locus)
            forward = orientation_dependency(
                arm,
                sequence,
                batch_size=batch_size,
                attention_mode="full",
            )
            reverse = orientation_dependency(
                arm,
                reverse_complement(sequence),
                batch_size=batch_size,
                attention_mode="full",
            )[::-1, ::-1]
            combined = (mean_symmetrize(forward) + mean_symmetrize(reverse)) / 2
            maps.append((locus.name, forward, reverse, combined))
            np.savez_compressed(
                map_dir / f"{locus.name}.npz",
                forward_directed=forward,
                reverse_directed_forward_coordinates=reverse,
                fwd_rc=combined,
            )
            for label, matrix in (
                ("forward_directed", forward),
                ("reverse_directed_forward_coordinates", reverse),
                ("registered_fwd_rc", combined),
            ):
                summaries.append({"locus": locus.name, **triangle_summary(matrix, label)})

            if locus.name != "tRNA_Arg_TCT":
                continue
            half_batch = max(1, batch_size // 2)
            half_batch_forward = orientation_dependency(
                arm,
                sequence,
                batch_size=half_batch,
                attention_mode="full",
            )
            batch_difference = np.abs(forward - half_batch_forward)
            causal = orientation_dependency(
                arm,
                sequence,
                batch_size=batch_size,
                attention_mode="causal",
            )
            forbidden = np.tril(causal, k=-1)
            checks.extend(
                [
                    {
                        "comparison": "tRNA_paired_baseline_batch_stability",
                        "dataset": locus.name,
                        "score": "dependency",
                        "n_rows": int(forward.size),
                        "max_abs_error": float(batch_difference.max()),
                        "mean_abs_error": float(batch_difference.mean()),
                        "tolerance": SCORE_PARITY_TOLERANCE,
                        "passed": bool(float(batch_difference.max()) <= SCORE_PARITY_TOLERANCE),
                    },
                    {
                        "comparison": "tRNA_paired_baseline_causal_forbidden_triangle",
                        "dataset": locus.name,
                        "score": "dependency",
                        "n_rows": int(forbidden.size),
                        "max_abs_error": float(forbidden.max()),
                        "mean_abs_error": float(np.abs(forbidden).mean()),
                        "tolerance": 1e-6,
                        "passed": bool(float(forbidden.max()) <= 1e-6),
                    },
                ]
            )
            np.savez_compressed(
                map_dir / "tRNA_Arg_TCT-controls.npz",
                full_attention=forward,
                full_attention_half_batch=half_batch_forward,
                causal_attention=causal,
            )
            _plot_causal_control(
                forward,
                causal,
                figure_dir / "nucleotide-dependency-causal-control",
            )

    plot_dependency_panel(maps, figure_dir / "nucleotide-dependency-panel")
    del arm
    gc.collect()
    torch.cuda.empty_cache()
    return pd.DataFrame(summaries), checks


def _publish_wandb(
    output_dir: Path,
    metrics: pd.DataFrame,
    checks: pd.DataFrame,
    dependency_summary: pd.DataFrame,
    *,
    vep_batch_size: int,
    dependency_batch_size: int,
) -> str | None:
    if not os.getenv("WANDB_API_KEY"):
        return None
    run = wandb.init(
        project="marin",
        group="dna-exp479",
        name="dna-exp479-corrected-inference-recheck",
        tags=["MNTP-479", "issue-479", "inference-recheck", "paired-baseline"],
        config={
            "development_split": "odd autosomes and X",
            "vep_batch_size": vep_batch_size,
            "dependency_batch_size": dependency_batch_size,
            "score_parity_tolerance": SCORE_PARITY_TOLERANCE,
        },
    )
    run.log(
        {
            "anchor_auprc": wandb.Table(dataframe=metrics),
            "parity_checks": wandb.Table(dataframe=checks),
            "dependency_summary": wandb.Table(dataframe=dependency_summary),
            "nucleotide_dependency": wandb.Image(
                str(output_dir / "figures" / "nucleotide-dependency-panel.png")
            ),
            "causal_control": wandb.Image(
                str(output_dir / "figures" / "nucleotide-dependency-causal-control.png")
            ),
        }
    )
    run.summary["all_corrected_checks_passed"] = bool(checks["passed"].all())
    run.summary["maximum_parity_abs_error"] = float(checks["max_abs_error"].max())
    url = run.url
    run.finish(exit_code=0)
    return url


def run_inference_recheck(
    *,
    artifact_dir: Path,
    output_dir: Path,
    hf_repo_id: str,
    vep_batch_size: int,
    dependency_batch_size: int,
    n_bootstrap: int,
) -> None:
    """Recheck original-batch VEP parity and corrected nucleotide dependencies."""

    if not torch.cuda.is_available():
        raise RuntimeError("inference recheck requires the Lambda GH200")
    if hf_repo_id not in CHECKPOINT_REPOS:
        raise ValueError(f"unexpected publication repository {hf_repo_id}")
    if vep_batch_size <= 0 or dependency_batch_size <= 1:
        raise ValueError("recheck batch sizes must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = artifact_dir / "inference-recheck-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = _repo_files()
    reference = download_reference(artifact_dir / "reference")
    frames = {
        spec.name: attach_reference_windows(load_variant_frame(spec), reference)
        for spec in DATASETS
    }

    metric_rows: list[dict[str, object]] = []
    points = _anchor_points()
    for point in points:
        assert_budget_reserve()
        arm = load_point(point, cache_dir=cache_dir, files=files)
        arm.model.to(device="cuda", dtype=torch.bfloat16).eval()
        metric_rows.extend(
            evaluate_point(
                point,
                arm,
                frames,
                output_dir,
                batch_size=vep_batch_size,
                n_bootstrap=n_bootstrap,
            )
        )
        del arm
        gc.collect()
        torch.cuda.empty_cache()
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "anchor-auprc.csv", index=False)

    checks: list[dict[str, object]] = []
    for point_id, prefix in (
        ("source-clm-direct-step0000", "source_clm"),
        ("transferred-step0000", "full_attention_no_adaptation"),
        ("transferred-step1000", "transferred_mntp"),
        ("scratch-step1000", "scratch_mntp"),
        ("clm-original-step1000", "clm_continuation"),
    ):
        checks.extend(
            compare_existing_scores(
                output_dir=output_dir,
                point_id=point_id,
                existing_prefix=prefix,
                cache_dir=cache_dir,
                files=files,
            )
        )

    dependency_summary, dependency_checks = corrected_dependency_audit(
        artifact_dir=artifact_dir,
        output_dir=output_dir,
        cache_dir=cache_dir,
        files=files,
        batch_size=dependency_batch_size,
    )
    checks.extend(dependency_checks)
    check_frame = pd.DataFrame(checks)
    check_frame.to_csv(output_dir / "corrected-parity-checks.csv", index=False)
    dependency_summary.to_csv(output_dir / "nucleotide-dependency-triangles.csv", index=False)

    wandb_url = _publish_wandb(
        output_dir,
        metrics,
        check_frame,
        dependency_summary,
        vep_batch_size=vep_batch_size,
        dependency_batch_size=dependency_batch_size,
    )
    manifest = {
        "development_split": "odd autosomes and X only",
        "vep_batch_size": vep_batch_size,
        "dependency_batch_size": dependency_batch_size,
        "n_bootstrap": n_bootstrap,
        "paired_dependency_baseline": True,
        "all_corrected_checks_passed": bool(check_frame["passed"].all()),
        "wandb_url": wandb_url,
        "publication_path": "evaluation/inference-recheck",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    assert_budget_reserve()
    HfApi().upload_folder(
        folder_path=output_dir,
        path_in_repo="evaluation/inference-recheck",
        repo_id=hf_repo_id,
        repo_type="model",
        commit_message="Upload exp479 corrected inference recheck",
    )
    if not bool(check_frame["passed"].all()):
        failed = check_frame.loc[~check_frame["passed"], "comparison"].tolist()
        raise RuntimeError(f"corrected inference recheck failed: {failed}")
