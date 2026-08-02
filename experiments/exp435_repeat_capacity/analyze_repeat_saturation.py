"""Analyze the frozen repeat-feature saturation intervention."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import polars as pl
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ISSUE, sha256_file, write_json
from motif_context_common import bh_adjust
from saturation_common import (
    MIN_CONTEXTS,
    MUTATIONS_PER_CONTEXT,
    OFFSETS,
    RUN_ID,
    VIEW_KEYS,
)

ALPHA = 0.05
TEST_FAMILIES = ("motif_loss", "motif_specificity")
SUBSTITUTIONS = tuple(
    f"{reference}>{alternate}"
    for reference in "ACGT"
    for alternate in "ACGT"
    if reference != alternate
)


def validate_extraction(
    root: Path,
) -> tuple[dict[str, Any], pl.DataFrame, pl.DataFrame]:
    manifest_path = root / "manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["issue"] == ISSUE and manifest["run_id"] == RUN_ID
    assert manifest["analysis_status"] == "post_hoc_repeat_motif_saturation"
    for name, expected in manifest["artifacts"].items():
        path = root / name
        assert path.is_file() and path.stat().st_size == expected["bytes"]
        assert sha256_file(path) == expected["sha256"]
    contexts = pl.read_parquet(root / "contexts.parquet")
    responses = pl.read_parquet(root / "mutation_responses.parquet")
    assert contexts.height == len(VIEW_KEYS) * 64
    assert responses.height == contexts.height * MUTATIONS_PER_CONTEXT
    assert set(
        zip(
            contexts["block"].to_list(),
            contexts["feature_id"].to_list(),
            contexts["orientation"].to_list(),
            strict=True,
        )
    ) == set(VIEW_KEYS)
    assert responses.group_by("saturation_context_id").len()[
        "len"
    ].unique().to_list() == [MUTATIONS_PER_CONTEXT]
    assert set(responses["model_offset"].unique()) == set(OFFSETS)
    return manifest, contexts, responses


def build_context_effects(responses: pl.DataFrame) -> pl.DataFrame:
    """Collapse correlated edits to one inferential value per context."""

    keys = ["block", "feature_id", "orientation", "saturation_context_id"]
    motif = (
        responses.filter(pl.col("motif_loss"))
        .group_by(*keys)
        .agg(
            pl.len().alias("motif_edits"),
            pl.col("delta").mean().alias("mean_motif_delta"),
            pl.col("delta").median().alias("median_motif_delta"),
            pl.col("relative_delta").mean().alias("mean_motif_relative_delta"),
            pl.col("relative_delta").median().alias("median_motif_relative_delta"),
            pl.col("thresholded_to_zero").mean().alias("motif_zero_fraction"),
            pl.col("net_kmers_lost").mean().alias("mean_net_kmers_lost"),
        )
    )
    neutral = (
        responses.filter(pl.col("neutral"))
        .group_by(*keys, "model_ref", "model_alt")
        .agg(
            pl.len().alias("neutral_edits_for_substitution"),
            pl.col("delta").mean().alias("neutral_mean_delta"),
            pl.col("relative_delta").mean().alias("neutral_mean_relative_delta"),
        )
    )
    matched = (
        responses.filter(pl.col("motif_loss"))
        .join(
            neutral,
            on=[*keys, "model_ref", "model_alt"],
            how="inner",
            validate="m:1",
        )
        .with_columns(
            (pl.col("delta") - pl.col("neutral_mean_delta")).alias(
                "specificity_contrast"
            ),
            (pl.col("relative_delta") - pl.col("neutral_mean_relative_delta")).alias(
                "relative_specificity_contrast"
            ),
        )
        .group_by(*keys)
        .agg(
            pl.len().alias("matched_motif_edits"),
            pl.col("specificity_contrast").mean(),
            pl.col("relative_specificity_contrast").mean(),
            pl.col("neutral_edits_for_substitution")
            .sum()
            .alias("available_neutral_edits"),
        )
    )
    result = motif.join(matched, on=keys, how="left", validate="1:1").sort(*keys)
    assert result.filter(
        ~pl.col("mean_motif_delta").is_finite()
        | ~pl.col("mean_motif_relative_delta").is_finite()
    ).is_empty()
    return result


def one_sided_p_values(values: np.ndarray) -> tuple[float, float]:
    """Return one-sided t and signed-rank p-values for values below zero."""

    assert values.ndim == 1 and values.size >= MIN_CONTEXTS
    assert np.isfinite(values).all()
    if np.all(values == 0):
        return 1.0, 1.0
    t_result = stats.ttest_1samp(values, 0.0, alternative="less")
    rank_result = stats.wilcoxon(values, alternative="less")
    assert np.isfinite(t_result.pvalue) and np.isfinite(rank_result.pvalue)
    return float(t_result.pvalue), float(rank_result.pvalue)


def build_planned_tests(context_effects: pl.DataFrame) -> pl.DataFrame:
    """Run the two preregistered context-level test families."""

    rows: list[dict[str, Any]] = []
    for block, feature_id, orientation in VIEW_KEYS:
        current = context_effects.filter(
            (pl.col("block") == block)
            & (pl.col("feature_id") == feature_id)
            & (pl.col("orientation") == orientation)
        )
        for family, column in (
            ("motif_loss", "mean_motif_delta"),
            ("motif_specificity", "specificity_contrast"),
        ):
            values = current[column].drop_nulls().to_numpy()
            enough = values.size >= MIN_CONTEXTS
            if enough:
                t_p, rank_p = one_sided_p_values(values)
                mean = float(values.mean())
                median = float(np.median(values))
                standard_deviation = float(values.std(ddof=1))
                standardized_mean = (
                    mean / standard_deviation if standard_deviation > 0 else None
                )
            else:
                t_p = rank_p = mean = median = standard_deviation = None
                standardized_mean = None
            rows.append(
                {
                    "block": block,
                    "feature_id": feature_id,
                    "orientation": orientation,
                    "family": family,
                    "value_column": column,
                    "n_contexts": values.size,
                    "minimum_contexts_met": enough,
                    "mean": mean,
                    "median": median,
                    "standard_deviation": standard_deviation,
                    "standardized_mean": standardized_mean,
                    "t_p": t_p,
                    "rank_p": rank_p,
                }
            )
    tests = pl.DataFrame(rows).sort("block", "family", "feature_id", "orientation")
    adjusted_frames: list[pl.DataFrame] = []
    for block in sorted({key[0] for key in VIEW_KEYS}):
        for family in TEST_FAMILIES:
            group = tests.filter(
                (pl.col("block") == block) & (pl.col("family") == family)
            )
            t_values = group["t_p"].to_list()
            rank_values = group["rank_p"].to_list()
            t_q: list[float | None] = [None] * group.height
            rank_q: list[float | None] = [None] * group.height
            t_indices = [
                index for index, value in enumerate(t_values) if value is not None
            ]
            rank_indices = [
                index for index, value in enumerate(rank_values) if value is not None
            ]
            if t_indices:
                adjusted = bh_adjust(np.array([t_values[index] for index in t_indices]))
                for index, value in zip(t_indices, adjusted, strict=True):
                    t_q[index] = float(value)
            if rank_indices:
                adjusted = bh_adjust(
                    np.array([rank_values[index] for index in rank_indices])
                )
                for index, value in zip(rank_indices, adjusted, strict=True):
                    rank_q[index] = float(value)
            adjusted_frames.append(
                group.with_columns(
                    pl.Series("t_q", t_q, dtype=pl.Float64),
                    pl.Series("rank_q", rank_q, dtype=pl.Float64),
                )
            )
    result = pl.concat(adjusted_frames).sort(
        "block", "feature_id", "orientation", "family"
    )
    assert result.height == len(VIEW_KEYS) * len(TEST_FAMILIES)
    return result


def build_view_summary(
    context_effects: pl.DataFrame, tests: pl.DataFrame
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for block, feature_id, orientation in VIEW_KEYS:
        current = context_effects.filter(
            (pl.col("block") == block)
            & (pl.col("feature_id") == feature_id)
            & (pl.col("orientation") == orientation)
        )
        loss = tests.filter(
            (pl.col("block") == block)
            & (pl.col("feature_id") == feature_id)
            & (pl.col("orientation") == orientation)
            & (pl.col("family") == "motif_loss")
        ).row(0, named=True)
        specificity = tests.filter(
            (pl.col("block") == block)
            & (pl.col("feature_id") == feature_id)
            & (pl.col("orientation") == orientation)
            & (pl.col("family") == "motif_specificity")
        ).row(0, named=True)
        loss_supported = bool(
            loss["minimum_contexts_met"]
            and loss["mean"] < 0
            and loss["median"] < 0
            and loss["t_q"] < ALPHA
            and loss["rank_q"] < ALPHA
        )
        specificity_supported = bool(
            specificity["minimum_contexts_met"]
            and specificity["mean"] < 0
            and specificity["median"] < 0
            and specificity["t_q"] < ALPHA
            and specificity["rank_q"] < ALPHA
        )
        rows.append(
            {
                "block": block,
                "feature_id": feature_id,
                "orientation": orientation,
                "contexts_with_motif_loss": current.height,
                "contexts_with_matched_neutral": int(
                    current["specificity_contrast"].is_not_null().sum()
                ),
                "median_context_motif_delta": (
                    float(current["mean_motif_delta"].median())
                    if current.height
                    else None
                ),
                "median_context_relative_motif_delta": (
                    float(current["mean_motif_relative_delta"].median())
                    if current.height
                    else None
                ),
                "mean_motif_zero_fraction": (
                    float(current["motif_zero_fraction"].mean())
                    if current.height
                    else None
                ),
                "motif_loss_supported": loss_supported,
                "motif_specificity_supported": loss_supported and specificity_supported,
                "loss_t_q": loss["t_q"],
                "loss_rank_q": loss["rank_q"],
                "specificity_t_q": specificity["t_q"],
                "specificity_rank_q": specificity["rank_q"],
            }
        )
    return pl.DataFrame(rows).sort("block", "feature_id", "orientation")


def build_feature_summary(view_summary: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for block, feature_id in sorted({(key[0], key[1]) for key in VIEW_KEYS}):
        current = view_summary.filter(
            (pl.col("block") == block) & (pl.col("feature_id") == feature_id)
        ).sort("orientation")
        assert current.height == 2
        by_orientation = {
            str(row["orientation"]): row for row in current.iter_rows(named=True)
        }
        rows.append(
            {
                "block": block,
                "feature_id": feature_id,
                "forward_motif_loss_supported": by_orientation["forward"][
                    "motif_loss_supported"
                ],
                "reverse_complement_motif_loss_supported": by_orientation[
                    "reverse_complement"
                ]["motif_loss_supported"],
                "strand_stable_motif_loss": all(
                    bool(row["motif_loss_supported"]) for row in by_orientation.values()
                ),
                "forward_motif_specificity_supported": by_orientation["forward"][
                    "motif_specificity_supported"
                ],
                "reverse_complement_motif_specificity_supported": by_orientation[
                    "reverse_complement"
                ]["motif_specificity_supported"],
                "strand_stable_motif_specificity": all(
                    bool(row["motif_specificity_supported"])
                    for row in by_orientation.values()
                ),
            }
        )
    return pl.DataFrame(rows).sort("block", "feature_id")


def build_response_profiles(responses: pl.DataFrame) -> pl.DataFrame:
    return (
        responses.with_columns(
            pl.concat_str("model_ref", pl.lit(">"), "model_alt").alias("substitution")
        )
        .group_by("block", "feature_id", "orientation", "model_offset", "substitution")
        .agg(
            pl.len().alias("n"),
            pl.col("saturation_context_id").n_unique().alias("n_contexts"),
            pl.col("delta").mean().alias("mean_delta"),
            pl.col("delta").median().alias("median_delta"),
            pl.col("relative_delta").mean().alias("mean_relative_delta"),
            pl.col("relative_delta").median().alias("median_relative_delta"),
            pl.col("motif_loss").mean().alias("motif_loss_fraction"),
            pl.col("thresholded_to_zero").mean().alias("thresholded_to_zero_fraction"),
        )
        .sort("block", "feature_id", "orientation", "model_offset", "substitution")
    )


def plot_response_heatmaps(profiles: pl.DataFrame, output_dir: Path) -> list[Path]:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir()
    paths: list[Path] = []
    for block, feature_id in sorted({(key[0], key[1]) for key in VIEW_KEYS}):
        current = profiles.filter(
            (pl.col("block") == block) & (pl.col("feature_id") == feature_id)
        )
        matrices: list[np.ndarray] = []
        for orientation in ("forward", "reverse_complement"):
            view = current.filter(pl.col("orientation") == orientation)
            matrix = np.full((len(SUBSTITUTIONS), len(OFFSETS)), np.nan)
            substitution_index = {
                item: index for index, item in enumerate(SUBSTITUTIONS)
            }
            offset_index = {item: index for index, item in enumerate(OFFSETS)}
            for row in view.iter_rows(named=True):
                matrix[
                    substitution_index[str(row["substitution"])],
                    offset_index[int(row["model_offset"])],
                ] = float(row["mean_relative_delta"])
            matrices.append(matrix)
        finite = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices])
        assert finite.size
        limit = max(float(np.quantile(np.abs(finite), 0.98)), 0.05)
        figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
        image = None
        for axis, orientation, matrix in zip(
            axes, ("forward", "reverse complement"), matrices, strict=True
        ):
            image = axis.imshow(
                matrix,
                aspect="auto",
                interpolation="nearest",
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                extent=(
                    OFFSETS[0] - 0.5,
                    OFFSETS[-1] + 0.5,
                    len(SUBSTITUTIONS) - 0.5,
                    -0.5,
                ),
            )
            axis.axvline(0, color="black", linewidth=0.8, alpha=0.6)
            axis.set_title(orientation)
            axis.set_xlabel("Model-relative offset (bp)")
            axis.set_xticks([-30, -15, 0, 15, 30])
        axes[0].set_yticks(range(len(SUBSTITUTIONS)), labels=SUBSTITUTIONS)
        axes[0].set_ylabel("Substitution")
        assert image is not None
        colorbar = figure.colorbar(image, ax=axes, shrink=0.88, pad=0.02)
        colorbar.set_label("Mean relative SAE activation change")
        figure.suptitle(f"Block {block}, feature {feature_id}: single-base saturation")
        figure.subplots_adjust(
            left=0.08, right=0.89, bottom=0.12, top=0.86, wspace=0.08
        )
        stem = f"block{block:02d}-feature{feature_id:05d}-saturation"
        for suffix in ("svg", "png"):
            path = plot_dir / f"{stem}.{suffix}"
            figure.savefig(path, dpi=180 if suffix == "png" else None)
            paths.append(path)
        plt.close(figure)
    return paths


def analyze(extraction_dir: Path, output_dir: Path) -> dict[str, Any]:
    assert extraction_dir.is_dir() and not output_dir.exists()
    extraction_manifest, _, responses = validate_extraction(extraction_dir)
    output_dir.mkdir(parents=True)
    context_effects = build_context_effects(responses)
    tests = build_planned_tests(context_effects)
    view_summary = build_view_summary(context_effects, tests)
    feature_summary = build_feature_summary(view_summary)
    profiles = build_response_profiles(responses)

    tables = {
        "context_effects.parquet": context_effects,
        "planned_tests.parquet": tests,
        "view_summary.parquet": view_summary,
        "feature_summary.parquet": feature_summary,
        "response_profiles.parquet": profiles,
    }
    for name, frame in tables.items():
        frame.write_parquet(output_dir / name, compression="zstd")
    plot_paths = plot_response_heatmaps(profiles, output_dir)
    loss_views = int(view_summary["motif_loss_supported"].sum())
    specific_views = int(view_summary["motif_specificity_supported"].sum())
    strand_loss = int(feature_summary["strand_stable_motif_loss"].sum())
    strand_specific = int(feature_summary["strand_stable_motif_specificity"].sum())
    result: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "issue": ISSUE,
        "run_id": RUN_ID,
        "analysis_status": "post_hoc_repeat_motif_saturation",
        "extraction_manifest_sha256": sha256_file(extraction_dir / "manifest.json"),
        "extraction_commit": extraction_manifest["experiment_commit"],
        "multiple_testing": (
            "BH separately within layer, test family, and test type across the "
            "prespecified feature-by-orientation views"
        ),
        "minimum_contexts": MIN_CONTEXTS,
        "success_criteria": {
            "motif_loss": "negative mean+median and t_q<0.05 and rank_q<0.05",
            "motif_specificity": "motif_loss plus negative paired mean+median and both q<0.05",
            "strand_stable": "criterion holds separately in forward and reverse-complement",
        },
        "summary": {
            "views": len(VIEW_KEYS),
            "features": len(VIEW_KEYS) // 2,
            "motif_loss_supported_views": loss_views,
            "motif_specificity_supported_views": specific_views,
            "strand_stable_motif_loss_features": strand_loss,
            "strand_stable_motif_specificity_features": strand_specific,
        },
    }
    result_path = output_dir / "results.json"
    write_json(result_path, result)
    results_md = output_dir / "RESULTS.md"
    results_md.write_text(
        "# Repeat-feature saturation intervention\n\n"
        f"- Motif-loss supported views: {loss_views}/{len(VIEW_KEYS)}\n"
        f"- Motif-specific supported views: {specific_views}/{len(VIEW_KEYS)}\n"
        f"- Strand-stable motif-loss features: {strand_loss}/{len(VIEW_KEYS) // 2}\n"
        f"- Strand-stable motif-specific features: {strand_specific}/{len(VIEW_KEYS) // 2}\n\n"
        "All inference is context-level. See `planned_tests.parquet`, "
        "`view_summary.parquet`, and the full substitution heatmaps.\n"
    )
    artifact_paths = [
        *(output_dir / name for name in tables),
        *plot_paths,
        result_path,
        results_md,
    ]
    manifest = {
        **result,
        "artifacts": {
            str(path.relative_to(output_dir)): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in artifact_paths
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.extraction_dir, args.output_dir)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
