"""Analyze the frozen feature-1662 saturation perturbation experiment."""

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
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from common import ISSUE, sha256_file, write_json
from saturation_common import (
    CONTEXTS_PER_CODON_POSITION,
    ORIENTATIONS,
    POSITIONS,
    RUN_ID,
    bh_adjust,
)

ALPHA = 0.05
MIN_CONSEQUENCE_PAIRS = 30


def standardized_mean_difference(left: np.ndarray, right: np.ndarray) -> float:
    assert left.size >= 2 and right.size >= 2
    pooled = (
        (left.size - 1) * left.var(ddof=1) + (right.size - 1) * right.var(ddof=1)
    ) / (left.size + right.size - 2)
    assert pooled > 0
    return float((left.mean() - right.mean()) / np.sqrt(pooled))


def paired_standardized_mean_difference(left: np.ndarray, right: np.ndarray) -> float:
    difference = left - right
    assert difference.size >= 2 and difference.std(ddof=1) > 0
    return float(difference.mean() / difference.std(ddof=1))


def paired_p_values(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    assert left.shape == right.shape and left.size >= 2
    difference = left - right
    if np.all(difference == 0):
        return 1.0, 1.0
    t_result = stats.ttest_rel(left, right)
    rank_result = stats.wilcoxon(left, right, alternative="two-sided")
    assert np.isfinite(t_result.pvalue) and np.isfinite(rank_result.pvalue)
    return float(t_result.pvalue), float(rank_result.pvalue)


def independent_p_values(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    assert left.size >= 2 and right.size >= 2
    t_result = stats.ttest_ind(left, right, equal_var=False)
    rank_result = stats.mannwhitneyu(left, right, alternative="two-sided")
    assert np.isfinite(t_result.pvalue) and np.isfinite(rank_result.pvalue)
    return float(t_result.pvalue), float(rank_result.pvalue)


def add_bh(frame: pl.DataFrame, *, p_column: str, q_column: str) -> pl.DataFrame:
    values = frame[p_column].to_list()
    finite_indices = [
        index
        for index, value in enumerate(values)
        if value is not None and np.isfinite(value)
    ]
    adjusted: list[float | None] = [None] * len(values)
    if finite_indices:
        finite_values = np.array([values[index] for index in finite_indices])
        for index, value in zip(finite_indices, bh_adjust(finite_values), strict=True):
            adjusted[index] = float(value)
    return frame.with_columns(pl.Series(q_column, adjusted, dtype=pl.Float64))


def validate_extraction(
    extraction_dir: Path,
) -> tuple[dict[str, Any], pl.DataFrame, pl.DataFrame]:
    manifest_path = extraction_dir / "manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["issue"] == ISSUE and manifest["run_id"] == RUN_ID
    assert manifest["analysis_status"] == "post_hoc_mechanistic_perturbation"
    for name, metadata in manifest["artifacts"].items():
        path = extraction_dir / name
        assert path.is_file() and path.stat().st_size == metadata["bytes"]
        assert sha256_file(path) == metadata["sha256"]
    contexts = pl.read_parquet(extraction_dir / "contexts.parquet")
    responses = pl.read_parquet(extraction_dir / "feature1662_responses.parquet")
    assert contexts.height == 3 * CONTEXTS_PER_CODON_POSITION
    assert contexts["context_index"].to_list() == list(range(contexts.height))
    assert responses.height == contexts.height * 93 * len(ORIENTATIONS)
    assert set(responses["orientation"].unique()) == set(ORIENTATIONS)
    assert set(responses["genomic_offset"].unique()) == set(POSITIONS)
    assert responses.group_by("orientation", "context_index").len()[
        "len"
    ].unique().to_list() == [93]
    return manifest, contexts, responses


def generic_focal_tests(responses: pl.DataFrame) -> pl.DataFrame:
    per_offset = responses.group_by(
        "orientation", "context_index", "genomic_offset"
    ).agg(pl.col("abs_delta").mean().alias("mean_abs_delta"))
    rows: list[dict[str, Any]] = []
    for orientation in ORIENTATIONS:
        current = per_offset.filter(pl.col("orientation") == orientation)
        center = (
            current.filter(pl.col("genomic_offset") == 0)
            .select("context_index", pl.col("mean_abs_delta").alias("center"))
            .sort("context_index")
        )
        neighbors = (
            current.filter(pl.col("genomic_offset").is_in([-1, 1]))
            .group_by("context_index")
            .agg(pl.col("mean_abs_delta").mean().alias("neighbors"))
            .sort("context_index")
        )
        paired = center.join(neighbors, on="context_index", validate="1:1")
        left = paired["center"].to_numpy()
        right = paired["neighbors"].to_numpy()
        t_p, rank_p = paired_p_values(left, right)
        rows.append(
            {
                "family": "generic_focal_sensitivity",
                "orientation": orientation,
                "contrast": "center_vs_plus_minus_1",
                "n_left": left.size,
                "n_right": right.size,
                "left_mean": float(left.mean()),
                "right_mean": float(right.mean()),
                "mean_difference": float((left - right).mean()),
                "standardized_mean_difference": paired_standardized_mean_difference(
                    left, right
                ),
                "rank_biserial": None,
                "t_p": t_p,
                "rank_p": rank_p,
            }
        )
    frame = pl.DataFrame(rows)
    frame = add_bh(frame, p_column="t_p", q_column="t_q")
    return add_bh(frame, p_column="rank_p", q_column="rank_q")


def codon_phase_tests(contexts: pl.DataFrame, responses: pl.DataFrame) -> pl.DataFrame:
    center = (
        responses.filter(pl.col("genomic_offset") == 0)
        .group_by("orientation", "context_index")
        .agg(pl.col("abs_delta").mean().alias("mean_abs_delta"))
        .join(
            contexts.select("context_index", "focal_codon_position"),
            on="context_index",
            validate="m:1",
        )
    )
    rows: list[dict[str, Any]] = []
    for orientation in ORIENTATIONS:
        current = center.filter(pl.col("orientation") == orientation)
        middle = current.filter(pl.col("focal_codon_position") == 2)[
            "mean_abs_delta"
        ].to_numpy()
        assert middle.size == CONTEXTS_PER_CODON_POSITION
        for comparison in (1, 3):
            other = current.filter(pl.col("focal_codon_position") == comparison)[
                "mean_abs_delta"
            ].to_numpy()
            assert other.size == CONTEXTS_PER_CODON_POSITION
            t_p, rank_p = independent_p_values(middle, other)
            mann = stats.mannwhitneyu(middle, other, alternative="two-sided")
            rows.append(
                {
                    "family": "codon_phase",
                    "orientation": orientation,
                    "contrast": f"position_2_vs_{comparison}",
                    "n_left": middle.size,
                    "n_right": other.size,
                    "left_mean": float(middle.mean()),
                    "right_mean": float(other.mean()),
                    "mean_difference": float(middle.mean() - other.mean()),
                    "standardized_mean_difference": standardized_mean_difference(
                        middle, other
                    ),
                    "rank_biserial": float(
                        2 * mann.statistic / (middle.size * other.size) - 1
                    ),
                    "t_p": t_p,
                    "rank_p": rank_p,
                }
            )
    frame = pl.DataFrame(rows)
    frame = add_bh(frame, p_column="t_p", q_column="t_q")
    return add_bh(frame, p_column="rank_p", q_column="rank_q")


def consequence_tests(responses: pl.DataFrame) -> pl.DataFrame:
    center = responses.filter(pl.col("genomic_offset") == 0).with_columns(
        pl.when(pl.col("consequence") == "synonymous")
        .then(pl.lit("synonymous"))
        .otherwise(pl.lit("nonsynonymous"))
        .alias("consequence_group")
    )
    per_context = center.group_by(
        "orientation", "context_index", "consequence_group"
    ).agg(pl.col("abs_delta").mean().alias("mean_abs_delta"))
    rows: list[dict[str, Any]] = []
    for orientation in ORIENTATIONS:
        current = per_context.filter(pl.col("orientation") == orientation)
        nonsynonymous = current.filter(
            pl.col("consequence_group") == "nonsynonymous"
        ).select("context_index", pl.col("mean_abs_delta").alias("nonsynonymous"))
        synonymous = current.filter(pl.col("consequence_group") == "synonymous").select(
            "context_index", pl.col("mean_abs_delta").alias("synonymous")
        )
        paired = nonsynonymous.join(
            synonymous, on="context_index", how="inner", validate="1:1"
        ).sort("context_index")
        left = paired["nonsynonymous"].to_numpy()
        right = paired["synonymous"].to_numpy()
        enough = left.size >= MIN_CONSEQUENCE_PAIRS
        if enough:
            t_p, rank_p = paired_p_values(left, right)
            standardized = paired_standardized_mean_difference(left, right)
        else:
            t_p, rank_p, standardized = None, None, None
        rows.append(
            {
                "family": "coding_consequence",
                "orientation": orientation,
                "contrast": "nonsynonymous_vs_synonymous",
                "n_left": left.size,
                "n_right": right.size,
                "left_mean": float(left.mean()) if left.size else None,
                "right_mean": float(right.mean()) if right.size else None,
                "mean_difference": (
                    float((left - right).mean()) if left.size else None
                ),
                "standardized_mean_difference": standardized,
                "rank_biserial": None,
                "t_p": t_p,
                "rank_p": rank_p,
                "minimum_pairs_met": enough,
            }
        )
    frame = pl.DataFrame(rows)
    frame = add_bh(frame, p_column="t_p", q_column="t_q")
    return add_bh(frame, p_column="rank_p", q_column="rank_q")


def build_profiles(contexts: pl.DataFrame, responses: pl.DataFrame) -> pl.DataFrame:
    joined = responses.join(
        contexts.select("context_index", "focal_codon_position"),
        on="context_index",
        validate="m:1",
    )
    return (
        joined.group_by("orientation", "focal_codon_position", "transcript_offset")
        .agg(
            pl.len().alias("n"),
            pl.col("context_index").n_unique().alias("n_contexts"),
            pl.col("abs_delta").mean().alias("mean_abs_delta"),
            pl.col("abs_delta").median().alias("median_abs_delta"),
            (pl.col("abs_delta").std(ddof=1) / pl.len().sqrt()).alias("se_abs_delta"),
            pl.col("delta").mean().alias("mean_delta"),
            pl.col("delta").median().alias("median_delta"),
            (pl.col("delta").std(ddof=1) / pl.len().sqrt()).alias("se_delta"),
            (pl.col("abs_delta") > 0).sum().alias("nonzero_responses"),
        )
        .sort("orientation", "focal_codon_position", "transcript_offset")
    )


def build_consequence_summary(
    contexts: pl.DataFrame, responses: pl.DataFrame
) -> pl.DataFrame:
    return (
        responses.filter(pl.col("genomic_offset") == 0)
        .join(
            contexts.select("context_index", "focal_codon_position"),
            on="context_index",
            validate="m:1",
        )
        .group_by("orientation", "focal_codon_position", "consequence")
        .agg(
            pl.len().alias("n"),
            pl.col("context_index").n_unique().alias("n_contexts"),
            pl.col("abs_delta").mean().alias("mean_abs_delta"),
            pl.col("abs_delta").median().alias("median_abs_delta"),
            pl.col("delta").mean().alias("mean_delta"),
            pl.col("delta").median().alias("median_delta"),
            (pl.col("abs_delta") > 0).sum().alias("nonzero_responses"),
        )
        .sort("orientation", "focal_codon_position", "consequence")
    )


def plot_profile(
    profiles: pl.DataFrame,
    *,
    output_dir: Path,
    signed: bool,
) -> list[Path]:
    data = profiles.to_pandas().copy()
    data["codon position"] = data["focal_codon_position"].astype(str)
    mean_column = "mean_delta" if signed else "mean_abs_delta"
    se_column = "se_delta" if signed else "se_abs_delta"
    y_label = (
        "Mean signed activation change" if signed else "Mean absolute activation change"
    )
    palette = {"1": "#3b528b", "2": "#21918c", "3": "#5ec962"}
    sns.set_theme(context="talk", style="whitegrid")
    grid = sns.relplot(
        data=data,
        x="transcript_offset",
        y=mean_column,
        hue="codon position",
        col="orientation",
        kind="line",
        marker="o",
        palette=palette,
        facet_kws={"sharey": False},
        height=4.2,
        aspect=1.25,
    )
    for orientation, axis in grid.axes_dict.items():
        current = data[data["orientation"] == orientation]
        for codon_position, color in palette.items():
            group = current[current["codon position"] == codon_position].sort_values(
                "transcript_offset"
            )
            axis.fill_between(
                group["transcript_offset"],
                group[mean_column] - group[se_column],
                group[mean_column] + group[se_column],
                color=color,
                alpha=0.16,
                linewidth=0,
            )
        axis.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)
        if signed:
            axis.axhline(0, color="black", linewidth=1, alpha=0.4)
    grid.set_axis_labels("", "")
    grid.set_titles("{col_name}")
    title = (
        "Feature 1662 signed saturation profile"
        if signed
        else "Feature 1662 saturation sensitivity"
    )
    grid.figure.suptitle(f"{title}\nerror bands = ±1 SE", y=1.08)
    grid.figure.supxlabel("Transcript-oriented offset from variant (bp)", y=0.03)
    grid.figure.supylabel(y_label, x=0.015)
    grid.figure.subplots_adjust(top=0.78, bottom=0.20, left=0.09)
    stem = "signed_saturation_profile" if signed else "saturation_profile"
    svg_path = output_dir / f"{stem}.svg"
    png_path = output_dir / f"{stem}.png"
    grid.figure.savefig(svg_path, bbox_inches="tight")
    grid.figure.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(grid.figure)
    return [svg_path, png_path]


def strict_success(frame: pl.DataFrame, *, expected_rows: int) -> bool:
    if frame.height != expected_rows:
        return False
    required = (
        (pl.col("mean_difference") > 0)
        & (pl.col("t_q") < ALPHA)
        & (pl.col("rank_q") < ALPHA)
    )
    if "minimum_pairs_met" in frame.columns:
        required = required & pl.col("minimum_pairs_met")
    return bool(frame.select(required.fill_null(False).all()).item())


def analyze(extraction_dir: Path, output_dir: Path) -> dict[str, Any]:
    assert extraction_dir.is_dir() and not output_dir.exists()
    extraction_manifest, contexts, responses = validate_extraction(extraction_dir)
    output_dir.mkdir(parents=True)

    generic = generic_focal_tests(responses)
    phase = codon_phase_tests(contexts, responses)
    consequence = consequence_tests(responses)
    tests = pl.concat([generic, phase, consequence], how="diagonal_relaxed").sort(
        "family", "orientation", "contrast"
    )
    profiles = build_profiles(contexts, responses)
    consequence_summary = build_consequence_summary(contexts, responses)

    test_path = output_dir / "planned_tests.parquet"
    profile_path = output_dir / "saturation_profiles.parquet"
    consequence_path = output_dir / "consequence_summary.parquet"
    tests.write_parquet(test_path, compression="zstd")
    profiles.write_parquet(profile_path, compression="zstd")
    consequence_summary.write_parquet(consequence_path, compression="zstd")
    plot_paths = plot_profile(profiles, output_dir=output_dir, signed=False)
    plot_paths.extend(plot_profile(profiles, output_dir=output_dir, signed=True))

    success = {
        "generic_focal_sensitivity": strict_success(generic, expected_rows=2),
        "codon_phase": strict_success(phase, expected_rows=4),
        "coding_consequence": strict_success(consequence, expected_rows=2),
    }
    result: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "issue": ISSUE,
        "run_id": RUN_ID,
        "analysis_status": "post_hoc_mechanistic_perturbation",
        "feature_id": 1662,
        "extraction_manifest_sha256": sha256_file(extraction_dir / "manifest.json"),
        "extraction_commit": extraction_manifest["experiment_commit"],
        "multiple_testing": {
            "generic_focal_sensitivity": (
                "BH across forward and reverse-complement, separately for paired t "
                "and Wilcoxon"
            ),
            "codon_phase": (
                "BH across 2 orientations x 2 contrasts, separately for Welch and "
                "Mann-Whitney"
            ),
            "coding_consequence": (
                "BH across forward and reverse-complement, separately for paired t "
                "and Wilcoxon"
            ),
        },
        "minimum_paired_contexts_for_consequence_test": MIN_CONSEQUENCE_PAIRS,
        "strict_success": success,
        "planned_tests": tests.to_dicts(),
        "descriptive_profile_cells": profiles.height,
        "descriptive_consequence_cells": consequence_summary.height,
    }
    write_json(output_dir / "results.json", result)
    (output_dir / "RESULTS.md").write_text(
        "# Feature 1662 saturation perturbation\n\n"
        f"Strict outcomes: `{json.dumps(success, sort_keys=True)}`\n\n"
        "The full planned-test rows are in `planned_tests.parquet`; full unsigned "
        "and signed transcript-oriented profiles are in "
        "`saturation_profiles.parquet`.\n"
    )
    artifact_paths = [
        test_path,
        profile_path,
        consequence_path,
        *plot_paths,
        output_dir / "results.json",
        output_dir / "RESULTS.md",
    ]
    manifest = {
        **result,
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
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
    print(json.dumps(result["strict_success"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
