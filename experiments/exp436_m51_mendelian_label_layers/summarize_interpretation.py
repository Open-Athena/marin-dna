"""Summarize post hoc biological interpretation of final-layer feature 9086.

The feature and four locus regions were nominated after inspecting #436 outcomes and
top contexts. These analyses are exploratory sensitivity and mechanism-generation,
not an independent confirmation of the original all-feature association scan.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scipy
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score

from analyze_focal import bh_adjust
from extract_focal import (
    EXPECTED_ROWS,
    ISSUE,
    ORIENTATIONS,
    sha256_file,
    write_json,
)
from interpret_focal import verify_manifest_artifacts
from train import assert_commit

PRIMARY_ARM = "block19-25m"
PRIMARY_BUDGET = 25_000_200
FEATURE_ID = 9086
RESPONSES = ("abs_delta", "delta")
SCORE_OUTCOMES = ("minus_llr_avg", "probe_score")


@dataclass(frozen=True)
class Locus:
    name: str
    ensembl_id: str
    chrom: str
    start0: int
    end0: int
    category: str


# Ensembl GRCh38 lookup coordinates are returned as 1-based inclusive. They are
# converted here once to the project-wide 0-based, half-open convention.
LOCI = (
    Locus(
        name="FTL",
        ensembl_id="ENSG00000087086",
        chrom="19",
        start0=48_965_274,
        end0=48_967_896,
        category="ferritin_5utr",
    ),
    Locus(
        name="FTH1",
        ensembl_id="ENSG00000167996",
        chrom="11",
        start0=61_959_717,
        end0=61_967_640,
        category="ferritin_5utr",
    ),
    Locus(
        name="TERC",
        ensembl_id="ENSG00000270141",
        chrom="3",
        start0=169_764_609,
        end0=169_765_060,
        category="structured_ncrna",
    ),
    Locus(
        name="RMRP",
        ensembl_id="ENSG00000277027",
        chrom="9",
        start0=35_657_749,
        end0=35_658_019,
        category="structured_ncrna",
    ),
)


@dataclass(frozen=True)
class SensitivityTarget:
    name: str
    subset: str | None
    excluded_loci: tuple[str, ...]
    comparison_group: str


SENSITIVITY_TARGETS = (
    SensitivityTarget("overall", None, (), "overall"),
    SensitivityTarget(
        "overall_without_FTL_FTH1_TERC_RMRP",
        None,
        ("FTL", "FTH1", "TERC", "RMRP"),
        "overall",
    ),
    SensitivityTarget("5_prime_UTR_variant", "5_prime_UTR_variant", (), "5utr"),
    SensitivityTarget(
        "5_prime_UTR_without_FTL_FTH1",
        "5_prime_UTR_variant",
        ("FTL", "FTH1"),
        "5utr",
    ),
    SensitivityTarget(
        "non_coding_transcript_exon_variant",
        "non_coding_transcript_exon_variant",
        (),
        "ncrna",
    ),
    SensitivityTarget(
        "non_coding_transcript_exon_without_TERC_RMRP",
        "non_coding_transcript_exon_variant",
        ("TERC", "RMRP"),
        "ncrna",
    ),
)


def locus_name_expression() -> pl.Expr:
    expression = pl.lit(None, dtype=pl.String)
    for locus in reversed(LOCI):
        inside = (
            (pl.col("chrom") == locus.chrom)
            & (pl.col("pos0") >= locus.start0)
            & (pl.col("pos0") < locus.end0)
        )
        expression = pl.when(inside).then(pl.lit(locus.name)).otherwise(expression)
    return expression.alias("locus")


def annotate_loci(frame: pl.DataFrame) -> pl.DataFrame:
    required = {"chrom", "pos"}
    assert required <= set(frame.columns)
    assert frame.filter(pl.col("pos") < 1).is_empty()
    annotated = frame.with_columns(
        (pl.col("pos").cast(pl.Int64) - 1).alias("pos0")
    ).with_columns(locus_name_expression())

    metadata = pl.DataFrame(
        {
            "locus": [locus.name for locus in LOCI],
            "locus_start0": [locus.start0 for locus in LOCI],
            "locus_end0": [locus.end0 for locus in LOCI],
            "locus_category": [locus.category for locus in LOCI],
            "ensembl_id": [locus.ensembl_id for locus in LOCI],
        }
    )
    annotated = annotated.join(metadata, on="locus", how="left").with_columns(
        (pl.col("pos0") - pl.col("locus_start0")).alias("locus_offset0")
    )
    inside = annotated.filter(pl.col("locus").is_not_null())
    assert inside.filter(
        (pl.col("pos0") < pl.col("locus_start0"))
        | (pl.col("pos0") >= pl.col("locus_end0"))
    ).is_empty()
    return annotated


def select_primary_feature(responses: pl.DataFrame) -> pl.DataFrame:
    required = {
        "arm",
        "budget",
        "orientation",
        "feature_id",
        "panel_row",
        "label",
        "delta",
        "abs_delta",
        "chrom",
        "pos",
        *SCORE_OUTCOMES,
    }
    assert required <= set(responses.columns)
    selected = responses.filter(
        (pl.col("arm") == PRIMARY_ARM)
        & (pl.col("budget") == PRIMARY_BUDGET)
        & (pl.col("feature_id") == FEATURE_ID)
    )
    assert selected.height == EXPECTED_ROWS * len(ORIENTATIONS)
    assert set(selected["orientation"].unique().to_list()) == set(ORIENTATIONS)
    for orientation in ORIENTATIONS:
        strand = selected.filter(pl.col("orientation") == orientation)
        assert strand.height == EXPECTED_ROWS
        assert strand["panel_row"].n_unique() == EXPECTED_ROWS
        assert strand["panel_row"].to_list() == list(range(EXPECTED_ROWS))
    assert selected.filter(
        (pl.col("abs_delta") - pl.col("delta").abs()).abs() > 1e-5
    ).is_empty()
    return annotate_loci(selected)


def _binary_statistics(
    frame: pl.DataFrame,
    *,
    response: str,
) -> dict[str, Any]:
    assert response in RESPONSES
    labels = frame["label"].cast(pl.UInt8).to_numpy()
    values = frame[response].cast(pl.Float64).to_numpy()
    assert labels.ndim == values.ndim == 1 and labels.size == values.size
    assert np.isfinite(values).all()
    positives = values[labels == 1]
    negatives = values[labels == 0]
    assert positives.size >= 2 and negatives.size >= 2

    mean_positive = float(positives.mean())
    mean_negative = float(negatives.mean())
    variance_positive = float(positives.var(ddof=1))
    variance_negative = float(negatives.var(ddof=1))
    pooled_sd = np.sqrt((variance_positive + variance_negative) / 2)
    standardized = (
        (mean_positive - mean_negative) / pooled_sd if pooled_sd > 0 else np.nan
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        welch = stats.ttest_ind(positives, negatives, equal_var=False)
        mann_whitney = stats.mannwhitneyu(
            positives,
            negatives,
            alternative="two-sided",
            method="asymptotic",
        )
    rank_biserial = (
        2 * float(mann_whitney.statistic) / (positives.size * negatives.size) - 1
    )
    auprc = float(average_precision_score(labels, values))
    auprc_negated = float(average_precision_score(labels, -values))
    prevalence = float(positives.size / labels.size)
    return {
        "n": int(labels.size),
        "n_positive": int(positives.size),
        "n_negative": int(negatives.size),
        "prevalence": prevalence,
        "nonzero_support": int(np.count_nonzero(values)),
        "mean_positive": mean_positive,
        "mean_negative": mean_negative,
        "mean_difference": mean_positive - mean_negative,
        "standardized_mean_difference": float(standardized),
        "welch_statistic": float(welch.statistic),
        "welch_p": float(welch.pvalue),
        "u_statistic": float(mann_whitney.statistic),
        "rank_biserial": rank_biserial,
        "mann_whitney_p": float(mann_whitney.pvalue),
        "auprc": auprc,
        "auprc_negated": auprc_negated,
        "best_auprc": max(auprc, auprc_negated),
        "best_auprc_direction": "higher" if auprc >= auprc_negated else "lower",
        "best_auprc_lift": max(auprc, auprc_negated) / prevalence,
    }


def hotspot_sensitivity(responses: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for target in SENSITIVITY_TARGETS:
        target_frame = responses
        if target.subset is not None:
            target_frame = target_frame.filter(pl.col("subset") == target.subset)
        if target.excluded_loci:
            target_frame = target_frame.filter(
                ~pl.col("locus")
                .fill_null("__not_a_locus__")
                .is_in(target.excluded_loci)
            )
        for orientation in ORIENTATIONS:
            strand = target_frame.filter(pl.col("orientation") == orientation)
            for response in RESPONSES:
                rows.append(
                    {
                        "feature_id": FEATURE_ID,
                        "arm": PRIMARY_ARM,
                        "orientation": orientation,
                        "response": response,
                        "target": target.name,
                        "comparison_group": target.comparison_group,
                        "excluded_loci": ",".join(target.excluded_loci),
                        **_binary_statistics(strand, response=response),
                    }
                )
    result = pl.DataFrame(rows)
    corrected: list[pl.DataFrame] = []
    for _, family in result.group_by(["orientation", "response"], maintain_order=True):
        corrected.append(
            family.with_columns(
                pl.Series("welch_q", bh_adjust(family["welch_p"].to_numpy())),
                pl.Series(
                    "mann_whitney_q",
                    bh_adjust(family["mann_whitney_p"].to_numpy()),
                ),
            )
        )
    return pl.concat(corrected).sort(
        ["comparison_group", "target", "orientation", "response"]
    )


def label_stratified_score_correlations(
    responses: pl.DataFrame,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    strata: tuple[tuple[str, int | None], ...] = (
        ("all", None),
        ("benign", 0),
        ("pathogenic", 1),
    )
    for orientation in ORIENTATIONS:
        strand = responses.filter(pl.col("orientation") == orientation)
        for stratum_name, label in strata:
            selected = (
                strand if label is None else strand.filter(pl.col("label") == label)
            )
            for response in RESPONSES:
                x = selected[response].cast(pl.Float64).to_numpy()
                for outcome in SCORE_OUTCOMES:
                    y = selected[outcome].cast(pl.Float64).to_numpy()
                    finite = np.isfinite(x) & np.isfinite(y)
                    x_valid = x[finite]
                    y_valid = y[finite]
                    assert x_valid.size >= 10
                    assert np.unique(x_valid).size >= 2
                    assert np.unique(y_valid).size >= 2
                    pearson = pearsonr(x_valid, y_valid)
                    spearman = spearmanr(x_valid, y_valid)
                    rows.append(
                        {
                            "feature_id": FEATURE_ID,
                            "arm": PRIMARY_ARM,
                            "orientation": orientation,
                            "label_stratum": stratum_name,
                            "response": response,
                            "outcome": outcome,
                            "n": int(x_valid.size),
                            "pearson_r": float(pearson.statistic),
                            "pearson_p": float(pearson.pvalue),
                            "spearman_rho": float(spearman.statistic),
                            "spearman_p": float(spearman.pvalue),
                        }
                    )
    result = pl.DataFrame(rows)
    return result.with_columns(
        pl.Series("pearson_q", bh_adjust(result["pearson_p"].to_numpy())),
        pl.Series("spearman_q", bh_adjust(result["spearman_p"].to_numpy())),
    ).sort(["outcome", "orientation", "response", "label_stratum"])


def locus_response_summary(responses: pl.DataFrame) -> pl.DataFrame:
    locus_rows = responses.filter(pl.col("locus").is_not_null())
    assert set(locus_rows["locus"].unique().to_list()) == {locus.name for locus in LOCI}
    totals = responses.group_by("orientation").agg(
        pl.col("abs_delta").sum().alias("orientation_total_abs_delta")
    )
    result = (
        locus_rows.group_by(
            [
                "orientation",
                "locus",
                "locus_category",
                "ensembl_id",
                "locus_start0",
                "locus_end0",
                "label",
            ],
            maintain_order=True,
        )
        .agg(
            pl.len().alias("n"),
            pl.col("match_group").n_unique().alias("match_groups"),
            (pl.col("abs_delta") > 0).sum().alias("nonzero_abs_delta"),
            pl.col("abs_delta").sum().alias("sum_abs_delta"),
            pl.col("abs_delta").mean().alias("mean_abs_delta"),
            pl.col("abs_delta").median().alias("median_abs_delta"),
            pl.col("abs_delta").max().alias("max_abs_delta"),
            pl.col("delta").mean().alias("mean_delta"),
            pl.col("delta").median().alias("median_delta"),
            pl.col("delta").min().alias("min_delta"),
            pl.col("delta").max().alias("max_delta"),
        )
        .join(totals, on="orientation", how="left")
        .with_columns(
            (pl.col("sum_abs_delta") / pl.col("orientation_total_abs_delta")).alias(
                "orientation_abs_delta_mass_fraction"
            )
        )
        .sort(["locus", "orientation", "label"])
    )
    assert result.filter(pl.col("n") <= 0).is_empty()
    return result


def plot_locus_responses(
    responses: pl.DataFrame,
    output_dir: Path,
) -> list[Path]:
    selected = responses.filter(pl.col("locus").is_not_null())
    fig, axes = plt.subplots(
        len(LOCI),
        len(ORIENTATIONS),
        figsize=(14.0, 12.5),
        sharex=False,
        sharey=False,
    )
    colors = {0: "#9ca3af", 1: "#7c3aed"}
    labels = {0: "benign", 1: "pathogenic"}
    for row_index, locus in enumerate(LOCI):
        for column_index, orientation in enumerate(ORIENTATIONS):
            axis = axes[row_index, column_index]
            group = selected.filter(
                (pl.col("locus") == locus.name) & (pl.col("orientation") == orientation)
            )
            for label in (0, 1):
                label_rows = group.filter(pl.col("label") == label)
                axis.scatter(
                    label_rows["locus_offset0"].to_numpy(),
                    label_rows["delta"].to_numpy(),
                    s=22,
                    alpha=0.72,
                    color=colors[label],
                    edgecolors="none",
                    label=labels[label],
                )
            axis.axhline(0, color="#374151", linewidth=0.7, alpha=0.6)
            axis.set_title(
                f"{locus.name} · {orientation.replace('_', ' ')}",
                fontsize=10,
            )
            axis.set_xlabel("Offset from Ensembl gene start (bp)")
            axis.set_ylabel("Feature 9086 signed Δ")
            positives = group.filter(pl.col("label") == 1).height
            negatives = group.height - positives
            axis.text(
                0.98,
                0.96,
                f"n+={positives}, n−={negatives}",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="#4b5563",
            )
            if row_index == 0 and column_index == 0:
                axis.legend(frameon=False, loc="best")
    fig.suptitle(
        "Feature 9086 responds strongly at ferritin 5′-UTR and structured-ncRNA loci",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.01,
        "Post hoc loci nominated from outcome-aware top contexts; panels are exploratory.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.07,
        top=0.94,
        hspace=0.34,
        wspace=0.22,
    )
    paths = [
        output_dir / "feature9086_locus_responses.svg",
        output_dir / "feature9086_locus_responses.png",
    ]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], dpi=180, bbox_inches="tight")
    plt.close(fig)
    return paths


def render_results_markdown(
    sensitivity: pl.DataFrame,
    correlations: pl.DataFrame,
    locus_summary: pl.DataFrame,
) -> str:
    sensitivity_rows = sensitivity.select(
        "target",
        "orientation",
        "response",
        "n",
        "n_positive",
        "prevalence",
        "best_auprc",
        "best_auprc_lift",
        "best_auprc_direction",
    ).to_dicts()
    lines = [
        "# Feature 9086 post hoc locus sensitivity",
        "",
        "Feature 9086 and the four locus regions were selected after inspecting",
        "Mendelian-label associations and top contexts. These results are exploratory.",
        "",
        "## Hotspot-exclusion sensitivity",
        "",
        "| target | orientation | response | n | positives | prevalence | best AUPRC | lift | direction |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in sensitivity_rows:
        lines.append(
            f"| {row['target']} | {row['orientation']} | {row['response']} | "
            f"{row['n']} | {row['n_positive']} | {row['prevalence']:.3f} | "
            f"{row['best_auprc']:.3f} | {row['best_auprc_lift']:.2f} | "
            f"{row['best_auprc_direction']} |"
        )
    lines.extend(
        [
            "",
            "Raw AUPRC is not directly comparable after exclusions because prevalence",
            "changes. Lift is best AUPRC divided by the corresponding prevalence.",
            "",
            "## Label-stratified official-score correlations",
            "",
            "| outcome | orientation | response | stratum | n | Pearson r | Spearman rho |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    for row in correlations.to_dicts():
        lines.append(
            f"| {row['outcome']} | {row['orientation']} | {row['response']} | "
            f"{row['label_stratum']} | {row['n']} | {row['pearson_r']:.3f} | "
            f"{row['spearman_rho']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Locus response summary",
            "",
            "| locus | orientation | label | n | nonzero | mean abs Δ | median abs Δ | max abs Δ |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in locus_summary.to_dicts():
        lines.append(
            f"| {row['locus']} | {row['orientation']} | {row['label']} | "
            f"{row['n']} | {row['nonzero_abs_delta']} | "
            f"{row['mean_abs_delta']:.2f} | {row['median_abs_delta']:.2f} | "
            f"{row['max_abs_delta']:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def summarize(
    *,
    interpretation_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    assert interpretation_root.is_dir()
    assert not output_dir.exists()
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert_commit(experiment_commit)
    started = time.monotonic()

    input_manifest_path = interpretation_root / "manifest.json"
    input_manifest = json.loads(input_manifest_path.read_text())
    verify_manifest_artifacts(interpretation_root, input_manifest)
    responses_path = interpretation_root / "candidate_responses.parquet"
    responses = select_primary_feature(pl.read_parquet(responses_path))

    sensitivity = hotspot_sensitivity(responses)
    correlations = label_stratified_score_correlations(responses)
    locus_summary = locus_response_summary(responses)

    output_dir.mkdir(parents=True)
    tables = {
        "feature9086_hotspot_sensitivity.parquet": sensitivity,
        "feature9086_label_stratified_score_correlations.parquet": correlations,
        "feature9086_locus_summary.parquet": locus_summary,
    }
    for filename, frame in tables.items():
        frame.write_parquet(output_dir / filename, compression="zstd")
    plot_paths = plot_locus_responses(responses, output_dir)
    markdown_path = output_dir / "RESULTS.md"
    markdown_path.write_text(
        render_results_markdown(sensitivity, correlations, locus_summary)
    )

    artifact_paths = [
        *(output_dir / filename for filename in tables),
        *plot_paths,
        markdown_path,
    ]
    artifacts = {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in artifact_paths
    }
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "issue": ISSUE,
        "run_id": os.environ.get("RUN_ID", ""),
        "experiment_commit": experiment_commit,
        "elapsed_seconds": time.monotonic() - started,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "polars": pl.__version__,
        "scipy": scipy.__version__,
        "inputs": {
            "interpretation_manifest_sha256": sha256_file(input_manifest_path),
            "candidate_responses_sha256": sha256_file(responses_path),
            "verified_interpretation_artifacts": len(input_manifest["artifacts"]),
        },
        "protocol": {
            "status": "post_hoc_exploratory",
            "feature_id": FEATURE_ID,
            "arm": PRIMARY_ARM,
            "responses": list(RESPONSES),
            "score_outcomes": list(SCORE_OUTCOMES),
            "coordinate_convention": (
                "panel pos is converted from 1-based at input to 0-based half-open"
            ),
            "loci": [asdict(locus) for locus in LOCI],
            "sensitivity_targets": [asdict(target) for target in SENSITIVITY_TARGETS],
            "hotspot_fdr": (
                "BH across the six fixed sensitivity targets within each "
                "orientation x response family"
            ),
            "score_correlation_fdr": (
                "BH across all fixed orientation x response x outcome x "
                "label-stratum correlations"
            ),
        },
        "rows": {
            "hotspot_sensitivity": sensitivity.height,
            "label_stratified_score_correlations": correlations.height,
            "locus_summary": locus_summary.height,
        },
        "artifacts": artifacts,
    }
    write_json(output_dir / "results.json", result)
    result["artifacts"]["results.json"] = {
        "bytes": (output_dir / "results.json").stat().st_size,
        "sha256": sha256_file(output_dir / "results.json"),
    }
    write_json(output_dir / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interpretation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        interpretation_root=args.interpretation_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
