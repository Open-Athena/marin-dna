"""Summarize the post-hoc characterization of issue #421 selected SAE features."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

FEATURE_LABELS = {
    (10, 11_137): "block 10 · feature 11,137",
    (19, 219): "block 19 · feature 219",
    (19, 11_928): "block 19 · feature 11,928",
}
FEATURE_ORDER = list(FEATURE_LABELS.values())
STRAND_LABELS = {"forward": "FWD", "reverse_complement": "RC"}
STRAND_ORDER = ["FWD", "RC"]
PROTOCOL_LABELS = {
    "raw": "Raw Pearson",
    "log1p_feature": "log1p Pearson",
    "spearman": "Spearman",
    "trim_both_top_1pct": "Trim both 1%",
}
PROTOCOL_ORDER = list(PROTOCOL_LABELS.values())
METRIC_LABELS = {
    "ref_gc_fraction": "GC fraction",
    "ref_cpg_count": "CpG count",
    "ref_entropy": "Entropy",
    "repeat_fraction": "Repeat fraction",
}
METRIC_ORDER = list(METRIC_LABELS.values())
DOMAIN_ORDER = [
    "Overall",
    "5′ UTR",
    "Missense",
    "ncRNA exon",
    "TSS-proximal",
    "Splicing",
    "Distal regulatory",
    "3′ UTR",
    "Synonymous",
]
DOMAIN_LABELS = {
    "5_prime_UTR_variant": "5′ UTR",
    "missense_variant": "Missense",
    "non_coding_transcript_exon_variant": "ncRNA exon",
    "tss_proximal": "TSS-proximal",
    "splicing": "Splicing",
    "distal": "Distal regulatory",
    "3_prime_UTR_variant": "3′ UTR",
    "synonymous_variant": "Synonymous",
}


def _label_features(frame: pl.DataFrame) -> pl.DataFrame:
    labels = pl.DataFrame(
        {
            "report_block": [key[0] for key in FEATURE_LABELS],
            "feature_id": [key[1] for key in FEATURE_LABELS],
            "feature": list(FEATURE_LABELS.values()),
        }
    )
    return frame.join(
        labels,
        on=["report_block", "feature_id"],
        how="inner",
        validate="m:1",
    ).with_columns(pl.col("orientation").replace_strict(STRAND_LABELS).alias("strand"))


def label_summary(category: pl.DataFrame) -> pl.DataFrame:
    overall = category.filter(
        (pl.col("dimension") == "label") & (pl.col("level") == "pathogenic")
    ).with_columns(pl.lit("Overall").alias("domain"))
    within = category.filter(pl.col("dimension") == "label_within_subset").with_columns(
        pl.col("level").replace_strict(DOMAIN_LABELS).alias("domain")
    )
    result = _label_features(pl.concat([overall, within], how="vertical")).select(
        "report_block",
        "feature_id",
        "feature",
        "orientation",
        "strand",
        "domain",
        "n_level",
        "n_other",
        "mean_level",
        "mean_other",
        "rank_biserial",
        "mann_whitney_p",
        "mann_whitney_q",
        "mean_difference",
        "welch_p",
        "welch_q",
    )
    assert result.height == len(FEATURE_LABELS) * len(STRAND_LABELS) * len(DOMAIN_ORDER)
    return result.sort(["feature", "strand", "domain"])


def target_robustness(target: pl.DataFrame) -> pl.DataFrame:
    result = _label_features(
        target.filter(pl.col("protocol").is_in(PROTOCOL_LABELS))
    ).with_columns(
        pl.col("protocol").replace_strict(PROTOCOL_LABELS).alias("protocol_label")
    )
    assert set(result["protocol_label"].unique()) == set(PROTOCOL_ORDER)
    return result.sort(["feature", "strand", "target_id", "protocol_label"])


def sequence_summary(sequence: pl.DataFrame) -> pl.DataFrame:
    result = _label_features(
        sequence.filter(pl.col("metric").is_in(METRIC_LABELS))
    ).with_columns(
        pl.col("metric").replace_strict(METRIC_LABELS).alias("metric_label"),
        pl.col("statistic").str.to_titlecase().alias("statistic_label"),
    )
    assert set(result["metric_label"].unique()) == set(METRIC_ORDER)
    return result.sort(["feature", "statistic_label", "metric_label", "strand"])


def gc_profile(profile: pl.DataFrame) -> pl.DataFrame:
    result = (
        profile.filter(pl.col("base").is_in(["C", "G"]))
        .group_by(
            "report_block",
            "feature_id",
            "orientation",
            "position",
        )
        .agg(
            pl.col("frequency_difference").sum().alias("gc_frequency_difference"),
            pl.first("top_n"),
            pl.first("background_n"),
        )
        .rename({"position": "relative_position"})
    )
    result = _label_features(result)
    assert set(result["relative_position"].unique()) == set(range(-31, 32))
    return result.sort(["feature", "strand", "relative_position"])


def annotation_extremes(
    category: pl.DataFrame, rows_per_family: int = 4
) -> pl.DataFrame:
    candidates = _label_features(
        category.filter(
            pl.col("dimension").is_in(
                ["subset", "repeat_position", "repeat_class", "repeat_family"]
            )
        )
    ).with_columns(pl.col("rank_biserial").abs().alias("absolute_rank_biserial"))
    return (
        candidates.sort(
            ["feature", "strand", "dimension", "absolute_rank_biserial"],
            descending=[False, False, False, True],
        )
        .group_by("feature", "strand", "dimension", maintain_order=True)
        .head(rows_per_family)
    )


def save_target_plot(frame: pl.DataFrame, output_dir: Path) -> None:
    plot = frame.to_pandas()
    plot["protocol_label"] = __import__("pandas").Categorical(
        plot["protocol_label"], categories=PROTOCOL_ORDER, ordered=True
    )
    sns.set_theme(style="whitegrid", context="notebook")
    grid = sns.relplot(
        data=plot,
        x="protocol_label",
        y="effect",
        hue="target_name",
        row="strand",
        row_order=STRAND_ORDER,
        col="feature",
        col_order=FEATURE_ORDER,
        kind="line",
        marker="o",
        dashes=False,
        estimator=None,
        sort=False,
        height=3.5,
        aspect=1.22,
        facet_kws={"sharey": True},
    )
    grid.set_axis_labels("", "Association (r or ρ)")
    grid.set_titles("{col_name}\n{row_name}")
    for axis in grid.axes.flat:
        axis.tick_params(axis="x", rotation=32)
        axis.axhline(0.0, color="0.35", linewidth=0.8)
    grid.legend.set_title("AlphaGenome outcome")
    grid.figure.subplots_adjust(top=0.84, bottom=0.20, left=0.08, wspace=0.15)
    grid.figure.suptitle(
        "Two accessibility features are rank- and trim-robust; feature 11,928 is not"
    )
    for suffix in ("png", "svg"):
        grid.figure.savefig(
            output_dir / f"alphagenome_tail_robustness.{suffix}",
            bbox_inches="tight",
            dpi=180,
        )
    plt.close(grid.figure)


def save_label_plot(frame: pl.DataFrame, output_dir: Path) -> None:
    plot = frame.to_pandas()
    plot["domain"] = __import__("pandas").Categorical(
        plot["domain"], categories=DOMAIN_ORDER[::-1], ordered=True
    )
    sns.set_theme(style="whitegrid", context="notebook")
    grid = sns.relplot(
        data=plot,
        x="rank_biserial",
        y="domain",
        hue="strand",
        hue_order=STRAND_ORDER,
        style="strand",
        style_order=STRAND_ORDER,
        col="feature",
        col_order=FEATURE_ORDER,
        kind="scatter",
        s=90,
        height=5.0,
        aspect=0.88,
        facet_kws={"sharex": True, "sharey": True},
    )
    grid.set_axis_labels("", "")
    grid.set_titles("{col_name}")
    for axis in grid.axes.flat:
        axis.axvline(0.0, color="0.35", linewidth=0.8)
    grid.legend.set_title("Strand")
    grid.figure.supxlabel("Rank-biserial effect (pathogenic − benign)", y=0.04)
    grid.figure.subplots_adjust(top=0.80, bottom=0.16, wspace=0.17)
    grid.figure.suptitle(
        "Feature 11,928 is consistently larger for Mendelian pathogenic variants"
    )
    for suffix in ("png", "svg"):
        grid.figure.savefig(
            output_dir / f"mendelian_label_associations.{suffix}",
            bbox_inches="tight",
            dpi=180,
        )
    plt.close(grid.figure)


def save_sequence_plot(frame: pl.DataFrame, output_dir: Path) -> None:
    plot = frame.to_pandas()
    plot["metric_label"] = __import__("pandas").Categorical(
        plot["metric_label"], categories=METRIC_ORDER, ordered=True
    )
    sns.set_theme(style="whitegrid", context="notebook")
    grid = sns.catplot(
        data=plot,
        x="metric_label",
        y="effect",
        hue="strand",
        hue_order=STRAND_ORDER,
        row="statistic_label",
        row_order=["Pearson", "Spearman"],
        col="feature",
        col_order=FEATURE_ORDER,
        kind="bar",
        errorbar=None,
        height=3.4,
        aspect=1.18,
        sharey=True,
    )
    grid.set_axis_labels("", "")
    grid.set_titles("{col_name}\n{row_name}")
    for axis in grid.axes.flat:
        axis.tick_params(axis="x", rotation=28)
        axis.axhline(0.0, color="0.35", linewidth=0.8)
    grid.legend.set_title("Strand")
    grid.figure.supylabel("Association with |alt − ref activation| (r or ρ)", x=0.01)
    grid.figure.subplots_adjust(top=0.84, bottom=0.18, left=0.08, wspace=0.15)
    grid.figure.suptitle(
        "Features 11,137 and 219 are strongly GC/CpG-context dependent"
    )
    for suffix in ("png", "svg"):
        grid.figure.savefig(
            output_dir / f"sequence_context_associations.{suffix}",
            bbox_inches="tight",
            dpi=180,
        )
    plt.close(grid.figure)


def save_gc_profile_plot(frame: pl.DataFrame, output_dir: Path) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    grid = sns.relplot(
        data=frame.to_pandas(),
        x="relative_position",
        y="gc_frequency_difference",
        hue="strand",
        hue_order=STRAND_ORDER,
        col="feature",
        col_order=FEATURE_ORDER,
        kind="line",
        dashes=False,
        height=3.7,
        aspect=1.18,
    )
    grid.set_axis_labels("", "")
    grid.set_titles("{col_name}")
    for axis in grid.axes.flat:
        axis.axhline(0.0, color="0.35", linewidth=0.8)
        axis.axvline(0.0, color="0.35", linewidth=0.8, linestyle=":")
    grid.legend.set_title("Strand")
    grid.figure.supxlabel("Position relative to variant", y=0.04)
    grid.figure.supylabel("Top-100 minus background GC frequency", x=0.01)
    grid.figure.subplots_adjust(top=0.76, bottom=0.18, left=0.08, wspace=0.15)
    grid.figure.suptitle(
        "Top feature-219 responses are GC-rich across the local window"
    )
    for suffix in ("png", "svg"):
        grid.figure.savefig(
            output_dir / f"top_context_gc_profile.{suffix}",
            bbox_inches="tight",
            dpi=180,
        )
    plt.close(grid.figure)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        "category_associations": args.result_root / "category_associations.parquet",
        "target_sensitivity": args.result_root / "target_sensitivity.parquet",
        "sequence_metric_associations": args.result_root
        / "sequence_metric_associations.parquet",
        "position_base_enrichment": args.result_root
        / "position_base_enrichment.parquet",
        "orientation_concordance": args.result_root / "orientation_concordance.parquet",
        "feature_summary": args.result_root / "feature_summary.parquet",
    }
    tables = {
        "mendelian_label_associations.csv": label_summary(
            pl.read_parquet(sources["category_associations"])
        ),
        "alphagenome_tail_robustness.csv": target_robustness(
            pl.read_parquet(sources["target_sensitivity"])
        ),
        "sequence_context_associations.csv": sequence_summary(
            pl.read_parquet(sources["sequence_metric_associations"])
        ),
        "top_context_gc_profile.csv": gc_profile(
            pl.read_parquet(sources["position_base_enrichment"])
        ),
        "annotation_extremes.csv": annotation_extremes(
            pl.read_parquet(sources["category_associations"])
        ),
        "orientation_concordance.csv": pl.read_parquet(
            sources["orientation_concordance"]
        ),
        "feature_summary.csv": pl.read_parquet(sources["feature_summary"]),
    }
    for filename, frame in tables.items():
        frame.write_csv(args.output_dir / filename)

    save_label_plot(tables["mendelian_label_associations.csv"], args.output_dir)
    save_target_plot(tables["alphagenome_tail_robustness.csv"], args.output_dir)
    save_sequence_plot(tables["sequence_context_associations.csv"], args.output_dir)
    save_gc_profile_plot(tables["top_context_gc_profile.csv"], args.output_dir)

    metadata = {
        "source_result_root": str(args.result_root),
        "source_sha256": {name: _sha256(path) for name, path in sources.items()},
        "rows": {name: frame.height for name, frame in tables.items()},
    }
    (args.output_dir / "summary_manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
