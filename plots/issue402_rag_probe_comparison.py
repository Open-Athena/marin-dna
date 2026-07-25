#!/usr/bin/env python3
"""Compare issue #402's frozen-embedding probe with likelihood scoring."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

DEFAULT_INPUT = (
    "gs://marin-us-east5/users/ubuntu/evals/"
    "dna-exp402-rag-h640-p46m-1b/ropefix/step-7628/"
    "mendelian_traits_probe/probe_metrics.parquet"
)
SCORE_LABELS = {
    "minus_llr_avg": "Likelihood score",
    "probe_score": "Frozen embedding probe",
}
SCORE_COLORS = {
    "Likelihood score": "#777777",
    "Frozen embedding probe": "#3366cc",
}
SUBSET_LABELS = {
    "_macro_avg_": "Macro average",
    "tss_proximal": "TSS proximal",
    "splicing": "Splicing",
    "3_prime_UTR_variant": "3′ UTR",
    "missense_variant": "Missense",
    "distal": "Distal regulatory",
    "5_prime_UTR_variant": "5′ UTR",
    "non_coding_transcript_exon_variant": "Non-coding exon",
}
SUBSET_ORDER = list(SUBSET_LABELS.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/output/issue402_rag_probe_comparison"),
    )
    return parser.parse_args()


def load_comparison(input_path: str) -> pl.DataFrame:
    """Load the preregistered, adequately powered subsets and assert invariants."""
    metrics = pl.read_parquet(input_path)
    assert metrics.schema == {
        "score_type": pl.String,
        "subset": pl.String,
        "value": pl.Float64,
        "se": pl.Float64,
        "n": pl.Int64,
        "n_pos": pl.Int64,
        "n_chrom": pl.Int64,
    }
    assert set(metrics["score_type"]) == set(SCORE_LABELS)
    eligible = metrics.filter(pl.col("subset").is_in(SUBSET_LABELS))
    assert eligible.height == 2 * len(SUBSET_LABELS)
    assert set(eligible["subset"]) == set(SUBSET_LABELS)
    assert eligible.filter(
        ~pl.col("value").is_finite() | ~pl.col("se").is_finite()
    ).is_empty()
    paired_counts = eligible.group_by("subset").agg(
        pl.len().alias("n_scores"),
        pl.col("n").n_unique().alias("n_n"),
        pl.col("n_pos").n_unique().alias("n_n_pos"),
        pl.col("n_chrom").n_unique().alias("n_n_chrom"),
    )
    assert paired_counts.select(
        (pl.col("n_scores") == 2).all()
        & (pl.col("n_n") == 1).all()
        & (pl.col("n_n_pos") == 1).all()
        & (pl.col("n_n_chrom") == 1).all()
    ).item()
    return eligible.with_columns(
        pl.col("score_type").replace_strict(SCORE_LABELS).alias("score"),
        pl.col("subset").replace_strict(SUBSET_LABELS).alias("display_subset"),
    )


def plot_comparison(comparison: pl.DataFrame, output_dir: Path) -> None:
    """Render paired subset AUPRC values with capless ±1 SE bars."""
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.write_parquet(output_dir / "metrics.parquet", compression="zstd")
    data = comparison.to_pandas()
    sns.set_theme(style="whitegrid", context="talk")
    grid = sns.catplot(
        data=data,
        x="value",
        y="display_subset",
        hue="score",
        order=SUBSET_ORDER,
        hue_order=list(SCORE_COLORS),
        kind="point",
        errorbar=None,
        linestyles="none",
        markers=["o", "s"],
        palette=SCORE_COLORS,
        height=6.8,
        aspect=1.35,
    )
    axis = grid.ax
    y_positions = {subset: index for index, subset in enumerate(SUBSET_ORDER)}
    for subset in SUBSET_ORDER:
        values = data.loc[data["display_subset"] == subset, "value"]
        assert len(values) == 2
        axis.plot(
            [values.min(), values.max()],
            [y_positions[subset], y_positions[subset]],
            color="#bbbbbb",
            linewidth=1,
            zorder=0,
        )
    for score, color in SCORE_COLORS.items():
        score_rows = data[data["score"] == score]
        axis.errorbar(
            score_rows["value"],
            [y_positions[subset] for subset in score_rows["display_subset"]],
            xerr=score_rows["se"],
            fmt="none",
            capsize=0,
            color=color,
            alpha=0.7,
            linewidth=1.2,
        )
    axis.axvline(0.1, color="#666666", linestyle="--", linewidth=1)
    grid.set_axis_labels("Chromosome-weighted AUPRC", "")
    grid.figure.suptitle("46M ortholog-RAG Mendelian representation probe")
    grid.figure.text(
        0.5,
        0.015,
        "Same chromosome-held-out variants for both scores; error bars = ±1 SE; "
        "dashed line = 10% prevalence.",
        ha="center",
        fontsize=10,
    )
    grid.figure.subplots_adjust(top=0.88, bottom=0.14)
    grid.figure.savefig(output_dir / "figure.svg", bbox_inches="tight")
    grid.figure.savefig(output_dir / "figure.png", dpi=180, bbox_inches="tight")
    plt.close(grid.figure)


def main() -> None:
    args = parse_args()
    comparison = load_comparison(args.input)
    plot_comparison(comparison, args.output_dir)
    print(comparison.sort("subset", "score_type"))


if __name__ == "__main__":
    main()
