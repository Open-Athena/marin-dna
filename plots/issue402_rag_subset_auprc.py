#!/usr/bin/env python3
"""Plot issue #402 final-checkpoint AUPRC by biological subset."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

ROOTS = {
    "46M": (
        "gs://marin-us-east5/evals/dna-exp402-rag-h640-p46m-30k/2026.07.26/step-29999"
    ),
    "104M": (
        "gs://marin-us-east5/evals/dna-exp402-rag-h768-p104m-30k/2026.07.26/step-29999"
    ),
}
MODEL_ORDER = ["46M", "104M"]
MODEL_COLORS = {"46M": "#3366cc", "104M": "#d95f02"}
BENCHMARK_ORDER = ["Mendelian", "Complex", "SGE"]
BENCHMARK_SPECS = {
    "Mendelian": ("mendelian_traits", "minus_llr_avg"),
    "Complex": ("complex_traits", "abs_llr_avg"),
    "SGE": ("sge", "minus_llr_avg"),
}
SUBSET_ORDERS = {
    "Mendelian": [
        "missense_variant",
        "synonymous_variant",
        "splicing",
        "5_prime_UTR_variant",
        "tss_proximal",
        "non_coding_transcript_exon_variant",
        "distal",
        "3_prime_UTR_variant",
        "mature_miRNA_variant",
    ],
    "Complex": [
        "distal",
        "missense_variant",
        "3_prime_UTR_variant",
        "tss_proximal",
        "non_coding_transcript_exon_variant",
        "5_prime_UTR_variant",
        "synonymous_variant",
        "splicing",
    ],
    "SGE": ["missense_variant", "splicing", "both"],
}
SUBSET_LABELS = {
    "missense_variant": "Missense",
    "synonymous_variant": "Synonymous",
    "splicing": "Splicing",
    "5_prime_UTR_variant": "5′ UTR",
    "tss_proximal": "TSS proximal",
    "non_coding_transcript_exon_variant": "Non-coding exon",
    "distal": "Distal",
    "3_prime_UTR_variant": "3′ UTR",
    "mature_miRNA_variant": "Mature miRNA",
    "both": "Combined",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/output/issue402_rag_subset_auprc"),
    )
    return parser.parse_args()


def _load_benchmark(model: str, benchmark: str) -> pl.DataFrame:
    slug, score_type = BENCHMARK_SPECS[benchmark]
    metrics = pl.read_parquet(f"{ROOTS[model]}/{slug}/metrics.parquet")
    if benchmark == "SGE":
        selected = metrics.filter(
            (pl.col("metric") == "AUPRC")
            & (pl.col("accession") == "_macro_avg_")
            & (pl.col("gene") == "_macro_avg_")
            & (pl.col("score_type") == score_type)
            & (pl.col("subset") != "_macro_avg_")
        ).select(
            "subset",
            pl.col("value").alias("auprc"),
            "se",
            pl.col("n").alias("n_units"),
            pl.col("n_pos").alias("n_positive"),
        )
        support_kind = "accessions"
    else:
        selected = metrics.filter(
            (pl.col("score_type") == score_type)
            & ~pl.col("subset").is_in(["_global_", "_macro_avg_"])
        ).select(
            "subset",
            pl.col("value").alias("auprc"),
            "se",
            pl.col("n_groups").alias("n_units"),
            (pl.col("n_rows") // pl.lit(10)).alias("n_positive"),
        )
        support_kind = "matched groups"
    expected = SUBSET_ORDERS[benchmark]
    assert selected.height == len(expected)
    assert set(selected["subset"]) == set(expected)
    assert selected.filter(
        ~pl.col("auprc").is_finite()
        | ~pl.col("se").is_finite()
        | (pl.col("n_units") <= 0)
        | (pl.col("n_positive") <= 0)
    ).is_empty()
    return selected.with_columns(
        pl.lit(model).alias("model"),
        pl.lit(benchmark).alias("benchmark"),
        pl.lit(score_type).alias("score_type"),
        pl.lit(support_kind).alias("support_kind"),
        pl.col("subset").replace_strict(SUBSET_LABELS).alias("subset_label"),
        pl.col("subset")
        .replace_strict({subset: index for index, subset in enumerate(expected)})
        .cast(pl.Int64)
        .alias("subset_index"),
    )


def load_subset_metrics() -> pl.DataFrame:
    """Load every non-aggregate subset row at the corrected final checkpoint."""
    data = pl.concat(
        [
            _load_benchmark(model, benchmark)
            for model in MODEL_ORDER
            for benchmark in BENCHMARK_ORDER
        ]
    ).with_columns(
        (
            (pl.col("benchmark") != "Mendelian")
            | (pl.col("subset") != "mature_miRNA_variant")
        ).alias("plot_eligible")
    )
    expected_rows = len(MODEL_ORDER) * sum(map(len, SUBSET_ORDERS.values()))
    assert data.height == expected_rows
    assert data.filter(~pl.col("plot_eligible")).select(
        "model", "subset", "n_units", "auprc"
    ).sort("model").to_dicts() == [
        {
            "model": "104M",
            "subset": "mature_miRNA_variant",
            "n_units": 1,
            "auprc": 1.0,
        },
        {
            "model": "46M",
            "subset": "mature_miRNA_variant",
            "n_units": 1,
            "auprc": 1.0,
        },
    ]
    return data.sort("benchmark", "subset_index", "model")


def plot_subset_metrics(data: pl.DataFrame, output_dir: Path) -> None:
    """Render supported subsets with capless bootstrap-SE bars."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data.write_parquet(output_dir / "metrics.parquet", compression="zstd")
    plotted = data.filter(pl.col("plot_eligible"))
    frame = plotted.to_pandas()
    sns.set_theme(style="whitegrid", context="talk")
    grid = sns.relplot(
        data=frame,
        x="auprc",
        y="subset_index",
        hue="model",
        hue_order=MODEL_ORDER,
        style="model",
        style_order=MODEL_ORDER,
        col="benchmark",
        col_order=BENCHMARK_ORDER,
        kind="scatter",
        markers=True,
        palette=MODEL_COLORS,
        s=90,
        height=5.1,
        aspect=1.05,
        facet_kws={"sharex": False, "sharey": False},
    )
    grid.set_axis_labels("", "")
    for benchmark, axis in grid.axes_dict.items():
        subset = frame[frame["benchmark"] == benchmark]
        labels = [
            SUBSET_LABELS[name]
            for name in SUBSET_ORDERS[benchmark]
            if not (benchmark == "Mendelian" and name == "mature_miRNA_variant")
        ]
        axis.set_title(benchmark)
        axis.set_yticks(range(len(labels)), labels)
        axis.invert_yaxis()
        for model in MODEL_ORDER:
            model_rows = subset[subset["model"] == model].sort_values("subset_index")
            axis.errorbar(
                model_rows["auprc"],
                model_rows["subset_index"],
                xerr=model_rows["se"],
                fmt="none",
                ecolor=MODEL_COLORS[model],
                elinewidth=1.2,
                capsize=0,
                alpha=0.85,
            )
    grid.figure.suptitle("Final corrected AUPRC by annotated subset")
    grid.figure.supxlabel("AUPRC", y=0.085)
    grid.figure.text(
        0.5,
        0.01,
        "Step 29,999 (30,000 updates); points ±1 bootstrap SE. SGE is "
        "macro-averaged across three "
        "accessions; benchmark x-axes are independent.\nMature miRNA (one matched group, "
        "AUPRC 1.0) is reported in the table only.",
        ha="center",
        fontsize=10,
    )
    grid.figure.subplots_adjust(top=0.84, bottom=0.22, wspace=0.75)
    grid.figure.savefig(output_dir / "figure.svg", bbox_inches="tight")
    grid.figure.savefig(output_dir / "figure.png", dpi=180, bbox_inches="tight")
    plt.close(grid.figure)


def main() -> None:
    args = parse_args()
    data = load_subset_metrics()
    plot_subset_metrics(data, args.output_dir)
    print(
        data.select(
            "benchmark",
            "subset",
            "model",
            "auprc",
            "se",
            "n_units",
            "n_positive",
            "support_kind",
        )
    )


if __name__ == "__main__":
    main()
