#!/usr/bin/env python3
"""Compare issue #402 final RAG checkpoints with mammalian phyloP by subset."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

RAG_ROOTS = {
    "RAG 46M": (
        "gs://marin-us-east5/users/ubuntu/evals/"
        "dna-exp402-rag-h640-p46m-1b/ropefix/step-7628"
    ),
    "RAG 104M": (
        "gs://marin-us-east5/users/ubuntu/evals/"
        "dna-exp402-rag-h768-p104m-1b/ropefix/step-7628"
    ),
}
PHYLOP_ROOT = (
    "gs://marin-us-east5/users/ubuntu/evals/"
    "dna-exp402-rag-phylop447m/exact-test-a57a69c"
)
METHOD_ORDER = ["RAG 46M", "RAG 104M", "phyloP 447-way"]
METHOD_COLORS = {
    "RAG 46M": "#3366cc",
    "RAG 104M": "#d95f02",
    "phyloP 447-way": "#238b45",
}
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
        default=Path("plots/output/issue402_rag_vs_phylop_subset_auprc"),
    )
    return parser.parse_args()


def _select_subset_rows(
    metrics: pl.DataFrame,
    *,
    benchmark: str,
    score_type: str,
) -> pl.DataFrame:
    """Select non-aggregate subset metrics with comparable support columns."""
    if benchmark == "SGE":
        return metrics.filter(
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
    return metrics.filter(
        (pl.col("score_type") == score_type)
        & ~pl.col("subset").is_in(["_global_", "_macro_avg_"])
    ).select(
        "subset",
        pl.col("value").alias("auprc"),
        "se",
        pl.col("n_groups").alias("n_units"),
        (pl.col("n_rows") // pl.lit(10)).alias("n_positive"),
    )


def _load_method(method: str, benchmark: str) -> pl.DataFrame:
    slug, rag_score_type = BENCHMARK_SPECS[benchmark]
    is_phylop = method == "phyloP 447-way"
    root = PHYLOP_ROOT if is_phylop else RAG_ROOTS[method]
    score_type = "score" if is_phylop else rag_score_type
    selected = _select_subset_rows(
        pl.read_parquet(f"{root}/{slug}/metrics.parquet"),
        benchmark=benchmark,
        score_type=score_type,
    )
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
        pl.lit(method).alias("method"),
        pl.lit(benchmark).alias("benchmark"),
        pl.lit(score_type).alias("score_type"),
        pl.col("subset").replace_strict(SUBSET_LABELS).alias("subset_label"),
        pl.col("subset")
        .replace_strict({subset: index for index, subset in enumerate(expected)})
        .cast(pl.Int64)
        .alias("subset_index"),
    )


def load_subset_metrics() -> pl.DataFrame:
    """Load exact-row RAG and phyloP metrics and assert support parity."""
    data = pl.concat(
        [
            _load_method(method, benchmark)
            for method in METHOD_ORDER
            for benchmark in BENCHMARK_ORDER
        ]
    ).with_columns(
        (
            (pl.col("benchmark") != "Mendelian")
            | (pl.col("subset") != "mature_miRNA_variant")
        ).alias("plot_eligible")
    )
    expected_rows = len(METHOD_ORDER) * sum(map(len, SUBSET_ORDERS.values()))
    assert data.height == expected_rows
    support = (
        data.group_by("benchmark", "subset")
        .agg(
            pl.col("n_units").n_unique().alias("n_unique_units"),
            pl.col("n_positive").n_unique().alias("n_unique_positive"),
            pl.len().alias("n_methods"),
        )
        .sort("benchmark", "subset")
    )
    assert support.filter(
        (pl.col("n_unique_units") != 1)
        | (pl.col("n_unique_positive") != 1)
        | (pl.col("n_methods") != len(METHOD_ORDER))
    ).is_empty(), support
    assert data.filter(~pl.col("plot_eligible")).select(
        "method", "n_units", "auprc"
    ).sort("method").to_dicts() == [
        {"method": "RAG 104M", "n_units": 1, "auprc": 1.0},
        {"method": "RAG 46M", "n_units": 1, "auprc": 1.0},
        {"method": "phyloP 447-way", "n_units": 1, "auprc": 1.0},
    ]
    return data.sort("benchmark", "subset_index", "method")


def plot_subset_metrics(data: pl.DataFrame, output_dir: Path) -> None:
    """Render exact-row per-subset AUPRC with capless bootstrap-SE bars."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data.write_parquet(output_dir / "metrics.parquet", compression="zstd")
    frame = data.filter(pl.col("plot_eligible")).to_pandas()
    sns.set_theme(style="whitegrid", context="talk")
    grid = sns.relplot(
        data=frame,
        x="auprc",
        y="subset_index",
        hue="method",
        hue_order=METHOD_ORDER,
        style="method",
        style_order=METHOD_ORDER,
        col="benchmark",
        col_order=BENCHMARK_ORDER,
        kind="scatter",
        markers=True,
        palette=METHOD_COLORS,
        s=85,
        height=5.25,
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
        for method in METHOD_ORDER:
            method_rows = subset[subset["method"] == method].sort_values("subset_index")
            axis.errorbar(
                method_rows["auprc"],
                method_rows["subset_index"],
                xerr=method_rows["se"],
                fmt="none",
                ecolor=METHOD_COLORS[method],
                elinewidth=1.1,
                capsize=0,
                alpha=0.85,
            )
    grid.figure.suptitle("Final RAG checkpoints versus mammalian phyloP by subset")
    grid.figure.supxlabel("AUPRC", y=0.105)
    grid.figure.text(
        0.5,
        0.012,
        "Exact same test rows; points ±1 bootstrap SE. Complex RAG = "
        "|mean(forward LLR, RC LLR)|; Mendelian/SGE RAG = −mean LLR; "
        "phyloP = raw signed score.\nSGE is macro-averaged across three accessions; "
        "benchmark x-axes are independent. Mature miRNA (one group, AUPRC 1.0) "
        "is table-only.",
        ha="center",
        fontsize=9.5,
    )
    grid.figure.subplots_adjust(top=0.83, bottom=0.25, wspace=0.75)
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
            "method",
            "auprc",
            "se",
            "n_units",
            "n_positive",
            "score_type",
        )
    )


if __name__ == "__main__":
    main()
