#!/usr/bin/env python3
"""Audit and visualize issue #402 46M step-15k to step-20k score drift."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

DEFAULT_ROOT = "gs://marin-us-east5/evals/dna-exp402-rag-h640-p46m-b2m-30k/2026.07.26.5"
BENCHMARK_SPECS = {
    "Mendelian": ("mendelian_traits", "minus_llr_avg"),
    "Complex": ("complex_traits", "abs_llr_avg"),
    "SGE": ("sge", "minus_llr_avg"),
}
BENCHMARK_ORDER = list(BENCHMARK_SPECS)
DYNAMIC_COLUMNS = {
    "ref_loglikelihood_fwd",
    "alt_loglikelihood_fwd",
    "llr_fwd",
    "ref_loglikelihood_rc",
    "alt_loglikelihood_rc",
    "llr_rc",
    "ref_loglikelihood_avg",
    "alt_loglikelihood_avg",
    "llr_avg",
    "minus_llr_avg",
    "abs_llr_avg",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=DEFAULT_ROOT)
    parser.add_argument("--step-a", type=int, default=15_000)
    parser.add_argument("--step-b", type=int, default=20_000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/output/issue402_rag_large_batch_checkpoint_drift"),
    )
    return parser.parse_args()


def _headline(metrics: pl.DataFrame, benchmark: str, score_column: str) -> float:
    if benchmark == "SGE":
        selected = metrics.filter(
            (pl.col("metric") == "AUPRC")
            & (pl.col("subset") == "_macro_avg_")
            & (pl.col("accession") == "_macro_avg_")
            & (pl.col("gene") == "_macro_avg_")
            & (pl.col("score_type") == score_column)
        )
    else:
        selected = metrics.filter(
            (pl.col("subset") == "_global_") & (pl.col("score_type") == score_column)
        )
    assert selected.height == 1, selected
    value = float(selected["value"].item())
    assert math.isfinite(value) and 0 <= value <= 1
    return value


def _subset_metrics(
    metrics: pl.DataFrame,
    benchmark: str,
    score_column: str,
) -> pl.DataFrame:
    if benchmark == "SGE":
        selected = metrics.filter(
            (pl.col("metric") == "AUPRC")
            & (pl.col("accession") == "_macro_avg_")
            & (pl.col("gene") == "_macro_avg_")
            & (pl.col("score_type") == score_column)
            & (pl.col("subset") != "_macro_avg_")
        )
    else:
        selected = metrics.filter(
            (pl.col("score_type") == score_column)
            & ~pl.col("subset").is_in(["_global_", "_macro_avg_"])
        )
    result = selected.select("subset", pl.col("value").alias("auprc"), "se")
    assert result.height > 0
    assert result["subset"].n_unique() == result.height
    assert result.filter(
        ~pl.col("auprc").is_finite()
        | ~pl.col("se").is_finite()
        | (pl.col("auprc") < 0)
        | (pl.col("auprc") > 1)
        | (pl.col("se") < 0)
    ).is_empty()
    return result


def _read_checkpoint(
    input_root: str,
    step: int,
    slug: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    root = f"{input_root.rstrip('/')}/step-{step}/{slug}"
    return (
        pl.read_parquet(f"{root}/variants.parquet"),
        pl.read_parquet(f"{root}/metrics.parquet"),
    )


def _rank_correlation(frame: pl.DataFrame) -> float:
    ranked = frame.select(
        pl.col("score_a").rank("average").alias("rank_a"),
        pl.col("score_b").rank("average").alias("rank_b"),
    )
    value = float(ranked.select(pl.corr("rank_a", "rank_b")).item())
    assert math.isfinite(value) and -1 <= value <= 1
    return value


def _top_fraction_overlap(frame: pl.DataFrame, fraction: float) -> float:
    assert 0 < fraction < 1
    count = math.ceil(frame.height * fraction)
    top_a = set(
        frame.sort("score_a", descending=True).head(count)["variant_id"].to_list()
    )
    top_b = set(
        frame.sort("score_b", descending=True).head(count)["variant_id"].to_list()
    )
    assert len(top_a) == len(top_b) == count
    return len(top_a & top_b) / count


def load_checkpoint_comparison(
    input_root: str,
    step_a: int,
    step_b: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Assert immutable-row parity and summarize per-variant score movement."""
    assert 0 < step_a < step_b
    comparisons: list[pl.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    subset_frames: list[pl.DataFrame] = []
    for benchmark, (slug, score_column) in BENCHMARK_SPECS.items():
        variants_a, metrics_a = _read_checkpoint(input_root, step_a, slug)
        variants_b, metrics_b = _read_checkpoint(input_root, step_b, slug)
        assert variants_a.schema == variants_b.schema
        assert variants_a.height == variants_b.height
        assert variants_a["variant_id"].n_unique() == variants_a.height
        assert variants_b["variant_id"].n_unique() == variants_b.height

        immutable_columns = [
            column for column in variants_a.columns if column not in DYNAMIC_COLUMNS
        ]
        immutable_a = variants_a.sort("variant_id").select(immutable_columns)
        immutable_b = variants_b.sort("variant_id").select(immutable_columns)
        assert immutable_a.equals(immutable_b), (
            benchmark,
            "variant metadata or labels differ between checkpoints",
        )

        comparison = (
            variants_a.select(
                "variant_id",
                "label",
                "subset",
                pl.col(score_column).alias("score_a"),
            )
            .join(
                variants_b.select(
                    "variant_id",
                    pl.col(score_column).alias("score_b"),
                ),
                on="variant_id",
                how="inner",
                validate="1:1",
            )
            .with_columns(
                pl.lit(benchmark).alias("benchmark"),
                pl.when("label")
                .then(pl.lit("positive"))
                .otherwise(pl.lit("control"))
                .alias("label_name"),
                (pl.col("score_b") - pl.col("score_a")).alias("score_delta"),
            )
        )
        assert comparison.height == variants_a.height
        assert comparison.filter(
            ~pl.col("score_a").is_finite()
            | ~pl.col("score_b").is_finite()
            | ~pl.col("score_delta").is_finite()
        ).is_empty()

        pearson = float(comparison.select(pl.corr("score_a", "score_b")).item())
        spearman = _rank_correlation(comparison)
        abs_delta = comparison.select(pl.col("score_delta").abs().alias("value"))
        auprc_a = _headline(metrics_a, benchmark, score_column)
        auprc_b = _headline(metrics_b, benchmark, score_column)
        summary_rows.append(
            {
                "benchmark": benchmark,
                "step_a": step_a,
                "step_b": step_b,
                "n_variants": comparison.height,
                "n_positive": comparison.filter("label").height,
                "pearson": pearson,
                "spearman": spearman,
                "top_10pct_overlap": _top_fraction_overlap(comparison, 0.1),
                "median_abs_score_delta": float(abs_delta["value"].median()),
                "p95_abs_score_delta": float(
                    abs_delta["value"].quantile(0.95, interpolation="linear")
                ),
                "score_std_a": float(comparison["score_a"].std()),
                "score_std_b": float(comparison["score_b"].std()),
                "headline_auprc_a": auprc_a,
                "headline_auprc_b": auprc_b,
                "headline_auprc_delta": auprc_b - auprc_a,
            }
        )
        subset_frames.append(
            _subset_metrics(metrics_a, benchmark, score_column)
            .rename({"auprc": "auprc_a", "se": "se_a"})
            .join(
                _subset_metrics(metrics_b, benchmark, score_column).rename(
                    {"auprc": "auprc_b", "se": "se_b"}
                ),
                on="subset",
                how="inner",
                validate="1:1",
            )
            .with_columns(
                pl.lit(benchmark).alias("benchmark"),
                (pl.col("auprc_b") - pl.col("auprc_a")).alias("auprc_delta"),
            )
        )
        comparisons.append(comparison)

    summary = pl.DataFrame(summary_rows).sort("benchmark")
    assert summary.height == len(BENCHMARK_SPECS)
    assert summary.filter(
        ~pl.col("pearson").is_finite()
        | ~pl.col("spearman").is_finite()
        | ~pl.col("top_10pct_overlap").is_finite()
        | (pl.col("top_10pct_overlap") < 0)
        | (pl.col("top_10pct_overlap") > 1)
    ).is_empty()
    return (
        pl.concat(comparisons).sort("benchmark", "variant_id"),
        summary,
        pl.concat(subset_frames).sort("benchmark", "subset"),
    )


def plot_checkpoint_comparison(
    comparison: pl.DataFrame,
    summary: pl.DataFrame,
    step_a: int,
    step_b: int,
    output_dir: Path,
) -> None:
    """Render a deterministic display sample with exact statistics annotated."""
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.write_parquet(output_dir / "variants.parquet", compression="zstd")
    summary.write_parquet(output_dir / "summary.parquet", compression="zstd")

    display = pl.concat(
        [
            comparison.filter(pl.col("benchmark") == benchmark).sample(
                n=min(
                    5_000,
                    comparison.filter(pl.col("benchmark") == benchmark).height,
                ),
                shuffle=True,
                seed=402,
            )
            for benchmark in BENCHMARK_ORDER
        ]
    )
    frame = display.to_pandas()
    sns.set_theme(style="whitegrid", context="talk")
    grid = sns.relplot(
        data=frame,
        x="score_a",
        y="score_b",
        hue="label_name",
        hue_order=["control", "positive"],
        col="benchmark",
        col_order=BENCHMARK_ORDER,
        kind="scatter",
        palette={"control": "#737373", "positive": "#d95f02"},
        alpha=0.25,
        s=16,
        height=4.8,
        aspect=1.0,
        facet_kws={"sharex": False, "sharey": False},
    )
    grid.set_axis_labels(
        f"VEP score at {step_a // 1_000}k", f"VEP score at {step_b // 1_000}k"
    )
    for benchmark, axis in grid.axes_dict.items():
        benchmark_frame = frame[frame["benchmark"] == benchmark]
        low = min(
            benchmark_frame["score_a"].quantile(0.005),
            benchmark_frame["score_b"].quantile(0.005),
        )
        high = max(
            benchmark_frame["score_a"].quantile(0.995),
            benchmark_frame["score_b"].quantile(0.995),
        )
        assert math.isfinite(low) and math.isfinite(high) and low < high
        axis.set_xlim(low, high)
        axis.set_ylim(low, high)
        axis.plot([low, high], [low, high], color="#252525", linestyle=":", linewidth=1)
        row = summary.filter(pl.col("benchmark") == benchmark).row(0, named=True)
        axis.set_title(benchmark)
        axis.text(
            0.03,
            0.97,
            f"Spearman = {row['spearman']:.3f}\n"
            f"top-decile overlap = {row['top_10pct_overlap']:.1%}\n"
            f"ΔAUPRC = {row['headline_auprc_delta']:+.4f}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
    grid.figure.suptitle(
        f"46M ortholog-RAG per-variant score drift: {step_a // 1_000}k → "
        f"{step_b // 1_000}k"
    )
    grid.figure.text(
        0.5,
        0.01,
        "Display: deterministic sample of ≤5,000 variants/benchmark, clipped to "
        "the displayed 0.5–99.5% score range. Statistics use every variant. "
        "Mendelian/SGE score = −mean LLR; Complex = |mean LLR|.",
        ha="center",
        fontsize=9.5,
    )
    grid.figure.subplots_adjust(top=0.82, bottom=0.17, wspace=0.3)
    grid.figure.savefig(output_dir / "figure.svg", bbox_inches="tight")
    grid.figure.savefig(output_dir / "figure.png", dpi=180, bbox_inches="tight")
    plt.close(grid.figure)


def main() -> None:
    args = parse_args()
    comparison, summary, subset_deltas = load_checkpoint_comparison(
        args.input_root,
        args.step_a,
        args.step_b,
    )
    plot_checkpoint_comparison(
        comparison,
        summary,
        args.step_a,
        args.step_b,
        args.output_dir,
    )
    subset_deltas.write_parquet(
        args.output_dir / "subset_deltas.parquet",
        compression="zstd",
    )
    print(summary)
    print(subset_deltas)


if __name__ == "__main__":
    main()
