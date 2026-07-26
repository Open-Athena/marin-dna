#!/usr/bin/env python3
"""Summarize issue #402's reproducible final 30k plot metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-root", type=Path, default=Path("plots/output"))
    return parser.parse_args()


def _read(plot_root: Path, recipe: str, name: str = "metrics.parquet") -> pl.DataFrame:
    path = plot_root / recipe / name
    assert path.is_file(), path
    return pl.read_parquet(path)


def summarize_curves(
    plot_root: Path,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    curves = _read(plot_root, "issue402_rag_scaling_curve")
    headline = curves.filter(
        (
            pl.col("benchmark").is_in(["Mendelian", "Complex"])
            & (pl.col("aggregate") == "global")
        )
        | ((pl.col("benchmark") == "SGE") & (pl.col("aggregate") == "macro"))
    )
    assert headline.height == 2 * 30 * 3
    final = headline.filter(pl.col("step") == 29_999).sort("benchmark", "model_size")
    assert final.height == 6
    best = (
        headline.sort(
            "model_size",
            "benchmark",
            "value",
            descending=[False, False, True],
        )
        .group_by("model_size", "benchmark", maintain_order=True)
        .head(1)
        .sort("benchmark", "model_size")
    )
    assert best.height == 6
    joined = _read(plot_root, "issue402_rag_30k_loss_auprc")
    trends = (
        joined.group_by("model", "benchmark")
        .agg(
            pl.corr("step", "auprc", method="spearman").alias("spearman_step_auprc"),
            pl.corr("validation_loss", "auprc", method="spearman").alias(
                "spearman_loss_auprc"
            ),
        )
        .sort("benchmark", "model")
    )
    assert trends.height == 6
    return final, best, trends


def summarize_position(plot_root: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    position = _read(plot_root, "issue402_rag_validation_position")
    position_trends = (
        position.group_by("model", "segment_index", "segment")
        .agg(
            pl.corr("within_segment_offset", "mean_loss", method="spearman").alias(
                "spearman_offset_loss"
            ),
            pl.col("mean_loss")
            .filter(pl.col("within_segment_offset") < 64)
            .mean()
            .alias("first_quarter_loss"),
            pl.col("mean_loss")
            .filter(pl.col("within_segment_offset") >= 191)
            .mean()
            .alias("last_quarter_loss"),
        )
        .with_columns(
            (pl.col("last_quarter_loss") - pl.col("first_quarter_loss")).alias(
                "last_minus_first"
            )
        )
        .sort("model", "segment_index")
    )
    human = position_trends.filter(pl.col("segment_index") == 7)
    assert human.height == 2
    return human, position_trends


def summarize_attention(plot_root: Path) -> pl.DataFrame:
    attention = _read(plot_root, "issue402_rag_attention_alignment")
    peaks = (
        attention.sort(
            "model",
            "Ortholog slot",
            "mean_attention",
            descending=[False, False, True],
        )
        .group_by("model", "Ortholog slot", maintain_order=True)
        .head(1)
        .sort("model", "Ortholog slot")
    )
    assert peaks.height == 4
    return peaks


def summarize_baselines(plot_root: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    comparison = _read(plot_root, "issue402_rag_vs_phylop_subset_auprc").filter(
        pl.col("plot_eligible")
    )
    phylop = comparison.filter(pl.col("method") == "phyloP 447-way").select(
        "benchmark", "subset", pl.col("auprc").alias("phylop_auprc")
    )
    rag = comparison.filter(pl.col("method") != "phyloP 447-way").join(
        phylop, on=["benchmark", "subset"], how="inner", validate="m:1"
    )
    wins = (
        rag.group_by("method", "benchmark")
        .agg(
            (pl.col("auprc") > pl.col("phylop_auprc")).sum().alias("rag_wins"),
            pl.len().alias("n_subsets"),
            (pl.col("auprc") - pl.col("phylop_auprc"))
            .mean()
            .alias("mean_auprc_difference"),
        )
        .sort("benchmark", "method")
    )
    assert wins.height == 6
    probe = _read(plot_root, "issue402_rag_probe_comparison").filter(
        pl.col("subset") == "_macro_avg_"
    )
    assert probe.height == 4
    return wins, probe.sort("model_size", "score_type")


def main() -> None:
    args = parse_args()
    final, best, trends = summarize_curves(args.plot_root)
    human_position, all_position = summarize_position(args.plot_root)
    attention_peaks = summarize_attention(args.plot_root)
    baseline_wins, macro_probe = summarize_baselines(args.plot_root)
    context = _read(args.plot_root, "issue402_rag_context_ablation").sort(
        "group", "model", "ablation_index"
    )
    vep_context = _read(args.plot_root, "issue402_rag_vep_context_ablation").sort(
        "benchmark", "model", "context_index"
    )
    indel = _read(
        args.plot_root,
        "issue402_rag_indel_attention",
        "aggregate_metrics.parquet",
    ).sort("model", "scope")

    with pl.Config(tbl_rows=100, tbl_cols=20, tbl_width_chars=180):
        for name, frame in (
            ("FINAL HEADLINE", final),
            ("BEST HEADLINE", best),
            ("SPEARMAN TRENDS", trends),
            ("HUMAN POSITION", human_position),
            ("ALL POSITION", all_position),
            ("ATTENTION PEAKS", attention_peaks),
            ("CONTEXT LOSS", context),
            ("VEP CONTEXT", vep_context),
            ("INDEL ATTENTION", indel),
            ("RAG VS PHYLOP", baseline_wins),
            ("MACRO PROBE", macro_probe),
        ):
            print(f"\n{name}\n{frame}")


if __name__ == "__main__":
    main()
