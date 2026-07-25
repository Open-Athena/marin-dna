#!/usr/bin/env python3
"""Plot issue #402 human-to-ortholog attention by aligned base offset."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

SANITY_ROOTS = {
    "46M": (
        "gs://marin-us-east5/users/ubuntu/evals/"
        "dna-exp402-rag-h640-p46m-1b/sanity-c274a04"
    ),
    "104M": (
        "gs://marin-us-east5/users/ubuntu/evals/"
        "dna-exp402-rag-h768-p104m-1b/sanity-c274a04"
    ),
}
MODEL_ORDER = ["46M", "104M"]
MODEL_COLORS = {"46M": "#3366cc", "104M": "#d95f02"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/output/issue402_rag_attention_alignment"),
    )
    return parser.parse_args()


def load_alignment() -> pl.DataFrame:
    """Pool attention over layers, slots, sampled queries, and documents."""
    source = pl.concat(
        [
            pl.read_parquet(f"{root}/attention_alignment.parquet")
            for root in SANITY_ROOTS.values()
        ]
    ).with_columns((pl.col("n_documents") * pl.col("n_query_offsets")).alias("weight"))
    assert source.height > 20_000
    assert sorted(source["model"].unique()) == sorted(SANITY_ROOTS)
    pooled = (
        source.group_by("model", "availability", "offset")
        .agg(
            (
                (pl.col("mean_attention") * pl.col("weight")).sum()
                / pl.col("weight").sum()
            ).alias("mean_attention")
        )
        .with_columns(
            pl.when(pl.col("availability") == "available")
            .then(pl.lit("Projected bases"))
            .otherwise(pl.lit("Missing (N) slot"))
            .alias("Ortholog slot")
        )
        .sort("model", "availability", "offset")
    )
    assert pooled.height == len(SANITY_ROOTS) * 2 * 65
    assert pooled.filter(pl.col("mean_attention") <= 0).is_empty()
    return pooled


def plot_alignment(data: pl.DataFrame, output_dir: Path) -> None:
    """Render the corresponding-position peak with missing-slot control."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data.write_parquet(output_dir / "metrics.parquet", compression="zstd")
    frame = data.to_pandas()
    sns.set_theme(style="whitegrid", context="talk")
    grid = sns.relplot(
        data=frame,
        x="offset",
        y="mean_attention",
        hue="model",
        hue_order=MODEL_ORDER,
        style="Ortholog slot",
        style_order=["Projected bases", "Missing (N) slot"],
        kind="line",
        dashes={"Projected bases": "", "Missing (N) slot": (3, 2)},
        palette=MODEL_COLORS,
        height=5.0,
        aspect=1.65,
    )
    axis = grid.ax
    axis.set_yscale("log")
    axis.axvline(1, color="#238b45", linestyle=":", linewidth=1.5)
    axis.annotate(
        "causal aligned target (+1)",
        xy=(1, 0.0083),
        xytext=(7, 0.011),
        arrowprops={"arrowstyle": "->", "color": "#238b45"},
        color="#176c35",
        fontsize=10,
    )
    grid.set_axis_labels(
        "Ortholog key offset relative to human query", "Mean attention (log scale)"
    )
    grid.figure.suptitle(
        "Human queries attend most strongly to the predictive aligned ortholog base"
    )
    grid.figure.text(
        0.5,
        0.015,
        "Mean over all heads/layers, seven slots, four documents, and every fourth "
        "human query. At query j, causal LM loss predicts base j+1.",
        ha="center",
        fontsize=10,
    )
    grid.figure.subplots_adjust(top=0.84, bottom=0.18)
    grid.figure.savefig(output_dir / "figure.svg", bbox_inches="tight")
    grid.figure.savefig(output_dir / "figure.png", dpi=180, bbox_inches="tight")
    plt.close(grid.figure)


def main() -> None:
    args = parse_args()
    data = load_alignment()
    plot_alignment(data, args.output_dir)
    print(
        data.sort(
            "model", "Ortholog slot", "mean_attention", descending=[False, False, True]
        )
        .group_by("model", "Ortholog slot", maintain_order=True)
        .head(5)
    )


if __name__ == "__main__":
    main()
