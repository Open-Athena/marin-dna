#!/usr/bin/env python3
"""Plot issue #402 next-token loss by position within each aligned segment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

SANITY_ROOTS = {
    "46M": (
        "gs://marin-us-east5/evals/"
        "dna-exp402-rag-h640-p46m-30k/2026.07.26/sanity-ac7016"
    ),
    "104M": (
        "gs://marin-us-east5/evals/"
        "dna-exp402-rag-h768-p104m-30k/2026.07.26/sanity-ac7016"
    ),
}
MODEL_ORDER = ["46M", "104M"]
BASE_TOKEN_TYPES = ["ortholog_base", "human_base"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/output/issue402_rag_validation_position"),
    )
    return parser.parse_args()


def load_position_loss() -> pl.DataFrame:
    """Load and smooth the frozen per-position validation summaries."""
    frames = [
        pl.read_parquet(f"{root}/validation_position_loss.parquet")
        for root in SANITY_ROOTS.values()
    ]
    data = pl.concat(frames).sort("model", "segment_index", "within_segment_offset")
    assert data.height == len(SANITY_ROOTS) * 2_047
    assert sorted(data["model"].unique()) == sorted(SANITY_ROOTS)
    bases = (
        data.filter(pl.col("layout_token_type").is_in(BASE_TOKEN_TYPES))
        .with_columns(
            pl.col("mean_loss")
            .rolling_mean(window_size=15, center=True, min_samples=1)
            .over("model", "segment_index")
            .alias("smoothed_loss")
        )
        .sort("model", "segment_index", "within_segment_offset")
    )
    assert bases.height == len(SANITY_ROOTS) * 8 * 255
    assert bases.filter(~pl.col("smoothed_loss").is_finite()).is_empty()
    return bases


def plot_position_loss(data: pl.DataFrame, output_dir: Path) -> None:
    """Render aligned within-segment loss curves for both model sizes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data.write_parquet(output_dir / "metrics.parquet", compression="zstd")
    frame = data.to_pandas()
    sns.set_theme(style="whitegrid", context="talk")
    grid = sns.relplot(
        data=frame,
        x="within_segment_offset",
        y="smoothed_loss",
        hue="segment_index",
        palette="viridis",
        hue_norm=(0, 7),
        col="model",
        col_order=MODEL_ORDER,
        kind="line",
        estimator=None,
        height=4.6,
        aspect=1.25,
        facet_kws={"sharex": True, "sharey": True},
    )
    grid.set_axis_labels("", "Token NLL")
    grid.set_titles("{col_name}")
    grid.set(xlim=(0, 254))
    if grid.legend is not None:
        grid.legend.set_title("Segment index")
    grid.figure.suptitle(
        "Next-token validation loss falls within aligned sequence segments"
    )
    grid.figure.supxlabel("Left-to-right offset within 255-base segment", y=0.075)
    grid.figure.text(
        0.5,
        0.015,
        "15-position centered mean over 2,048 validation documents. "
        "Segments 0–6 are orthologs in retrieval order; segment 7 is human.",
        ha="center",
        fontsize=10,
    )
    grid.figure.subplots_adjust(top=0.84, bottom=0.21)
    grid.figure.savefig(output_dir / "figure.svg", bbox_inches="tight")
    grid.figure.savefig(output_dir / "figure.png", dpi=180, bbox_inches="tight")
    plt.close(grid.figure)


def main() -> None:
    args = parse_args()
    data = load_position_loss()
    plot_position_loss(data, args.output_dir)
    print(
        data.group_by("model", "segment_index", "segment")
        .agg(pl.col("mean_loss").mean().alias("mean_loss"))
        .sort("model", "segment_index")
    )


if __name__ == "__main__":
    main()
