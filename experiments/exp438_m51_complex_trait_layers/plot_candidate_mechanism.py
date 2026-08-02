"""Plot feature 1662's post-hoc missense label enrichment by response decile."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

DISPLAY_NAMES = {
    "forward_abs_delta": "FWD |Δ|",
    "reverse_complement_abs_delta": "RC |Δ|",
    "mean_abs_delta": "Mean FWD/RC |Δ|",
    "max_abs_delta": "Max FWD/RC |Δ|",
}


def run(input_path: Path, output_dir: Path) -> None:
    assert input_path.is_file()
    output_dir.mkdir(parents=True, exist_ok=False)
    frame = pl.read_parquet(input_path).with_columns(
        pl.col("response").replace_strict(DISPLAY_NAMES).alias("response_label")
    )
    assert frame.height == 40
    sns.set_theme(style="whitegrid", context="talk")
    grid = sns.relplot(
        data=frame.to_pandas(),
        x="decile",
        y="positive_fraction",
        col="response_label",
        col_wrap=2,
        col_order=list(DISPLAY_NAMES.values()),
        kind="line",
        marker="o",
        color="#3264a8",
        height=3.4,
        aspect=1.35,
        facet_kws={"sharex": True, "sharey": True},
    )
    grid.set_axis_labels("Feature-response decile", "Causal-label fraction")
    grid.set_titles("{col_name}")
    grid.set(xlim=(0.7, 10.3), ylim=(0.04, 0.235), xticks=range(1, 11))
    for axis in grid.axes.flat:
        axis.axhline(0.10, color="#555555", linestyle="--", linewidth=1.2)
    grid.figure.suptitle(
        "Block-19 feature 1662: complex-trait missense enrichment",
        y=1.04,
        fontsize=18,
    )
    grid.figure.text(
        0.5,
        -0.01,
        "Post-hoc description; 250 variants per decile; dashed line = 10% prevalence",
        ha="center",
        fontsize=11,
    )
    grid.figure.savefig(
        output_dir / "feature1662_missense_deciles.svg", bbox_inches="tight"
    )
    grid.figure.savefig(
        output_dir / "feature1662_missense_deciles.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(grid.figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.input, args.output_dir)


if __name__ == "__main__":
    main()
