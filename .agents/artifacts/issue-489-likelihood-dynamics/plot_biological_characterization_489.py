"""Plot compact biological summaries for issue #489 trajectory groups."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent / "biology"
GROUP_ORDER = ("high_to_high", "low_to_high", "high_to_low", "low_to_low")
GROUP_LABELS = {
    "high_to_high": "H\N{RIGHTWARDS ARROW}H",
    "low_to_high": "L\N{RIGHTWARDS ARROW}H",
    "high_to_low": "H\N{RIGHTWARDS ARROW}L",
    "low_to_low": "L\N{RIGHTWARDS ARROW}L",
}
REGION_ORDER = ("cds", "upstream", "downstream", "ncrna", "enhancer")
REGION_LABELS = {
    "cds": "CDS",
    "upstream": "Upstream",
    "downstream": "Downstream",
    "ncrna": "ncRNA",
    "enhancer": "Enhancer",
}


def save(figure: plt.Figure, name: str) -> None:
    for suffix in ("svg", "png"):
        figure.savefig(ROOT / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_region_enrichment(frame: pd.DataFrame) -> None:
    values = frame.pivot(index="group", columns="region", values="region_enrichment_log2")
    values = values.loc[list(GROUP_ORDER), list(REGION_ORDER)]
    values.index = [GROUP_LABELS[value] for value in values.index]
    values.columns = [REGION_LABELS[value] for value in values.columns]
    figure, axis = plt.subplots(figsize=(6, 6))
    sns.heatmap(
        values,
        cmap="vlag",
        center=0,
        vmin=-1.35,
        vmax=1.35,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Log2 enrichment"},
        ax=axis,
    )
    axis.set_xlabel("Validation region")
    axis.set_ylabel("Trajectory group")
    axis.set_title("Region enrichment by trajectory group")
    axis.set_box_aspect(1)
    save(figure, "trajectory_region_enrichment")


def plot_conservation(frame: pd.DataFrame) -> None:
    palette = sns.color_palette(n_colors=len(REGION_ORDER))
    figure, axis = plt.subplots(figsize=(6, 6))
    base_y = np.arange(len(GROUP_ORDER), dtype=float)
    offsets = np.linspace(-0.28, 0.28, len(REGION_ORDER))
    for region_index, region in enumerate(REGION_ORDER):
        values = frame[frame["region"] == region].set_index("group").loc[list(GROUP_ORDER)]
        estimate = values["conservation_prevalence"].to_numpy()
        low = values["conservation_ci_low"].to_numpy()
        high = values["conservation_ci_high"].to_numpy()
        axis.errorbar(
            estimate,
            base_y + offsets[region_index],
            xerr=np.vstack((estimate - low, high - estimate)),
            color=palette[region_index],
            marker="o",
            linestyle="none",
            capsize=2,
            label=REGION_LABELS[region],
        )
    axis.set_yticks(base_y, [GROUP_LABELS[group] for group in GROUP_ORDER])
    axis.invert_yaxis()
    axis.set_xlim(0, 0.85)
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_xlabel("Conserved positions")
    axis.set_ylabel("Trajectory group")
    axis.set_title("Conservation by trajectory group and region")
    axis.grid(axis="x")
    axis.grid(axis="y", visible=False)
    axis.set_box_aspect(1)
    axis.legend(
        title="Validation region",
        loc="upper left",
        bbox_to_anchor=(0.0, -0.18),
        ncol=3,
        borderaxespad=0,
    )
    figure.subplots_adjust(bottom=0.30)
    save(figure, "trajectory_conservation_by_region")


def plot_region_lines(
    frame: pd.DataFrame,
    *,
    column: str,
    ylabel: str,
    title: str,
    output_name: str,
    percentage: bool = False,
) -> None:
    palette = sns.color_palette(n_colors=len(REGION_ORDER))
    figure, axis = plt.subplots(figsize=(6, 6))
    x = np.arange(len(GROUP_ORDER))
    for region_index, region in enumerate(REGION_ORDER):
        values = frame[frame["region"] == region].set_index("group").loc[list(GROUP_ORDER)]
        axis.plot(
            x,
            values[column],
            color=palette[region_index],
            marker="o",
            label=REGION_LABELS[region],
        )
    axis.set_xticks(x, [GROUP_LABELS[group] for group in GROUP_ORDER])
    axis.set_xlabel("Trajectory group")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    if percentage:
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.set_ylim(bottom=0)
    axis.set_box_aspect(1)
    axis.legend(
        title="Validation region",
        loc="upper left",
        bbox_to_anchor=(0.0, -0.18),
        ncol=3,
        borderaxespad=0,
    )
    figure.subplots_adjust(bottom=0.30)
    save(figure, output_name)


def main() -> None:
    sns.set_theme(style="whitegrid")
    frame = pd.read_parquet(ROOT / "region_group_statistics.parquet")
    assert len(frame) == len(GROUP_ORDER) * len(REGION_ORDER)
    plot_region_enrichment(frame)
    plot_conservation(frame)
    plot_region_lines(
        frame,
        column="mean_kmer7_nll",
        ylabel="Mean held-out 7-mer NLL",
        title="Local sequence predictability by trajectory group",
        output_name="trajectory_kmer7_nll",
    )
    plot_region_lines(
        frame,
        column="fraction_with_repeat_within_50bp",
        ylabel="Positions within 50 bp of a repeat",
        title="Proximity to annotated repeats",
        output_name="trajectory_repeat_proximity",
        percentage=True,
    )


if __name__ == "__main__":
    main()
