"""Plot pooled conservation prevalence for issue #489 trajectory groups."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
import seaborn as sns


ARTIFACT_ROOT = Path(__file__).resolve().parent
INPUT_PATH = ARTIFACT_ROOT / "biology" / "region_group_statistics.csv"
FIGURE_PATH = ARTIFACT_ROOT / "figures" / "global_trajectory_conservation.svg"
OUTPUT_PATH = ARTIFACT_ROOT / "global_trajectory_conservation.csv"
GROUP_ORDER = ("high_to_high", "low_to_high", "high_to_low", "low_to_low")
GROUP_LABELS = {
    "high_to_high": "H\N{RIGHTWARDS ARROW}H",
    "low_to_high": "L\N{RIGHTWARDS ARROW}H",
    "high_to_low": "H\N{RIGHTWARDS ARROW}L",
    "low_to_low": "L\N{RIGHTWARDS ARROW}L",
}
GROUP_COLORS = {
    "high_to_high": "#ff7f00",
    "low_to_high": "#e41a1c",
    "high_to_low": "#377eb8",
    "low_to_low": "#4daf4a",
}


def pooled_prevalence(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate exact conserved-position counts across validation regions."""
    frame = frame.copy()
    frame["n_conserved"] = (
        frame["n_positions"] * frame["conservation_prevalence"]
    ).round().astype("int64")
    pooled = (
        frame.groupby("group", as_index=False, observed=True)[
            ["n_positions", "n_conserved"]
        ]
        .sum()
        .set_index("group")
        .loc[list(GROUP_ORDER)]
        .reset_index()
    )
    pooled["conservation_prevalence"] = (
        pooled["n_conserved"] / pooled["n_positions"]
    )
    assert pooled["n_positions"].sum() == 14_002_032
    return pooled


def plot(frame: pd.DataFrame) -> None:
    """Render exact pooled proportions with a zero-based bar axis."""
    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(4, 4))
    bars = axis.bar(
        [GROUP_LABELS[group] for group in frame["group"]],
        frame["conservation_prevalence"],
        color=[GROUP_COLORS[group] for group in frame["group"]],
        width=0.68,
    )
    axis.bar_label(
        bars,
        labels=[f"{value:.1%}" for value in frame["conservation_prevalence"]],
        padding=4,
    )
    axis.set_xlabel("Trajectory group")
    axis.set_ylabel("Conserved positions")
    axis.set_title("Conservation by global trajectory type")
    axis.set_ylim(0, 0.60)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(axis="x", visible=False)
    axis.set_box_aspect(1)
    figure.savefig(FIGURE_PATH, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Write the exact pooled table and reviewed SVG."""
    frame = pd.read_csv(INPUT_PATH)
    pooled = pooled_prevalence(frame)
    pooled.to_csv(OUTPUT_PATH, index=False)
    plot(pooled)
    print(pooled.to_string(index=False))


if __name__ == "__main__":
    main()
