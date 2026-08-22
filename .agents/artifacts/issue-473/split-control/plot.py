"""Render the issue #473 full-window training-split control figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ARM_ORDER = ("Chr18-held-out", "Random-held-out")
ARM_COLORS = {
    "Chr18-held-out": "#4C72B0",
    "Random-held-out": "#DD8452",
}


def configure_style() -> None:
    """Use a compact Seaborn-derived Matplotlib presentation style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "svg.fonttype": "none",
        }
    )


def plot_validation_loss(frame: pd.DataFrame) -> None:
    """Plot each validation set in its own panel with an independent y-axis."""
    titles = {
        "Chr18-held-out": "Chr18 held out (offline replay)",
        "Random-held-out": "Random held out (W&B)",
    }
    ylabels = {
        "Chr18-held-out": "Offline NLL",
        "Random-held-out": "W&B validation loss",
    }
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(8.4, 4.3),
        sharex=True,
        sharey=False,
        layout="constrained",
    )
    for axis, arm in zip(axes, ARM_ORDER, strict=True):
        selected = frame.loc[frame["arm"] == arm].sort_values("step")
        assert selected["step"].is_unique
        axis.plot(
            selected["step"],
            selected["loss"],
            color=ARM_COLORS[arm],
            marker="o",
            linewidth=2,
            markersize=4,
        )
        axis.set_title(titles[arm])
        axis.set_xlabel("Training step")
        axis.set_ylabel(ylabels[arm])
        axis.set_box_aspect(1)
        axis.grid(axis="x", visible=False)
        low = float(selected["loss"].min())
        high = float(selected["loss"].max())
        margin = max((high - low) * 0.10, 0.004)
        axis.set_ylim(low - margin, high + margin)
        axis.set_xlim(250, 5_250)
    figure.suptitle("Validation loss by training split")
    figure.savefig(ROOT / "validation_loss_by_split.svg", bbox_inches="tight")
    plt.close(figure)


def plot_auprc(frame: pd.DataFrame) -> None:
    """Plot terminal AUPRC with per-arm SE and prevalence-based axes."""
    expected_panels = list(range(6))
    assert sorted(frame["panel_order"].unique().tolist()) == expected_panels
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(9.4, 6.7),
        sharex=False,
        sharey=False,
        layout="constrained",
    )
    for axis, panel_order in zip(axes.flat, expected_panels, strict=True):
        selected = (
            frame.loc[frame["panel_order"] == panel_order]
            .set_index("arm")
            .loc[list(ARM_ORDER)]
            .reset_index()
        )
        assert selected["baseline"].nunique() == 1
        baseline = float(selected["baseline"].iloc[0])
        values = selected["value"].to_numpy(dtype=float)
        errors = selected["se"].to_numpy(dtype=float)
        positions = np.arange(len(ARM_ORDER), dtype=float)
        axis.bar(
            positions,
            values - baseline,
            bottom=baseline,
            width=0.68,
            color=[ARM_COLORS[arm] for arm in ARM_ORDER],
            yerr=errors,
            error_kw={"ecolor": "#333333", "elinewidth": 1.2, "capsize": 0},
        )
        upper = float(np.max(values + errors))
        span = upper - baseline
        y_max = min(1.0, upper + max(0.02, span * 0.14))
        axis.set_ylim(baseline, y_max)
        axis.set_yticks(np.linspace(baseline, y_max, 4))
        axis.set_yticklabels([f"{tick:.2f}" for tick in axis.get_yticks()])
        axis.set_xticks(positions, ("Chr18", "Random"))
        axis.set_title(
            f"{selected['benchmark'].iloc[0]} · {selected['subset'].iloc[0]}"
        )
        axis.set_box_aspect(1)
        axis.grid(axis="x", visible=False)
        axis.grid(axis="y", alpha=0.35)
    for axis in axes[:, 0]:
        axis.set_ylabel("AUPRC")
    figure.suptitle("Development VEP at step 4,999")
    figure.savefig(ROOT / "auprc_by_split.svg", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Load the committed data and render both SVGs."""
    configure_style()
    plot_validation_loss(pd.read_csv(ROOT / "validation_loss.csv"))
    plot_auprc(pd.read_csv(ROOT / "auprc.csv"))


if __name__ == "__main__":
    main()
