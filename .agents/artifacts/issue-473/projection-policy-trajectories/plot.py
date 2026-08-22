"""Render full-window versus center-1 AUPRC trajectories."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

ROOT = Path(__file__).resolve().parent
ARM_ORDER = ("Full window", "Center 1 bp")
ARM_COLORS = {
    "Full window": "#0072B2",
    "Center 1 bp": "#D55E00",
}
X_TICKS = (1000, 2000, 3000, 4000, 4999)
X_TICK_LABELS = ("1,000", "2,000", "3,000", "4,000", "4,999")


def configure_style() -> None:
    """Apply a compact, colorblind-safe plotting style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.dpi": 150,
            "font.size": 9,
            "savefig.dpi": 150,
            "svg.fonttype": "none",
        }
    )


def plot_region(
    frame: pd.DataFrame,
    *,
    output_name: str,
    shape: tuple[int, int],
    figsize: tuple[float, float],
) -> None:
    """Plot one region's complete AUPRC trajectories with one-SE bars."""
    panel_order = sorted(frame["panel_order"].unique().tolist())
    assert panel_order == list(range(len(panel_order)))
    assert len(panel_order) == shape[0] * shape[1]
    figure, axes = plt.subplots(
        *shape,
        figsize=figsize,
        sharex=True,
        sharey=False,
        squeeze=False,
        layout="constrained",
    )
    for axis, order in zip(axes.flat, panel_order, strict=True):
        panel = frame.loc[frame["panel_order"] == order].copy()
        assert panel["baseline"].nunique() == 1
        assert set(panel["arm"]) == set(ARM_ORDER)
        assert panel.groupby("arm")["step"].nunique().eq(9).all()
        baseline = float(panel["baseline"].iloc[0])
        for arm in ARM_ORDER:
            selected = panel.loc[panel["arm"] == arm].sort_values("step")
            axis.errorbar(
                selected["step"],
                selected["value"],
                yerr=selected["se"],
                color=ARM_COLORS[arm],
                label=arm,
                marker="o",
                markersize=3.5,
                linewidth=1.6,
                elinewidth=0.85,
                capsize=0,
                alpha=0.95,
            )
        lower_with_se = float((panel["value"] - panel["se"]).min())
        upper_with_se = float((panel["value"] + panel["se"]).max())
        data_span = max(upper_with_se - min(baseline, lower_with_se), 0.02)
        lower = baseline
        if lower_with_se < baseline:
            lower = lower_with_se - 0.025 * data_span
        upper = min(1.0, upper_with_se + 0.08 * data_span)
        axis.set_ylim(lower, upper)
        axis.axhline(
            baseline,
            color="#777777",
            linestyle=(0, (3, 2)),
            linewidth=0.9,
            zorder=0,
        )
        axis.set_title(
            f"{panel['benchmark'].iloc[0]} · {panel['subset'].iloc[0]}",
            fontsize=10,
        )
        axis.set_box_aspect(1)
        axis.set_xticks(X_TICKS, X_TICK_LABELS)
        axis.tick_params(axis="x", labelsize=7.5)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        axis.grid(axis="x", visible=False)
        axis.grid(axis="y", alpha=0.3)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.supxlabel("Training step")
    figure.supylabel("AUPRC")
    figure.legend(
        handles,
        labels,
        title="Projection policy",
        loc="outside upper center",
        ncol=len(ARM_ORDER),
        frameon=False,
    )
    figure.savefig(ROOT / output_name, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Load committed values and render both SVGs."""
    configure_style()
    plot_region(
        pd.read_csv(ROOT / "cds_auprc.csv"),
        output_name="cds_auprc_by_projection.svg",
        shape=(2, 3),
        figsize=(9.4, 7.0),
    )
    plot_region(
        pd.read_csv(ROOT / "enhancer_auprc.csv"),
        output_name="enhancer_auprc_by_projection.svg",
        shape=(1, 2),
        figsize=(6.8, 4.1),
    )


if __name__ == "__main__":
    main()
