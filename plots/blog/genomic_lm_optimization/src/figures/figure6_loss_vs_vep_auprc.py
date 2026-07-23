"""Figure 6: parameter scaling — final loss vs VEP AUPRC by variant type (2x3)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figures.data import VEP_PANELS, save
from utils.figure_style import (
    FIGURE_WIDTH,
    X_LABEL_PAD,
    attach_params_legend_below,
    figsize,
)

_MARKER_AREA = 110.0


def build(results: pd.DataFrame, palette: dict) -> None:
    """2x3 scatter: final eval/loss (x) vs lm_eval AUPRC (y) for six VEP subsets.

    Marker color encodes params; size is uniform.
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize(FIGURE_WIDTH, 7.0))
    for ax, (subset, title, n) in zip(axes.flat, VEP_PANELS, strict=True):
        col = f"lm_eval/traitgym_mendelian_v2_255/{subset}/auprc"
        valid = results.dropna(subset=["eval_loss", col]).sort_values("params")
        for _, row in valid.iterrows():
            p = int(row["params"])
            ax.scatter(
                row["eval_loss"],
                row[col],
                s=_MARKER_AREA,
                color=palette[p],
                edgecolors="k",
                linewidths=0.5,
                zorder=3,
            )
        # Dotted linear best-fit + subtle Pearson ρ annotation.
        if len(valid) >= 2:
            xs = valid["eval_loss"].to_numpy(dtype=float)
            ys = valid[col].to_numpy(dtype=float)
            slope, intercept = np.polyfit(xs, ys, 1)
            x_line = np.array([xs.min(), xs.max()])
            ax.plot(
                x_line,
                slope * x_line + intercept,
                linestyle=":",
                color="0.35",
                linewidth=1.2,
                zorder=2,
            )
            rho = float(np.corrcoef(xs, ys)[0, 1])
            ax.text(
                0.97,
                0.95,
                rf"$\rho={rho:.2f}$",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="0.35",
            )
        ax.set_title(f"{title} (n={n:,})", fontsize=10)
        ax.grid(False)
        # A touch more vertical headroom so the large markers at the top/bottom
        # of each panel aren't clipped by the axes.
        ax.margins(y=0.12)

    # Shared axis labels: "loss" only on the middle column of the bottom row;
    # "AUPRC" on the leftmost column only.
    axes[-1, 1].set_xlabel("loss", labelpad=X_LABEL_PAD)
    for ax in axes[:, 0]:
        ax.set_ylabel("AUPRC")

    fig.suptitle(
        "Parameter scaling — loss vs VEP AUPRC by variant type", fontsize=11, y=0.96
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.98))

    params_present = sorted({int(p) for p in results["params"].dropna().unique()})
    attach_params_legend_below(
        fig, palette, params_present, width_scale=0.4, handlelength=1.0
    )
    save(fig, "figure6_loss_vs_vep_auprc")
