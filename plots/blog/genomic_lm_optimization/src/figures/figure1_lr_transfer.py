"""Figure 1: transfer-validation loss vs learning rate (single panel)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from figures.data import save
from utils.figure_style import (
    attach_stacked_legends_below,
    figsize,
    fmt_lr,
)
from utils.sweep_panel import plot_axis

# Compact (N, D, C) summary contrasting the small-scale reference Vizier sweep
# with the larger transfer-validation sweep depicted in Figures 1 & 2.
# Numbers come from docs/outline.md (`### Sweeps` + `#### Transfer validation
# sweep`); validation N and C are shown as ranges across the three scales.
_REFERENCE_SUBTITLE = "Reference: N = 25M · D = 2.5B · C = 4 × 10¹⁷"
_TARGET_SUBTITLE = "Targets: N = 255M–1B · D = 10B · C = 1.6–6.8 × 10¹⁹"


def build(df: pd.DataFrame, palette: dict, params: list[int]) -> None:
    # A compact near-square canvas matches this single panel's information
    # density. The article declares its exact display width, so typography stays
    # consistent with wider multi-panel figures.
    fig, ax = plt.subplots(figsize=figsize(7.2, 7.2))
    plot_axis(
        ax,
        df,
        axis_role="learning_rate",
        axis_field="learning_rate",
        # Plain text (no Greek symbol) so the whole label renders in the page
        # font, matching the y-axis/legend labels — any mathtext/symbol segment
        # forces the label into DejaVu, which renders at a different size.
        axis_label="Learning rate",
        log_scale=True,
        value_formatter=fmt_lr,
        palette=palette,
    )
    ax.set_box_aspect(1)
    # A little extra gap so the x-axis label clears the rotated LR tick labels.
    ax.xaxis.labelpad = 6
    fig.suptitle(
        "Transfer validation — Loss vs. learning rate",
        y=0.975,
    )
    fig.text(
        0.5,
        0.915,
        _REFERENCE_SUBTITLE,
        ha="center",
        va="top",
    )
    fig.text(
        0.5,
        0.875,
        _TARGET_SUBTITLE,
        ha="center",
        va="top",
    )

    # Stack the two legend families. At this figure's compact article width,
    # placing them side-by-side makes labels collide even though the plot itself
    # only needs one panel.
    attach_stacked_legends_below(
        fig,
        palette,
        params,
        include_reference=True,
        params_y=0.125,
        run_y=0.035,
    )
    fig.tight_layout(rect=(0, 0.21, 1, 0.88))
    save(fig, "figure1_lr_transfer")
