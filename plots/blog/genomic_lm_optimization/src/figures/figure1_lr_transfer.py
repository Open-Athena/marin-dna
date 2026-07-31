"""Figure 1: transfer-validation loss vs learning rate (single panel)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from figures.data import save
from utils.figure_style import (
    attach_stacked_legends_right,
    figsize,
    set_square_subplot_height,
)
from utils.sweep_panel import plot_axis

SUBPLOT_HEIGHT_PX = 148.0


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
        palette=palette,
        native_numeric_axis=True,
    )
    ax.set_box_aspect(1)
    # A little extra gap so the x-axis label clears the rotated LR tick labels.
    ax.xaxis.labelpad = 6
    fig.tight_layout(rect=(0, 0.21, 1, 1))
    set_square_subplot_height(fig, [ax], SUBPLOT_HEIGHT_PX)
    # A vertical legend block uses the plot's right side efficiently and avoids
    # a separate column-spacing decision for this compact square panel.
    attach_stacked_legends_right(
        fig,
        ax,
        palette,
        params,
        include_reference=True,
    )
    save(fig, "figure1_lr_transfer")
