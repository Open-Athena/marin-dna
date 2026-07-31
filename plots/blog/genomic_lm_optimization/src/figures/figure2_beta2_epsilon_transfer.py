"""Figure 2: transfer-validation loss vs beta2 / epsilon (1x2 panels)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from figures.data import save
from utils.figure_style import (
    FIGURE_WIDTH,
    attach_legends_below,
    figsize,
    pack_horizontal_axes,
    set_square_subplot_height,
)
from utils.sweep_panel import plot_axis

SUBPLOT_HEIGHT_PX = 148.0


def build(df: pd.DataFrame, palette: dict, params: list[int]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=figsize(FIGURE_WIDTH, 5.8))
    plot_axis(
        axes[0],
        df,
        axis_role="beta2",
        axis_field="beta2",
        axis_label="β₂",
        log_scale=False,
        palette=palette,
        native_numeric_axis=True,
        include_negative_control=False,
    )
    plot_axis(
        axes[1],
        df,
        axis_role="epsilon",
        axis_field="epsilon",
        axis_label="ε",
        log_scale=True,
        palette=palette,
        native_numeric_axis=True,
        include_negative_control=False,
    )
    for ax in axes:
        ax.set_box_aspect(1)
    fig.tight_layout(rect=(0, 0.31, 1, 1), w_pad=0)
    set_square_subplot_height(fig, axes, SUBPLOT_HEIGHT_PX)
    pack_horizontal_axes(fig, axes)
    attach_legends_below(
        fig,
        palette,
        params,
        include_reference=False,
        legend_y=0.35,
    )
    save(fig, "figure2_beta2_epsilon_transfer")
