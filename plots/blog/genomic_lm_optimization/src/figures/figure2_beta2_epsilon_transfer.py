"""Figure 2: transfer-validation loss vs beta2 / epsilon (1x2 panels)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from figures.data import save
from utils.figure_style import (
    FIGURE_WIDTH,
    attach_stacked_legends_below,
    figsize,
    fmt_beta2,
    fmt_epsilon,
)
from utils.sweep_panel import plot_axis


def build(df: pd.DataFrame, palette: dict, params: list[int]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=figsize(FIGURE_WIDTH, 5.8))
    plot_axis(
        axes[0],
        df,
        axis_role="beta2",
        axis_field="beta2",
        axis_label="β₂",
        log_scale=False,
        value_formatter=fmt_beta2,
        palette=palette,
        include_negative_control=False,
    )
    # Push the β₂ label down slightly relative to the global label pad.
    axes[0].xaxis.labelpad = 4
    plot_axis(
        axes[1],
        df,
        axis_role="epsilon",
        axis_field="epsilon",
        axis_label="ε",
        log_scale=True,
        value_formatter=fmt_epsilon,
        palette=palette,
        include_negative_control=False,
    )
    for ax in axes:
        ax.set_box_aspect(1)
    # Pull the ε label up closer to the tick labels.
    axes[1].xaxis.labelpad = -4
    fig.tight_layout(rect=(0, 0.31, 1, 1))
    attach_stacked_legends_below(
        fig,
        palette,
        params,
        include_reference=False,
        params_y=0.17,
        run_y=0.01,
    )
    save(fig, "figure2_beta2_epsilon_transfer")
