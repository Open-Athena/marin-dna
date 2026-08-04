"""Figure 3: per-region transfer loss vs each tuned hyper (3x3 panels)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from figures.data import save
from utils.figure_style import (
    FIGURE_WIDTH,
    attach_legends_below,
    figsize,
    pack_horizontal_axis_columns,
    set_square_subplot_height,
)
from utils.sweep_panel import plot_axis

SUBPLOT_HEIGHT_PX = 148.0

# Rows of figure 3: (region key, label used in axis text / row title).
_REGION_ROWS: tuple[tuple[str, str], ...] = (
    ("cds", "CDS"),
    ("upstream", "Upstream"),
    ("downstream", "Downstream"),
)
# Cols of figure 3: (axis role, axis field, axis label, log scale).
_HYPER_COLS: tuple[tuple[str, str, str, bool], ...] = (
    ("learning_rate", "learning_rate", "Learning rate (η)", True),
    ("beta2", "beta2", "β₂", False),
    ("epsilon", "epsilon", "ε", True),
)


def build(df: pd.DataFrame, palette: dict, params: list[int]) -> None:
    """3x3 grid: per-region transfer loss vs each tuned hyper.

    Rows are genomic regions (CDS / upstream / downstream); columns are the
    swept hypers (learning rate / β₂ / ε). y is `eval/val_<region>/loss` from
    the corresponding region column in the transfer CSV. Free y-axis per cell.
    Negative controls are omitted (this figure focuses on transfer-vs-direct
    sweep shapes per region).
    """
    fig, axes = plt.subplots(3, 3, figsize=figsize(FIGURE_WIDTH, 8.8))
    for r, (region_key, region_label) in enumerate(_REGION_ROWS):
        y_field = f"eval_loss_{region_key}"
        for c, (axis_role, axis_field, axis_label, log_scale) in enumerate(_HYPER_COLS):
            ax = axes[r, c]
            plot_axis(
                ax,
                df,
                axis_role=axis_role,
                axis_field=axis_field,
                axis_label=axis_label if r == 2 else "",
                log_scale=log_scale,
                palette=palette,
                native_numeric_axis=True,
                include_negative_control=False,
                y_field=y_field,
                y_label=f"{region_label} loss" if c == 0 else "",
            )
            if r != 2:
                ax.tick_params(axis="x", labelbottom=False)
                ax.set_xlabel("")
            ax.set_box_aspect(1)
    fig.tight_layout(rect=(0, 0.15, 1, 1), h_pad=0.5, w_pad=0)
    set_square_subplot_height(fig, axes.flat, SUBPLOT_HEIGHT_PX)
    pack_horizontal_axis_columns(fig, axes)
    attach_legends_below(
        fig,
        palette,
        params,
        include_reference=False,
        legend_y=0.13,
    )
    save(fig, "figure3_region_hyper_transfer")
