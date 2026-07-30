"""Figure 3: per-region transfer loss vs each tuned hyper (3x3 panels)."""

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
    fmt_lr,
)
from utils.sweep_panel import plot_axis

# Rows of figure 3: (region key, label used in axis text / row title).
_REGION_ROWS: tuple[tuple[str, str], ...] = (
    ("cds", "CDS"),
    ("upstream", "Upstream"),
    ("downstream", "Downstream"),
)
# Cols of figure 3: (axis role, axis field, axis label, log scale, formatter).
_HYPER_COLS: tuple[tuple[str, str, str, bool, "callable"], ...] = (
    ("learning_rate", "learning_rate", "Learning rate (η)", True, fmt_lr),
    ("beta2", "beta2", "β₂", False, fmt_beta2),
    ("epsilon", "epsilon", "ε", True, fmt_epsilon),
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
        for c, (axis_role, axis_field, axis_label, log_scale, fmt) in enumerate(
            _HYPER_COLS
        ):
            ax = axes[r, c]
            plot_axis(
                ax,
                df,
                axis_role=axis_role,
                axis_field=axis_field,
                axis_label=axis_label if r == 2 else "",
                log_scale=log_scale,
                value_formatter=fmt,
                palette=palette,
                include_negative_control=False,
                y_field=y_field,
                y_label=f"{region_label} Loss" if c == 0 else "",
                tick_stride=2 if c > 0 else 1,
            )
            if r != 2:
                ax.set_xticklabels([])
                ax.set_xlabel("")
    fig.suptitle(
        "Transfer validation — Per-region Loss vs. learning rate, β₂, and ε",
        y=0.985,
    )
    # Explicit margins (no tight_layout) so the title and legend hug the plot grid tightly.
    fig.subplots_adjust(
        top=0.9525, bottom=0.32, left=0.055, right=0.99, hspace=0.12, wspace=0.18
    )
    attach_stacked_legends_below(
        fig,
        palette,
        params,
        include_reference=False,
        params_y=0.12,
        run_y=0.005,
    )
    save(fig, "figure3_region_hyper_transfer")
