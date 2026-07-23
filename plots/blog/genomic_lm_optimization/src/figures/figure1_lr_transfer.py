"""Figure 1: transfer-validation loss vs learning rate (single panel)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from figures.data import save
from utils.figure_style import FIGURE_WIDTH, attach_legends_below, figsize, fmt_lr
from utils.sweep_panel import plot_axis

# Compact (N, D, C) summary contrasting the small-scale reference Vizier sweep
# with the larger transfer-validation sweep depicted in Figures 1 & 2.
# Numbers come from docs/outline.md (`### Sweeps` + `#### Transfer validation
# sweep`); validation N and C are shown as ranges across the three scales.
_TRANSFER_SUBTITLE = (
    "N = 25M, D = 2.5B, C = 4 × 10¹⁷ → N = 255M–1B, D = 10B, C = 1.6–6.8 × 10¹⁹"
)


def build(df: pd.DataFrame, palette: dict, params: list[int]) -> None:
    # Full FIGURE_WIDTH like the other figures so its labels render at the same
    # on-page size (a narrower figure gets scaled up more at the column width,
    # making its text look larger than the multi-panel figures').
    fig, ax = plt.subplots(figsize=figsize(FIGURE_WIDTH, 6.0))
    plot_axis(
        ax,
        df,
        axis_role="learning_rate",
        axis_field="learning_rate",
        # Plain text (no Greek symbol) so the whole label renders in the page
        # font, matching the y-axis/legend labels — any mathtext/symbol segment
        # forces the label into DejaVu, which renders at a different size.
        axis_label="learning rate",
        log_scale=True,
        value_formatter=fmt_lr,
        palette=palette,
    )
    # A little extra gap so the x-axis label clears the rotated LR tick labels.
    ax.xaxis.labelpad = 6
    ax.set_title(
        "Transfer validation — loss vs learning rate\n" + _TRANSFER_SUBTITLE,
        fontsize=11,
    )
    # Reserve room for the legends below.
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    attach_legends_below(fig, palette, params)
    save(fig, "figure1_lr_transfer")
