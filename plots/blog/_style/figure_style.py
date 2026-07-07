"""Shared presentation helpers for the figure set.

Figure dimensions, the earthy param palette, value formatters, the
below-axes legend strips, and tick formatting — everything that controls how
the figures *look* but not what data they show.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from matplotlib.transforms import Bbox
import matplotlib.pyplot as plt

# Natural (unscaled) dimensions. Width is constant across figures so they line
# up in any side-by-side rendering.
FIGURE_WIDTH = 12.0
FIGURE_HEIGHT = 5.0

# Figures are naturally wide but display at the ~720px blog column, which shrinks
# their point-sized text to roughly half the body copy. Author every figure at
# this fraction of its natural size — via figsize() below — so labels read at
# ~body size on the page, with layout (tight_layout, legends) computed at the
# final size. One knob; 1.0 = natural size, smaller = larger on-page text.
SCALE = 0.74


def figsize(w: float, h: float) -> tuple[float, float]:
    """Scale a natural (width, height) in inches by SCALE for on-page sizing.

    Every figure builds with ``figsize=figsize(...)`` so the figure is authored
    at its final size (no post-hoc resize), keeping layout WYSIWYG.
    """
    return (w * SCALE, h * SCALE)

# Warm, earthy palette tuned to the page theme (tan/brown). Replaces viridis,
# whose purples and greens clash with the warm background. A muted teal -> rust
# ramp: reads as earthy, holds good contrast on the #ece3d5 figure panels, and
# stays distinguishable across up to 8 ordered model-size classes.
PARAM_CMAP = LinearSegmentedColormap.from_list(
    "earth", ["#23403f", "#3f6b5e", "#7e8a45", "#b3823f", "#9c4f2f"]
)

# Color the param palette assigns to the largest model (the 4B scale) — the rust
# end of the ramp. palette() always maps the top scale to cmap(0.85)
# (0.15 + 0.7·(n-1)/(n-1)) regardless of model count, so this is the exact 4B
# color seen in the scaling figures. Reused by single-series figures (8, 9) that
# want to read as "the big model" without rebuilding the full palette.
LARGEST_MODEL_COLOR = PARAM_CMAP(0.85)

# Warm single-hue sequential (cream -> espresso) for magnitude heatmaps — same
# family as PARAM_CMAP so the figure set stays visually consistent.
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "earth_seq", ["#f4ecda", "#e2c089", "#c79150", "#9c6234", "#5e3418"]
)

# One strong earthy accent (terracotta) for single-series figures, drawn from
# the warm end of PARAM_CMAP.
SERIES_COLOR = "#9c4f2f"

# Earthy *qualitative* palette for categorical series (variant/trait types in
# figures 5, A1, A2) — the warm counterpart to tab10. Six distinct, muted hues
# (rust, teal, ochre, slate, olive, plum) that stay legible on the #ece3d5
# panels and harmonize with PARAM_CMAP. Order is fixed so a given category keeps
# its color across figures.
EARTH_QUAL = ["#9c4f2f", "#2f6f63", "#c0883a", "#465c6e", "#6f7d3f", "#8a5170"]

# Diverging map for signed quantities (e.g. correlation ρ): teal <-> brown, the
# two poles of the earthy palette, with a pale center at zero.
DIVERGING_CMAP = "BrBG_r"

# Top of the legend boxes in figure coordinates. Tuned so the legends sit just
# below the x-axis tick labels, not at the bottom of the figure.
LEGEND_Y = 0.10

# Padding between x tick labels and the x-axis label (matplotlib default is 4).
X_LABEL_PAD = 0

# Tight label↔marker spacing, generous between (marker, label) pairs.
LEGEND_KW = dict(
    fontsize=9,
    title_fontsize=9,
    frameon=False,
    handletextpad=0.3,
    columnspacing=2.2,
    borderpad=0.4,
)


def palette(param_counts: list[int]) -> dict[int, tuple]:
    cmap = PARAM_CMAP
    if len(param_counts) == 1:
        return {param_counts[0]: cmap(0.5)}
    return {p: cmap(0.15 + 0.7 * i / (len(param_counts) - 1)) for i, p in enumerate(param_counts)}


def params_label(num_params: int | float) -> str:
    n = int(num_params)
    if n >= 1_000_000_000:
        return f"{round(n / 1e9)}B"
    return f"{round(n / 1e6)}M"


def fmt_lr(lr: float) -> str:
    exp = int(np.floor(np.log10(lr)))
    mantissa = lr / (10**exp)
    return rf"${mantissa:.1f}\times 10^{{{exp}}}$"


def fmt_beta2(b: float) -> str:
    return f"{b:.4f}"


def fmt_epsilon(e: float) -> str:
    if e <= 0:
        return f"{e:g}"
    exp = int(np.floor(np.log10(e)))
    mantissa = e / (10**exp)
    return rf"${mantissa:.1f}\times 10^{{{exp}}}$"


def shape_legend_handles(include_reference: bool = True):
    """Proxy artists for the marker-shape legend (no axes side-effects)."""
    common = dict(color="w", markerfacecolor="lightgray", markeredgecolor="k", markeredgewidth=0.6, linestyle="")
    sweep = Line2D([0], [0], marker="o", markersize=8, **common)
    optimal = Line2D([0], [0], marker="s", markersize=8, **common)
    handles = [sweep, optimal]
    labels = ["sweep", "optimal (predicted)"]
    if include_reference:
        handles.append(Line2D([0], [0], marker="D", markersize=8, **common))
        labels.append("control (reference)")
    return handles, labels


def params_legend_handles(palette: dict, params: list[int]):
    """Proxy artists for the per-scale params legend (square markers, scale color)."""
    sorted_params = sorted(params)
    handles = [
        Line2D(
            [0], [0], marker="s", color="w", markerfacecolor=palette[p],
            markeredgecolor="k", markeredgewidth=0.6, markersize=8, linestyle="",
        )
        for p in sorted_params
    ]
    labels = [params_label(p) for p in sorted_params]
    return handles, labels


def attach_params_legend_below(
    fig, palette: dict, params: list[int], *, width_scale: float = 1.0,
    handlelength: float | None = None,
) -> None:
    """Single horizontal `model params` legend, centered just below the x-axis.

    `width_scale` shrinks/expands the inter-pair gap (columnspacing) — smaller
    values produce a more compact legend. `handlelength` (if given) tightens the
    marker-to-label gap for an even more compact strip.
    """
    p_handles, p_labels = params_legend_handles(palette, params)
    kw = {**LEGEND_KW, "columnspacing": LEGEND_KW["columnspacing"] * width_scale}
    if handlelength is not None:
        kw["handlelength"] = handlelength
    fig.legend(
        p_handles, p_labels,
        ncol=len(p_handles), title="model params",
        loc="upper center", bbox_to_anchor=(0.5, LEGEND_Y),
        **kw,
    )


# Tighter spacing for the two-legend strip (figs 1-3): entries sit close
# together (small columnspacing) and the marker handle hugs its label
# (short handlelength), so a legend doesn't sprawl across the figure.
TWO_LEGEND_KW = {**LEGEND_KW, "columnspacing": 0.6, "handlelength": 0.9}


def attach_legends_below(
    fig, palette: dict, params: list[int],
    include_reference: bool = True,
    legend_y: float = 0.09,
    gap: float = 0.03,
) -> None:
    """Two horizontal figure-level legends (`model params` + `run type`) placed
    as a single block, centered on x=0.5, just below the axes.

    The pair is centered by measuring each legend's rendered width and seating
    them symmetrically with a fixed `gap` — so the block stays centered no matter
    how many entries each legend has (e.g. with or without the control marker).
    Centering is on the *plotted content* (the axes block incl. their labels),
    not the raw figure: figures save with ``bbox_inches='tight'``, which crops to
    that content, so its center — not x=0.5 — is what reads as centered.
    """
    p_handles, p_labels = params_legend_handles(palette, params)
    s_handles, s_labels = shape_legend_handles(include_reference=include_reference)

    # Seat both at the left provisionally; positions are fixed up after measuring.
    leg_params = fig.legend(
        p_handles, p_labels, ncol=len(p_handles), title="model params",
        loc="upper left", bbox_to_anchor=(0.0, legend_y), **TWO_LEGEND_KW,
    )
    fig.add_artist(leg_params)
    leg_shape = fig.legend(
        s_handles, s_labels, ncol=len(s_handles), title="run type",
        loc="upper left", bbox_to_anchor=(0.0, legend_y), **TWO_LEGEND_KW,
    )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fw = fig.bbox.width
    # Visual center of the plotted content (axes + their tick/axis labels).
    content = Bbox.union([a.get_tightbbox(renderer) for a in fig.axes if a.get_visible()])
    center = (content.x0 + content.x1) / 2.0 / fw
    w1 = leg_params.get_window_extent().width / fw
    w2 = leg_shape.get_window_extent().width / fw
    x0 = center - (w1 + gap + w2) / 2.0
    leg_params.set_bbox_to_anchor((x0, legend_y), transform=fig.transFigure)
    leg_shape.set_bbox_to_anchor((x0 + w1 + gap, legend_y), transform=fig.transFigure)


def set_plain_decimal_yticks(ax) -> None:
    """Force y-axis tick labels to plain decimals (no scientific notation).

    Works for both linear and log y-scales. On log scales matplotlib's default
    LogFormatter renders values like ``6×10⁻¹``; in this figure's loss range
    (~0.5–1.5) plain decimals are more readable.
    """
    fmt = ScalarFormatter()
    fmt.set_scientific(False)
    fmt.set_useOffset(False)
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.set_minor_formatter(fmt)
