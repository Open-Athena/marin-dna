"""Shared presentation helpers for the figure set.

Figure dimensions, the earthy param palette, value formatters, the
below-axes legend strips, and tick formatting — everything that controls how
the figures *look* but not what data they show.
"""

from __future__ import annotations

import numpy as np
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from matplotlib.transforms import Bbox

from marin_dna.blog_figure_typography import FIGURE_GLOBAL_RENDER_SCALE

# Natural (unscaled) dimensions. Width is constant across figures so they line
# up in any side-by-side rendering.
FIGURE_WIDTH = 12.0
FIGURE_HEIGHT = 5.0

# Initial canvas geometry for the historical recipes. This is not a style or
# browser scale: every data plot receives the same final whole-SVG scale from
# ``FIGURE_GLOBAL_RENDER_SCALE`` when it is saved.
LAYOUT_SCALE = 0.74


def figsize(w: float, h: float) -> tuple[float, float]:
    """Return the historical recipe's initial canvas geometry in inches.

    Plot styling is unaffected; fonts, lines, and markers inherit Matplotlib
    defaults and the saved SVG receives the one shared whole-figure scale.
    """
    return (w * LAYOUT_SCALE, h * LAYOUT_SCALE)


def set_square_subplot_height(fig, axes, displayed_height_px: float) -> None:
    """Resize a canvas so square data axes render at ``displayed_height_px``.

    Width stays tied to height. This changes only plot geometry: text and data
    glyphs retain Matplotlib's shared defaults and are scaled later by the one
    global whole-SVG factor.
    """
    assert displayed_height_px > 0
    axes = tuple(axes)
    assert axes
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    extents = [axis.get_window_extent(renderer=renderer) for axis in axes]
    for extent in extents:
        assert np.isclose(extent.width, extent.height, rtol=1e-6), extent
    points_per_display_pixel = 72.0 / fig.dpi
    heights_points = [extent.height * points_per_display_pixel for extent in extents]
    assert max(heights_points) - min(heights_points) < 0.01, heights_points
    current_height_px = heights_points[0] * FIGURE_GLOBAL_RENDER_SCALE
    resize_factor = displayed_height_px / current_height_px

    # Scale the subplot block while preserving its outer gutters in physical
    # points. Fonts and legends therefore retain their standard size instead of
    # being squeezed together when a recipe requests a smaller data area.
    bounds_points = [
        (
            extent.x0 * points_per_display_pixel,
            extent.y0 * points_per_display_pixel,
            extent.width * points_per_display_pixel,
            extent.height * points_per_display_pixel,
        )
        for extent in extents
    ]
    block_x0 = min(x for x, _y, _width, _height in bounds_points)
    block_y0 = min(y for _x, y, _width, _height in bounds_points)
    block_x1 = max(x + width for x, _y, width, _height in bounds_points)
    block_y1 = max(y + height for _x, y, _width, height in bounds_points)
    figure_width_points = fig.get_figwidth() * 72.0
    figure_height_points = fig.get_figheight() * 72.0
    right_gutter = figure_width_points - block_x1
    top_gutter = figure_height_points - block_y1
    new_width_points = block_x0 + (block_x1 - block_x0) * resize_factor + right_gutter
    new_height_points = block_y0 + (block_y1 - block_y0) * resize_factor + top_gutter
    fig.set_size_inches(new_width_points / 72.0, new_height_points / 72.0, forward=True)
    for axis, (x, y, width, height) in zip(axes, bounds_points, strict=True):
        resized_x = block_x0 + (x - block_x0) * resize_factor
        resized_y = block_y0 + (y - block_y0) * resize_factor
        axis.set_position(
            (
                resized_x / new_width_points,
                resized_y / new_height_points,
                width * resize_factor / new_width_points,
                height * resize_factor / new_height_points,
            )
        )
    fig.canvas.draw()
    resized = axes[0].get_window_extent(renderer=fig.canvas.get_renderer())
    resized_height_px = resized.height * 72.0 / fig.dpi * FIGURE_GLOBAL_RENDER_SCALE
    assert np.isclose(resized_height_px, displayed_height_px, atol=0.1), (
        resized_height_px,
        displayed_height_px,
    )


def pack_horizontal_axes(fig, axes, gap_font_sizes: float = 1.0) -> None:
    """Pack axes left-to-right using their rendered labels as boundaries."""
    assert gap_font_sizes >= 0
    axes = tuple(axes)
    assert len(axes) > 1
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    gap_pixels = mpl.rcParams["font.size"] * fig.dpi / 72.0 * gap_font_sizes
    previous_right = axes[0].get_tightbbox(renderer=renderer).x1
    for axis in axes[1:]:
        tight_bbox = axis.get_tightbbox(renderer=renderer)
        shift_pixels = previous_right + gap_pixels - tight_bbox.x0
        position = axis.get_position()
        axis.set_position(
            (
                position.x0 + shift_pixels / fig.bbox.width,
                position.y0,
                position.width,
                position.height,
            )
        )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        previous_right = axis.get_tightbbox(renderer=renderer).x1


def pack_horizontal_axis_columns(fig, axes, gap_font_sizes: float = 1.0) -> None:
    """Pack a grid's columns while preserving alignment between its rows."""
    assert gap_font_sizes >= 0
    rows = tuple(tuple(row) for row in axes)
    assert rows
    column_count = len(rows[0])
    assert column_count > 1
    assert all(len(row) == column_count for row in rows)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    gap_pixels = mpl.rcParams["font.size"] * fig.dpi / 72.0 * gap_font_sizes
    previous_right = max(row[0].get_tightbbox(renderer=renderer).x1 for row in rows)
    for column_index in range(1, column_count):
        column = tuple(row[column_index] for row in rows)
        current_left = min(axis.get_tightbbox(renderer=renderer).x0 for axis in column)
        shift_pixels = previous_right + gap_pixels - current_left
        for axis in column:
            position = axis.get_position()
            axis.set_position(
                (
                    position.x0 + shift_pixels / fig.bbox.width,
                    position.y0,
                    position.width,
                    position.height,
                )
            )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        previous_right = max(
            axis.get_tightbbox(renderer=renderer).x1 for axis in column
        )


def center_axes_block(fig, axes, reference_axes) -> None:
    """Center one axes block on the rendered content of another."""
    axes = tuple(axes)
    reference_axes = tuple(reference_axes)
    assert axes
    assert reference_axes
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    block = Bbox.union([axis.get_tightbbox(renderer=renderer) for axis in axes])
    reference = Bbox.union(
        [axis.get_tightbbox(renderer=renderer) for axis in reference_axes]
    )
    shift_pixels = (reference.x0 + reference.x1) / 2.0 - (block.x0 + block.x1) / 2.0
    for axis in axes:
        position = axis.get_position()
        axis.set_position(
            (
                position.x0 + shift_pixels / fig.bbox.width,
                position.y0,
                position.width,
                position.height,
            )
        )


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

# Shared encodings for figures that compare model families and scoring
# protocols. Protocol changes color and line style; family changes marker only.
SCORING_PROTOCOL_COLORS = {
    "llr": "#465c6e",
    "probe": "#9c4f2f",
}
SCORING_PROTOCOL_LINESTYLES = {
    "llr": "--",
    "probe": "-",
}
MODEL_FAMILY_MARKERS = {
    "marindna": "o",
    "evo2": "s",
}
COMPARISON_ERRORBAR_ALPHA = 1.0

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
    frameon=False,
    handletextpad=0.3,
    columnspacing=2.2,
    borderpad=0.4,
)


def palette(param_counts: list[int]) -> dict[int, tuple]:
    cmap = PARAM_CMAP
    if len(param_counts) == 1:
        return {param_counts[0]: cmap(0.5)}
    return {
        p: cmap(0.15 + 0.7 * i / (len(param_counts) - 1))
        for i, p in enumerate(param_counts)
    }


def params_label(num_params: int | float) -> str:
    n = int(num_params)
    if n >= 1_000_000_000:
        return f"{round(n / 1e9)}B"
    return f"{round(n / 1e6)}M"


def fmt_lr(lr: float) -> str:
    exp = int(np.floor(np.log10(lr)))
    mantissa = lr / (10**exp)
    return f"{mantissa:.1f} × 10{_superscript(exp)}"


def fmt_beta2(b: float) -> str:
    return f"{b:.4f}"


def fmt_epsilon(e: float) -> str:
    if e <= 0:
        return f"{e:g}"
    exp = int(np.floor(np.log10(e)))
    mantissa = e / (10**exp)
    return f"{mantissa:.1f} × 10{_superscript(exp)}"


def _superscript(value: int) -> str:
    """Format an integer exponent without switching to a math-only font."""
    return str(value).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))


def shape_legend_handles(include_reference: bool = True):
    """Proxy artists for the marker-shape legend (no axes side-effects)."""
    common = dict(
        color="w",
        markerfacecolor="lightgray",
        markeredgecolor="k",
        linestyle="",
    )
    sweep = Line2D([0], [0], marker="o", **common)
    optimal = Line2D([0], [0], marker="s", **common)
    handles = [sweep, optimal]
    labels = ["Sweep", "Optimal (predicted)"]
    if include_reference:
        handles.append(Line2D([0], [0], marker="D", **common))
        labels.append("Control (reference)")
    return handles, labels


def params_legend_handles(palette: dict, params: list[int]):
    """Proxy artists for the per-scale params legend (square markers, scale color)."""
    sorted_params = sorted(params)
    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=palette[p],
            markeredgecolor="k",
            linestyle="",
        )
        for p in sorted_params
    ]
    labels = [params_label(p) for p in sorted_params]
    return handles, labels


def attach_params_legend_below(
    fig,
    palette: dict,
    params: list[int],
    *,
    width_scale: float = 1.0,
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
        p_handles,
        p_labels,
        ncol=len(p_handles),
        title="Model parameters",
        loc="upper center",
        bbox_to_anchor=(0.5, LEGEND_Y),
        **kw,
    )


# Tighter spacing for the two-legend strip (figs 1-3): entries sit close
# together (small columnspacing) and the marker handle hugs its label
# (short handlelength), so a legend doesn't sprawl across the figure.
TWO_LEGEND_KW = {**LEGEND_KW, "columnspacing": 0.6, "handlelength": 0.9}


def attach_legends_below(
    fig,
    palette: dict,
    params: list[int],
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
        p_handles,
        p_labels,
        ncol=len(p_handles),
        title="Model parameters",
        loc="upper left",
        bbox_to_anchor=(0.0, legend_y),
        **TWO_LEGEND_KW,
    )
    fig.add_artist(leg_params)
    leg_shape = fig.legend(
        s_handles,
        s_labels,
        ncol=len(s_handles),
        title="Run type",
        loc="upper left",
        bbox_to_anchor=(0.0, legend_y),
        **TWO_LEGEND_KW,
    )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fw = fig.bbox.width
    # Visual center of the plotted content (axes + their tick/axis labels).
    content = Bbox.union(
        [a.get_tightbbox(renderer) for a in fig.axes if a.get_visible()]
    )
    center = (content.x0 + content.x1) / 2.0 / fw
    w1 = leg_params.get_window_extent().width / fw
    w2 = leg_shape.get_window_extent().width / fw
    x0 = center - (w1 + gap + w2) / 2.0
    leg_params.set_bbox_to_anchor((x0, legend_y), transform=fig.transFigure)
    leg_shape.set_bbox_to_anchor((x0 + w1 + gap, legend_y), transform=fig.transFigure)


def attach_stacked_legends_below(
    fig,
    palette: dict,
    params: list[int],
    *,
    include_reference: bool = True,
    params_y: float = 0.10,
    run_y: float = 0.01,
) -> None:
    """Stack the parameter and run-type legends at compact display widths.

    The inline legend block above works for wide canvases. For figures rendered
    at roughly 500–700 CSS pixels, two centered rows preserve the same semantic
    hierarchy without compressing labels until they touch.
    """
    param_handles, param_labels = params_legend_handles(palette, params)
    fig.legend(
        param_handles,
        param_labels,
        ncol=len(param_handles),
        title="Model parameters",
        loc="lower center",
        bbox_to_anchor=(0.5, params_y),
        **TWO_LEGEND_KW,
    )
    run_handles, run_labels = shape_legend_handles(include_reference=include_reference)
    fig.legend(
        run_handles,
        run_labels,
        ncol=len(run_handles),
        title="Run type",
        loc="lower center",
        bbox_to_anchor=(0.5, run_y),
        **TWO_LEGEND_KW,
    )


def attach_stacked_legends_right(
    fig,
    ax,
    palette: dict,
    params: list[int],
    *,
    include_reference: bool = True,
) -> None:
    """Stack two one-column legend groups to the right of one subplot."""
    param_handles, param_labels = params_legend_handles(palette, params)
    param_legend = fig.legend(
        param_handles,
        param_labels,
        ncol=1,
        title="Model parameters",
        loc="upper left",
        bbox_to_anchor=(1.04, 1.08),
        bbox_transform=ax.transAxes,
        alignment="left",
        **TWO_LEGEND_KW,
    )
    fig.add_artist(param_legend)
    run_handles, run_labels = shape_legend_handles(include_reference=include_reference)
    fig.legend(
        run_handles,
        run_labels,
        ncol=1,
        title="Run type",
        loc="lower left",
        bbox_to_anchor=(1.04, -0.10),
        bbox_transform=ax.transAxes,
        alignment="left",
        **TWO_LEGEND_KW,
    )


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
