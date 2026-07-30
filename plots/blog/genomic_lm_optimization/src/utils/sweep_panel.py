"""Transfer-sweep panel renderer shared by Figures 1–3.

`plot_axis` draws one hyperparameter axis: sweep points, per-scale positive
controls, and (optionally) the negative control, with one series per parameter
scale on an evenly-spaced categorical grid.
"""

from __future__ import annotations

import math

import pandas as pd

from marin_dna.blog_figure_typography import FIGURE_NOTE_SIZE_PX
from utils.figure_style import X_LABEL_PAD

CONTROL_ROLES = ("positive-control", "negative-control")


def _interpolate_x(
    value: float, grid_values: list[float], grid_xs: list[int], log_scale: bool
) -> float:
    """Map a hparam value to an evenly-spaced grid x (log/linear). Extrapolates."""
    if log_scale:
        t = math.log(value)
        gt = [math.log(g) for g in grid_values]
    else:
        t, gt = value, list(grid_values)
    if t <= gt[0]:
        slope = (grid_xs[1] - grid_xs[0]) / (gt[1] - gt[0])
        return grid_xs[0] + slope * (t - gt[0])
    if t >= gt[-1]:
        slope = (grid_xs[-1] - grid_xs[-2]) / (gt[-1] - gt[-2])
        return grid_xs[-1] + slope * (t - gt[-1])
    for i in range(len(gt) - 1):
        if gt[i] <= t <= gt[i + 1]:
            frac = (t - gt[i]) / (gt[i + 1] - gt[i])
            return grid_xs[i] + frac * (grid_xs[i + 1] - grid_xs[i])
    return grid_xs[-1]


def plot_axis(
    ax,
    df: pd.DataFrame,
    *,
    axis_role: str,
    axis_field: str,
    axis_label: str,
    log_scale: bool,
    value_formatter,
    palette: dict,
    include_negative_control: bool = True,
    y_field: str = "eval_loss",
    y_label: str = "Loss",
    tick_stride: int = 1,
) -> None:
    """Plot `y_field` vs `axis_field` for the relevant rows. One series per param scale.

    Includes:
      - sweep points for runs with role == axis_role  (circles)
      - per-scale positive-control                    (square, on grid center)
      - 1B-only negative-control                      (square, x-position interpolated)
    """
    sub = (
        df[df["role"].isin([axis_role, *CONTROL_ROLES])]
        .dropna(subset=[axis_field, y_field])
        .copy()
    )

    # Union x grid: every unique axis value seen in sweep + positive-control. Negative
    # control's value is interpolated on top of this grid (it's untransferred).
    grid_mask = sub["role"].isin([axis_role, "positive-control"])
    grid_values = sorted(sub.loc[grid_mask, axis_field].unique().tolist())
    grid_xs = list(range(len(grid_values)))
    grid_lookup = {v: i for i, v in enumerate(grid_values)}

    for params in sorted(sub["params"].unique()):
        scale = sub[sub["params"] == params]
        color = palette[int(params)]

        # Per-scale line: connect mean y at each axis value, ascending.
        line_pts = (
            scale[scale["role"].isin([axis_role, "positive-control"])]
            .groupby(axis_field, as_index=False)[y_field]
            .mean()
            .sort_values(axis_field)
        )
        if not line_pts.empty:
            line_xs = [grid_lookup[v] for v in line_pts[axis_field]]
            ax.plot(
                line_xs,
                line_pts[y_field].values,
                color=color,
                linewidth=1,
                alpha=0.6,
                zorder=2,
            )

        # Marker size shared across roles so circles / squares / diamonds read as
        # equally weighted; only the shape encodes role.
        marker_size = 60

        # Sweep points: circles, raw individual runs.
        sweep = scale[scale["role"] == axis_role]
        if not sweep.empty:
            sweep_xs = [grid_lookup[v] for v in sweep[axis_field]]
            ax.scatter(
                sweep_xs,
                sweep[y_field].values,
                s=marker_size,
                color=color,
                marker="o",
                edgecolors="k",
                linewidths=0.4,
                zorder=3,
            )

        # Positive control: square, sits at the grid center value (same for every scale).
        pos = scale[scale["role"] == "positive-control"]
        if not pos.empty:
            pos_xs = [grid_lookup[v] for v in pos[axis_field]]
            ax.scatter(
                pos_xs,
                pos[y_field].values,
                s=marker_size,
                color=color,
                marker="s",
                edgecolors="k",
                linewidths=0.6,
                zorder=4,
            )

        # Negative control (1B only): diamond at the (interpolated) untransferred value.
        neg = (
            scale[scale["role"] == "negative-control"]
            if include_negative_control
            else scale.iloc[0:0]
        )
        if not neg.empty:
            for _, row in neg.iterrows():
                x = _interpolate_x(
                    float(row[axis_field]), grid_values, grid_xs, log_scale
                )
                ax.scatter(
                    [x],
                    [row[y_field]],
                    s=marker_size,
                    color=color,
                    marker="D",
                    edgecolors="k",
                    linewidths=0.6,
                    zorder=4,
                )

    assert tick_stride >= 1
    shown = list(range(0, len(grid_values), tick_stride))
    if shown[-1] != len(grid_values) - 1:
        shown.append(len(grid_values) - 1)
    ax.set_xticks([grid_xs[index] for index in shown])
    ax.set_xticklabels(
        [value_formatter(grid_values[index]) for index in shown],
        rotation=30,
        ha="right",
        fontsize=FIGURE_NOTE_SIZE_PX,
    )
    ax.set_xlabel(axis_label, labelpad=X_LABEL_PAD)
    ax.set_ylabel(y_label)
    ax.grid(False)
