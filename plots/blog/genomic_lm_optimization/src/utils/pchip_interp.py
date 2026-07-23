"""Shared PCHIP interpolation utilities for the figure set.

Two consumers:
  - appendix/pooled_vs_unpooled.py: overlay independently-fit PCHIP curves on a
    common grid restricted to data overlap (no extrapolation).
  - figure8_loss_vs_traitgym_correlation.py: align an eval/loss series and an
    lm_eval auprc series at matched steps (within the overlap) so they can
    be correlated as a time series.

Both rely on the same primitive: fit a monotone piecewise-cubic Hermite
interpolant on log10(x) and only evaluate inside the data range.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.interpolate import PchipInterpolator


def clean(x_raw: np.ndarray, y_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop non-finite values and any x<=0 (log10 is undefined there)."""
    x = np.asarray(x_raw, dtype=float)
    y = np.asarray(y_raw, dtype=float)
    m = np.isfinite(y) & np.isfinite(x) & (x > 0)
    return x[m], y[m]


def fit_curve(
    x_raw: np.ndarray, y_raw: np.ndarray
) -> tuple[np.ndarray, np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """PCHIP interpolator of y vs log10(x).

    PCHIP passes through every point but never overshoots, so noisy or
    sigmoid-shaped curves don't get the boundary oscillations a smoothing
    spline produces. Outside the data range we clamp to the endpoint value.
    """
    s, y = clean(x_raw, y_raw)
    if len(s) == 0:
        raise ValueError("no finite points to fit")
    if len(s) == 1:
        y0 = float(y[0])
        return s, y, lambda x: np.full(np.asarray(x, dtype=float).shape, y0)
    logx = np.log10(s)
    pch = PchipInterpolator(logx, y, extrapolate=False)
    y_lo, y_hi = float(y[0]), float(y[-1])
    lo, hi = float(logx[0]), float(logx[-1])

    def f(x):
        xl = np.log10(np.asarray(x, dtype=float))
        out = pch(xl)
        out = np.where(np.isnan(out) & (xl < lo), y_lo, out)
        out = np.where(np.isnan(out) & (xl > hi), y_hi, out)
        return out

    return s, y, f


def overlap_range(x_a: np.ndarray, x_b: np.ndarray) -> tuple[float, float] | None:
    """Return (lo, hi) of the x-overlap between two series, or None if empty.

    "Overlap" is the tolerance in this methodology: we never evaluate either
    fitted curve outside the range where the other series has measurements.
    """
    if len(x_a) < 2 or len(x_b) < 2:
        return None
    lo = max(float(np.min(x_a)), float(np.min(x_b)))
    hi = min(float(np.max(x_a)), float(np.max(x_b)))
    if not (lo < hi):
        return None
    return lo, hi


def interp_on_overlap(
    x_a: np.ndarray,
    y_a: np.ndarray,
    x_b: np.ndarray,
    y_b: np.ndarray,
    n_grid: int = 200,
    grid: np.ndarray | None = None,
):
    """Evaluate PCHIP fits of A and B on a common grid restricted to overlap.

    If `grid` is None, use n_grid log-spaced points across the overlap range.
    If `grid` is provided, it is filtered to the overlap range (so the caller
    can pass one of the series' native x-values for honest sample sizing).

    Returns (grid_used, y_a_eval, y_b_eval) or (None, None, None) when there is
    no usable overlap.
    """
    x_a, y_a = clean(x_a, y_a)
    x_b, y_b = clean(x_b, y_b)
    rng = overlap_range(x_a, x_b)
    if rng is None:
        return None, None, None
    lo, hi = rng
    _, _, fa = fit_curve(x_a, y_a)
    _, _, fb = fit_curve(x_b, y_b)
    if grid is None:
        g = np.geomspace(lo, hi, n_grid)
    else:
        g = np.asarray(grid, dtype=float)
        g = g[(g >= lo) & (g <= hi)]
        if len(g) < 2:
            return None, None, None
    return g, fa(g), fb(g)
