"""Figure 4: parameter-scaling loss curves + Kaplan scaling-law inset."""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from figures.data import FIGURES_DIR, save
from marin_dna.blog_figure_typography import (
    FIGURE_BODY_SIZE_PX,
    FIGURE_TITLE_SIZE_PX,
)
from utils.figure_style import (
    FIGURE_WIDTH,
    X_LABEL_PAD,
    attach_params_legend_below,
    figsize,
    set_plain_decimal_yticks,
)

# Scaling-sweep summary: N, D, C ranges across all 8 models. N spans the full
# range (46M–4B); D is one epoch over the training mixture (~84B tokens);
# C range is taken from `#### Parameter scaling sweep` in outline.md.
_SCALING_SUBTITLE = "N = 46M–4B, D = 84B, C = 2.5 × 10¹⁹–2.1 × 10²¹"

KAPLAN_FIT_REPORT_PATH = FIGURES_DIR / "figure4_loss_scaling.txt"


def _fit_kaplan_law(
    params: np.ndarray, losses: np.ndarray
) -> tuple[float, float, float]:
    """Fit L(N) = A * N^(-alpha) + L_inf via nonlinear least squares.

    Returns (A, alpha, L_inf). L_inf is the irreducible-loss asymptote.

    Raises RuntimeError if the optimizer did not converge, if any scipy warning
    fires during the call, if the parameter covariance is non-finite (parameters
    unidentifiable), or if any fit parameter lands within `bound_tol_frac` of a
    bound (which would indicate the bound is binding rather than the data
    determining the parameter).
    """
    from scipy.optimize import curve_fit

    def kaplan(N, A, alpha, L_inf):
        return A * N ** (-alpha) + L_inf

    L_min = float(np.min(losses))
    L_inf0 = max(0.0, L_min - 0.1)
    alpha0 = 0.1
    N_med = float(np.median(params))
    A0 = max((float(np.median(losses)) - L_inf0) * (N_med**alpha0), 1.0)

    lower = np.array([0.0, 0.0, 0.0])
    upper = np.array([np.inf, 5.0, L_min])
    names = ("A", "alpha", "L_inf")

    with warnings.catch_warnings():
        # Treat any scipy/numpy warning during the fit (covariance estimation,
        # convergence, divide-by-zero, etc.) as a hard failure.
        warnings.simplefilter("error")
        popt, pcov, infodict, mesg, ier = curve_fit(
            kaplan,
            params,
            losses,
            p0=[A0, alpha0, L_inf0],
            bounds=(lower, upper),
            maxfev=10000,
            full_output=True,
        )

    # ier ∈ {1, 2, 3, 4} indicates a successful termination of the underlying
    # least-squares algorithm; everything else (0, 5, ...) is a non-convergence.
    if ier not in (1, 2, 3, 4):
        raise RuntimeError(f"Kaplan fit did not converge: ier={ier}, mesg={mesg!r}")

    if not np.all(np.isfinite(pcov)):
        raise RuntimeError(
            f"Kaplan fit covariance is non-finite (parameters unidentifiable): pcov={pcov!r}"
        )

    # Reject fits where any parameter is glued to a bound. For finite bounds use a
    # fraction of the bound interval; for the semi-infinite A upper bound use a
    # fraction of the parameter scale.
    bound_tol_frac = 0.01
    for val, lo, hi, name in zip(popt, lower, upper, names, strict=True):
        scale = (hi - lo) if np.isfinite(hi) else max(abs(val), 1.0)
        margin = bound_tol_frac * scale
        if val - lo < margin or (np.isfinite(hi) and hi - val < margin):
            raise RuntimeError(
                f"Kaplan fit parameter {name}={val:.6g} is at bound "
                f"(lo={lo}, hi={hi}, margin={margin:.3g})"
            )

    A_fit, alpha_fit, L_inf_fit = float(popt[0]), float(popt[1]), float(popt[2])
    _write_kaplan_fit_report(
        params, losses, A_fit, alpha_fit, L_inf_fit, pcov, lower, upper, names
    )
    return A_fit, alpha_fit, L_inf_fit


def _write_kaplan_fit_report(
    params: np.ndarray,
    losses: np.ndarray,
    A: float,
    alpha: float,
    L_inf: float,
    pcov: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    names: tuple[str, ...],
) -> None:
    """Write Kaplan fit parameters, bounds, std-errs, and residual stats to a report file."""
    pred = A * params ** (-alpha) + L_inf
    resid = losses - pred
    rmse = float(np.sqrt(np.mean(resid**2)))
    max_abs = float(np.max(np.abs(resid)))
    perr = np.sqrt(np.diag(pcov))
    values = (A, alpha, L_inf)

    lines = [
        "Kaplan scaling-law fit",
        "======================",
        "",
        "L(N) = A · N^(-α) + L_∞",
        "",
        f"data points : {len(params)}",
        f"RMSE        : {rmse:.4g}",
        f"max |resid| : {max_abs:.4g}",
        "",
        f"{'param':<6} {'value':>14} {'± std err':>14}    bounds",
    ]
    for name, val, err, lo, hi in zip(names, values, perr, lower, upper, strict=True):
        hi_str = "∞" if not np.isfinite(hi) else f"{hi:.4g}"
        lines.append(f"{name:<6} {val:>14.6g} {err:>14.3g}    [{lo:.4g}, {hi_str}]")

    KAPLAN_FIT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    KAPLAN_FIT_REPORT_PATH.write_text("\n".join(lines) + "\n")


def build(history: pd.DataFrame, results: pd.DataFrame, palette: dict) -> None:
    """1x2: train/eval loss vs step. Inset on eval panel: final loss vs params with Kaplan fit."""
    name_to_params = dict(zip(results["run_name"], results["params"], strict=True))
    history = history.assign(params=history["run_name"].map(name_to_params))
    # Log-x cannot show step=0; the step-0 eval is pre-training and not informative anyway.
    history = history[history["step"] > 0]

    fig, axes = plt.subplots(1, 2, figsize=figsize(FIGURE_WIDTH, 6.5))
    panels = [
        (axes[0], "train/loss", "Train Loss"),
        (axes[1], "eval/loss", "Validation Loss"),
    ]
    # Eval-loss lines need a high zorder so they draw over the inset on the right panel.
    line_zorder = {axes[0]: 2, axes[1]: 20}
    for ax, metric, ylabel in panels:
        sub = history[history["metric"] == metric]
        for params in sorted(sub["params"].unique()):
            line = sub[sub["params"] == params].sort_values("step")
            ax.plot(
                line["step"],
                line["value"],
                color=palette[int(params)],
                linewidth=1.3,
                zorder=line_zorder[ax],
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Step", labelpad=X_LABEL_PAD)
        ax.set_ylabel(ylabel)
        ax.grid(False)
        # Loss values span ~0.5–1.5; render as plain decimals rather than 6×10⁻¹ etc.
        set_plain_decimal_yticks(ax)

    # Inset on eval-loss panel: final eval loss vs params with Kaplan scaling-law fit.
    # Sits in the bottom-left empty region (low-step / low-loss has no curve data there).
    _attach_kaplan_inset(axes[1], results, palette)

    # Keep the title and subtitle as plain Unicode text so the SVG normalizer
    # can render both in the same page font as the rest of the figure.
    fig.text(
        0.5,
        0.965,
        "Parameter scaling — Loss curves & scaling law",
        ha="center",
        va="center",
        fontsize=FIGURE_TITLE_SIZE_PX,
    )
    fig.text(
        0.5,
        0.925,
        _SCALING_SUBTITLE,
        ha="center",
        va="center",
        fontsize=FIGURE_BODY_SIZE_PX,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.90))

    params_present = sorted({int(p) for p in history["params"].dropna().unique()})
    attach_params_legend_below(
        fig, palette, params_present, width_scale=0.4, handlelength=1.0
    )
    save(fig, "figure4_loss_scaling")


def _attach_kaplan_inset(parent_ax, results: pd.DataFrame, palette: dict) -> None:
    """Embed a small params-vs-loss panel with Kaplan fit inside `parent_ax`.

    Inset axes uses parent-axes-fraction coords. The fit equation/constants sit
    in the open space just to the right of the inset.
    """
    fit_df = results.dropna(subset=["params", "eval_loss"]).sort_values("params")
    P = fit_df["params"].astype(float).to_numpy()
    L = fit_df["eval_loss"].astype(float).to_numpy()
    A, alpha, L_inf = _fit_kaplan_law(P, L)
    pred = A * P ** (-alpha) + L_inf
    ss_res = float(np.sum((L - pred) ** 2))
    ss_tot = float(np.sum((L - L.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # zorder=0 puts the inset under parent's eval-loss lines (which use zorder=20),
    # so the curves visually cross over the inset like a framed window beneath them.
    # Keep the bottom (y0) where it is; the height controls only the top edge.
    inset_bounds = (0.10, 0.17, 0.27, 0.40)
    inset = parent_ax.inset_axes(list(inset_bounds), zorder=0)

    # Replace the default rectangular background/spines with a rounded FancyBboxPatch
    # placed in parent axes-fraction coords, so the inset reads as a card with rounded edges.
    inset.patch.set_visible(False)
    for spine in inset.spines.values():
        spine.set_visible(False)
    rounded_bg = FancyBboxPatch(
        (inset_bounds[0], inset_bounds[1]),
        inset_bounds[2],
        inset_bounds[3],
        boxstyle="round,pad=0,rounding_size=0.025",
        transform=parent_ax.transAxes,
        facecolor="none",
        edgecolor="black",
        linewidth=0.2,
        zorder=0,
        clip_on=False,
    )
    parent_ax.add_patch(rounded_bg)

    # Y-range: zoom to the data points with a little padding. The asymptote and
    # fit details live in a text block to the right, so the inset needn't reach
    # all the way down to L_inf.
    dataspan = float(L.max() - L.min()) or 1.0
    inset.set_ylim(L.min() - 0.12 * dataspan, L.max() + 0.12 * dataspan)

    P_grid = np.geomspace(P.min() * 0.92, P.max() * 1.08, 200)
    inset.plot(
        P_grid,
        A * P_grid ** (-alpha) + L_inf,
        color="0.2",
        linewidth=1.3,
        alpha=0.92,
        zorder=2,
    )
    for p, loss in zip(P, L, strict=True):
        inset.scatter(
            [p],
            [loss],
            color=palette[int(p)],
            s=32,
            edgecolors="k",
            linewidths=0.5,
            zorder=3,
        )

    inset.set_xscale("log")
    inset.set_xlabel("Parameters (N)", fontsize=9, labelpad=2)
    inset.set_ylabel("")
    inset.tick_params(labelsize=8, length=2.5, pad=1)
    inset.minorticks_off()
    inset.grid(False)
    set_plain_decimal_yticks(inset)

    # Fit equation + constants in the open space just right of the inset (the
    # empty low-loss/early-step corner), bottom-aligned to the inset plot. No
    # background box (transparent) so it doesn't mask the loss curves; high
    # zorder keeps the text itself on top.
    parent_ax.text(
        inset_bounds[0] + inset_bounds[2] + 0.03,
        inset_bounds[1],
        "L(N) = A · N⁻ᵅ + L∞\n"
        f"A = {A:.3g}, α = {alpha:.3f}\n"
        f"L∞ = {L_inf:.3f}, R² = {r2:.3f}",
        transform=parent_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="0.15",
        linespacing=1.6,
        zorder=25,
    )
