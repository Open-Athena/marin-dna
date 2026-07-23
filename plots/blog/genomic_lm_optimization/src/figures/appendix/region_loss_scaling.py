"""Per-region Kaplan scaling fits.

Companion to `figure4_loss_scaling` in src/figures.py, which fits the pooled
eval loss vs N. Here we fit `L(N) = A * N^(-alpha) + L_inf` separately to each
of the three genomic regions (cds, upstream, downstream) using the final
`eval/val_<region>/loss` from each parameter-scaling-sweep run.

Usage:
    uv run src/figures/appendix/region_loss_scaling.py
"""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.savefig import save_figure  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
FIGURES_DIR = (
    ROOT.parents[2]
    / "plots"
    / "output"
    / "blog"
    / "genomic_lm_optimization"
    / "appendix"
)

WANDB_PROJECT = "eric-czech/marin"
RUN_PREFIX = "dna-bolinas-scaling-v0.5-"
REGIONS = ("cds", "upstream", "downstream")
COLORS = {"cds": "#4C72B0", "upstream": "#DD8452", "downstream": "#55A868"}

FIGURE_WIDTH = 12.0
FIGURE_HEIGHT = 5.0


def _params_from_tag(tags: list[str], run_name: str) -> int:
    for t in tags:
        if t.startswith("params="):
            return int(t.removeprefix("params="))
    raise KeyError(f"run {run_name!r}: no params= tag in {tags}")


def fetch() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {region -> (params_array, loss_array)} sorted by params."""
    api = wandb.Api(timeout=300)
    runs = list(
        api.runs(
            WANDB_PROJECT,
            filters={"display_name": {"$regex": f"^{re.escape(RUN_PREFIX)}"}},
        )
    )
    rows: list[dict] = []
    for r in runs:
        row = {"params": _params_from_tag(r.tags, r.name)}
        for region in REGIONS:
            v = r.summary.get(f"eval/val_{region}/loss")
            row[region] = float(v) if v is not None else float("nan")
        rows.append(row)
    rows.sort(key=lambda d: d["params"])
    out = {}
    for region in REGIONS:
        P = np.array([row["params"] for row in rows], dtype=float)
        L = np.array([row[region] for row in rows], dtype=float)
        mask = np.isfinite(L)
        out[region] = (P[mask], L[mask])
    return out


def fit_kaplan(P: np.ndarray, L: np.ndarray) -> tuple[float, float, float]:
    """Fit L(N) = A * N^(-alpha) + L_inf via nonlinear least squares."""

    def kaplan(N, A, alpha, L_inf):
        return A * N ** (-alpha) + L_inf

    L_min = float(np.min(L))
    L_inf0 = max(0.0, L_min - 0.1)
    alpha0 = 0.1
    N_med = float(np.median(P))
    A0 = max((float(np.median(L)) - L_inf0) * (N_med**alpha0), 1.0)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        popt, _ = curve_fit(
            kaplan,
            P,
            L,
            p0=[A0, alpha0, L_inf0],
            bounds=([0.0, 0.0, 0.0], [np.inf, 5.0, L_min]),
            maxfev=10000,
        )
    return tuple(popt)  # type: ignore[return-value]


def _save(fig, name: str) -> None:
    save_figure(fig, FIGURES_DIR, name)


def main() -> None:
    data = fetch()
    print(f"Pulled {len(next(iter(data.values()))[0])} runs.")

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    for region in REGIONS:
        P, L = data[region]
        if len(P) < 4:
            print(f"  {region}: only {len(P)} points, skipping fit")
            continue
        A, alpha, L_inf = fit_kaplan(P, L)
        print(f"  {region}: A={A:.3g}, alpha={alpha:.3f}, L_inf={L_inf:.4f}")
        color = COLORS[region]
        ax.scatter(
            P,
            L,
            color=color,
            s=42,
            edgecolors="k",
            linewidths=0.5,
            zorder=3,
            label=f"{region}: α={alpha:.2f}, L∞={L_inf:.3f}",
        )
        P_grid = np.geomspace(P.min() * 0.92, P.max() * 1.08, 200)
        ax.plot(
            P_grid,
            A * P_grid ** (-alpha) + L_inf,
            color=color,
            linewidth=1.3,
            alpha=0.9,
            zorder=2,
        )
        ax.axhline(
            L_inf,
            color=color,
            linewidth=0.7,
            linestyle=(0, (3, 2)),
            alpha=0.6,
            zorder=1,
        )

    ax.set_xscale("log")
    ax.set_xlabel("model params (N)")
    ax.set_ylabel("eval/val_<region>/loss")
    ax.set_title("Per-region Kaplan scaling fit (parameter scaling sweep, v0.5)")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    fig.tight_layout()
    _save(fig, "region_loss_scaling")


if __name__ == "__main__":
    main()
