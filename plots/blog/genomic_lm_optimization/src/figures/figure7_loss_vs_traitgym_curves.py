"""Figure 7: per-model VEP AUPRC curves across training steps for selected variant types.

Companion to `figure8_loss_vs_traitgym_correlation.py`. That figure summarized the
per-(model, variant) Spearman ρ between loss and AUPRC across training; this
figure shows the raw AUPRC trajectories the correlation was computed from,
for a representative subset of (model, variant) combinations:

  models   : 128M, 1B, 4B  (small / mid / large from the v0.5 scaling sweep)
  variants : missense, promoter (TSS-proximal), splicing

The 128M run is the outlier from the correlation figure (near-zero ρ across
most variants); 1B and 4B let the reader see how the curves change as the
model scales up.

Layout: 1x3, one panel per model. AUPRC on the y-axis (left panel only),
linear training step on the x-axis (middle panel only). Variant colors reuse
the tab10 slots from `figures.py:VEP_PANELS` so missense / promoter / splicing
read as the same color here, in figure 5, and in figure 6.

Output:
  plots/output/blog/genomic_lm_optimization/figure7_loss_vs_traitgym_curves.{png,pdf,svg}

Usage:
  uv run src/figures/figure7_loss_vs_traitgym_curves.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.eval_history import DEFAULT_MAX_GAP_FRACTION, dedup_eval_history  # noqa: E402
from utils.figure_style import EARTH_QUAL, figsize  # noqa: E402
from utils.pchip_interp import clean  # noqa: E402
from utils.savefig import save_figure  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "parameter_scaling_results.csv"
FIGURES_DIR = ROOT.parents[2] / "plots" / "output" / "blog" / "genomic_lm_optimization"

WANDB_PROJECT = "eric-czech/marin"
TRAITGYM_PREFIX = "lm_eval/traitgym_mendelian_v2_255"

# Models picked from the v0.5 scaling sweep. Order = small → large for the
# 1x3 panels (left → right).
MODELS: tuple[tuple[str, str], ...] = (
    ("128M", "dna-bolinas-scaling-v0.5-h896-p128M"),
    ("1B", "dna-bolinas-scaling-v0.5-h1920-p1B"),
    ("4B", "dna-bolinas-scaling-v0.5-h2944-p4B"),
)

# (trait key, display label, EARTH_QUAL slot from figures/data.py:VEP_PANELS).
# Slots: missense=0, tss_proximal=1, splicing=4.
TRAITS: tuple[tuple[str, str, int], ...] = (
    ("missense_variant", "missense", 0),
    ("tss_proximal", "promoter", 1),
    ("splicing", "splicing", 4),
)

# Match figures.py.
FIGURE_WIDTH = 12.0
FIGURE_HEIGHT = 4.4
_X_LABEL_PAD = 0

# Gaussian-kernel smoothing applied to each per-trace AUPRC trajectory.
# sigma is in units of "evals" (≈one log step apart). σ≈2 averages each point
# with its ±2 neighbors at full weight and tails off smoothly beyond — gives a
# cleaner trend line than a hard-edge moving average without the boundary
# artifacts of smoothing splines.
_SMOOTH_SIGMA = 2.0


def _resolve(api: wandb.Api, name: str):
    runs = list(
        api.runs(
            WANDB_PROJECT, filters={"display_name": {"$regex": f"^{re.escape(name)}$"}}
        )
    )
    if not runs:
        raise RuntimeError(f"no run matching display_name {name!r}")
    if len(runs) > 1:
        runs.sort(key=lambda r: r.created_at, reverse=True)
        print(f"  WARN: {len(runs)} runs match {name!r}; taking newest")
    return runs[0]


def _num_train_steps(run) -> int:
    """Pull `trainer.num_train_steps` from a (nested) wandb run.config."""
    cfg = run.config
    for part in ("trainer", "num_train_steps"):
        if not isinstance(cfg, dict) or part not in cfg:
            raise RuntimeError(
                f"{run.name}: trainer.num_train_steps missing in run.config"
            )
        cfg = cfg[part]
    return int(cfg)


def fetch_all() -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    api = wandb.Api(timeout=300)
    auprc_keys = [f"{TRAITGYM_PREFIX}/{t}/auprc" for t, _, _ in TRAITS]
    out: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for label, run_name in MODELS:
        print(f"fetching {label}: {run_name}")
        run = _resolve(api, run_name)
        nts = _num_train_steps(run)
        max_gap = DEFAULT_MAX_GAP_FRACTION * nts
        df = run.history(keys=auprc_keys, samples=10000)
        traces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for trait, _disp, _slot in TRAITS:
            col = f"{TRAITGYM_PREFIX}/{trait}/auprc"
            if df.empty or col not in df.columns:
                traces[trait] = (np.array([]), np.array([]))
                continue
            sub = df[["_step", col]].dropna().sort_values("_step")
            x, y = clean(
                sub["_step"].to_numpy(dtype=float), sub[col].to_numpy(dtype=float)
            )
            x, y = dedup_eval_history(x, y, max_gap)
            traces[trait] = (x, y)
        out[label] = traces
        ns = {t: len(traces[t][0]) for t, _, _ in TRAITS}
        print(
            f"  num_train_steps={nts}  max_gap={max_gap:.1f}  AUPRC rows per trait: {ns}"
        )
    return out


def _save(fig, name: str) -> None:
    save_figure(fig, FIGURES_DIR, name)


def _fmt_step(x: float, _pos) -> str:
    """Format step values as e.g. '50k', '100k' for compact tick labels."""
    if x == 0:
        return "0"
    if abs(x) >= 1000:
        v = x / 1000.0
        return f"{v:.0f}k" if v == int(v) else f"{v:.1f}k"
    return f"{x:.0f}"


def plot(data: dict) -> None:
    color_for = {trait: EARTH_QUAL[slot] for trait, _, slot in TRAITS}
    label_for = {trait: label for trait, label, _ in TRAITS}

    fig, axes = plt.subplots(
        1, 3, figsize=figsize(FIGURE_WIDTH, FIGURE_HEIGHT), sharey=True
    )

    for ax, (model_label, _run_name) in zip(axes, MODELS, strict=True):
        traces = data[model_label]
        for trait, _disp, _slot in TRAITS:
            x, y = traces[trait]
            if len(x) < 2:
                continue
            # Min/max normalize each trace so shapes can be compared across
            # traits (and across panels) on a shared [0, 1] y-axis. Absolute
            # AUPRC levels live in figures 5 / 6.
            y_min, y_max = float(np.min(y)), float(np.max(y))
            y_norm = (
                (y - y_min) / (y_max - y_min) if y_max > y_min else np.zeros_like(y)
            )
            # Raw measurements as semi-transparent scatter points; Gaussian
            # smoothed trace as the trend line on top.
            ax.scatter(
                x,
                y_norm,
                color=color_for[trait],
                s=14,
                edgecolors="k",
                linewidths=0.3,
                alpha=0.35,
                zorder=3,
            )
            smooth = gaussian_filter1d(y_norm, sigma=_SMOOTH_SIGMA, mode="nearest")
            ax.plot(
                x,
                smooth,
                color=color_for[trait],
                linewidth=1.8,
                zorder=4,
                alpha=0.95,
            )
        ax.set_title(model_label, fontsize=10)
        ax.grid(False)
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_step))
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0.0, 1.0])
        ax.set_yticklabels(["Min", "Max"])

    axes[0].set_ylabel("AUPRC (min–max normalized)")
    axes[1].set_xlabel("training step", labelpad=_X_LABEL_PAD)

    # Shared variant legend below the panels (single strip, matches the
    # variant-colored chips used in figures 5/6).
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            color=color_for[t],
            markeredgecolor="k",
            markeredgewidth=0.3,
            markersize=6,
            linewidth=1.3,
            label=label_for[t],
        )
        for t, _, _ in TRAITS
    ]
    # Use explicit subplots_adjust (not tight_layout) so the title and legend
    # placement is deterministic — tight_layout silently re-pads around the
    # suptitle and legend and reopens the gap we're trying to close.
    fig.subplots_adjust(top=0.76, bottom=0.22, left=0.055, right=0.99, wspace=0.06)
    fig.suptitle(
        "Parameter scaling — VEP AUPRC across training steps",
        fontsize=11,
        y=0.925,
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.085),
        ncol=len(handles),
        frameon=False,
        title="variant type",
        title_fontsize=9,
        fontsize=9,
        handletextpad=0.4,
        columnspacing=2.2,
        borderpad=0.4,
    )
    _save(fig, "figure7_loss_vs_traitgym_curves")


def main() -> None:
    # Sanity-check that the picked model run names are in the scaling-sweep CSV.
    results = pd.read_csv(DATA_PATH)
    known = set(results["run_name"])
    missing = [n for _, n in MODELS if n not in known]
    if missing:
        raise RuntimeError(f"unknown scaling run names: {missing}")

    data = fetch_all()
    plot(data)


if __name__ == "__main__":
    main()
