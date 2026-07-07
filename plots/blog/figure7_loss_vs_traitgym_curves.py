"""Figure 7 (M·LLR): VEP AUPRC across training steps, by model scale (new eval).

Redo of Eric's Fig 7 (1×3 panels, 128M / 1B / 4B) from the **new** offline eval:
per scale, the min-max-normalized trajectory of missense / promoter / splicing
AUPRC across the scored HF checkpoints, via ``blog_metrics`` on the #364 ladder
intermediates. Eric's version used dense wandb in-training history; ours samples
the saved HF steps (sparser, but the same offline eval as every other figure).
Normalizing each trace min→max lets shapes be compared across traits/scales;
absolute levels live in Figs 5/6.

Run:  uv run python -m plots.blog.figure7_loss_vs_traitgym_curves
Out:  plots/output/blog/figure7_loss_vs_traitgym_curves__mendelian_llr.{svg,png,pdf}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from scipy.ndimage import gaussian_filter1d

from marin_dna.pipelines.evals.blog_metrics import read_llr_metrics
from plots.blog._style.figure_style import EARTH_QUAL, FIGURE_WIDTH, figsize
from plots.blog._style.savefig import save_figure

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "blog"

# (label, run stem, scored HF steps) — small → large (left → right panels).
SCALES: tuple[tuple[str, str, tuple[int, ...]], ...] = (
    (
        "128M",
        "scaling-v0.5-h896-p128M",
        (160000, 170000, 180000, 190000, 200000, 210000, 215573),
    ),
    (
        "1B",
        "scaling-v0.5-h1920-p1B",
        (
            10000,
            20000,
            140000,
            150000,
            160000,
            170000,
            180000,
            190000,
            200000,
            210000,
            215573,
        ),
    ),
    (
        "4B",
        "scaling-v0.5-h2944-p4B",
        (
            80000,
            90000,
            100000,
            110000,
            120000,
            130000,
            140000,
            150000,
            160000,
            170000,
            180000,
            190000,
            200000,
            210000,
            215573,
        ),
    ),
)
# (subset, label, EARTH_QUAL color slot) — same colors as Figs 5/6.
TRAITS: tuple[tuple[str, str, int], ...] = (
    ("missense_variant", "missense", 0),
    ("tss_proximal", "promoter", 1),
    ("splicing", "splicing", 4),
)
_SMOOTH_SIGMA = 1.5


def _trajectory(
    stem: str, steps: tuple[int, ...]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{subset: (steps, auprc)} across the scored HF steps of one scale (skips unscored)."""
    acc: dict[str, tuple[list[int], list[float]]] = {t: ([], []) for t, _, _ in TRAITS}
    for s in steps:
        try:
            df = read_llr_metrics(f"{stem}-step-{s}", "mendelian_traits")
        except (LookupError, FileNotFoundError, OSError):
            continue  # not scored yet
        vals = {row["subset"]: row["value"] for row in df.iter_rows(named=True)}
        for t, _, _ in TRAITS:
            if t in vals:
                acc[t][0].append(s)
                acc[t][1].append(vals[t])
    return {t: (np.array(x), np.array(y)) for t, (x, y) in acc.items()}


def build() -> None:
    color_for = {t: EARTH_QUAL[slot] for t, _, slot in TRAITS}
    label_for = {t: lab for t, lab, _ in TRAITS}

    fig, axes = plt.subplots(1, 3, figsize=figsize(FIGURE_WIDTH, 4.4), sharey=True)
    for ax, (scale_label, stem, steps) in zip(axes, SCALES, strict=True):
        traj = _trajectory(stem, steps)
        for t, _, _ in TRAITS:
            x, y = traj[t]
            if len(x) < 2:
                continue
            y_min, y_max = float(y.min()), float(y.max())
            y_norm = (
                (y - y_min) / (y_max - y_min) if y_max > y_min else np.zeros_like(y)
            )
            ax.scatter(
                x,
                y_norm,
                color=color_for[t],
                s=16,
                edgecolors="k",
                linewidths=0.3,
                alpha=0.4,
                zorder=3,
            )
            smooth = (
                gaussian_filter1d(y_norm, sigma=_SMOOTH_SIGMA, mode="nearest")
                if len(y_norm) >= 3
                else y_norm
            )
            ax.plot(x, smooth, color=color_for[t], linewidth=1.8, zorder=4)
        ax.set_title(scale_label, fontsize=10)
        ax.grid(False)
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda v, _: f"{v / 1000:.0f}k" if v >= 1000 else f"{v:.0f}")
        )
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0.0, 1.0])
        ax.set_yticklabels(["Min", "Max"])

    axes[0].set_ylabel("AUPRC (min–max normalized)")
    axes[1].set_xlabel("training step")

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=color_for[t],
            markeredgecolor="k",
            markeredgewidth=0.3,
            markersize=6,
            linewidth=1.3,
            label=label_for[t],
        )
        for t, _, _ in TRAITS
    ]
    fig.subplots_adjust(top=0.80, bottom=0.22, left=0.06, right=0.99, wspace=0.06)
    fig.suptitle(
        "Parameter scaling — VEP AUPRC across training steps (new eval)",
        fontsize=11,
        y=0.94,
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.10),
        ncol=3,
        frameon=False,
        title="variant type",
        title_fontsize=9,
        fontsize=9,
    )
    save_figure(fig, OUTPUT_DIR, "figure7_loss_vs_traitgym_curves__mendelian_llr")


if __name__ == "__main__":
    build()
