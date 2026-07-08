"""Figure 7: VEP AUPRC across training steps, by model scale — all four worlds.

Redo of Eric's Fig 7 (1×3 panels, 128M / 1B / 4B) from the **new** offline eval,
rendered for every metric-world (Mendelian/SGE × zero-shot-LLR/probe): per scale,
the min-max-normalized trajectory of each trait's AUPRC across the scored HF
checkpoints, via ``_worlds`` readers on the #364 ladder intermediates. Eric's
version used dense wandb history; ours samples the saved HF steps. Normalizing each
trace min→max compares shapes across traits/scales; absolute levels live in Figs 5/6.

Run:  uv run python -m plots.blog.figure7_loss_vs_traitgym_curves
Out:  plots/output/blog/figure7_loss_vs_traitgym_curves__{world}.{svg,png,pdf}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from scipy.ndimage import gaussian_filter1d

from plots.blog._style.figure_style import EARTH_QUAL, FIGURE_WIDTH, figsize
from plots.blog._style.savefig import save_figure
from plots.blog._worlds import WORLDS, World

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
_SMOOTH_SIGMA = 1.5


def _trajectory(
    stem: str, steps: tuple[int, ...], world: World
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{subset: (steps, auprc)} across the scored HF steps of one scale (skips unscored).

    ``world.read(model_id)`` binds the dataset+method; below-gate probe subsets
    (null value) are skipped.
    """
    keys = [t for t, _, _ in world.traits]
    acc: dict[str, tuple[list[int], list[float]]] = {t: ([], []) for t in keys}
    for s in steps:
        try:
            df = world.read(f"{stem}-step-{s}")
        except (LookupError, FileNotFoundError, OSError):
            continue  # not scored yet
        vals = {row["subset"]: row["value"] for row in df.iter_rows(named=True)}
        for t in keys:
            if vals.get(t) is not None:
                acc[t][0].append(s)
                acc[t][1].append(vals[t])
    return {t: (np.array(x), np.array(y)) for t, (x, y) in acc.items()}


def build(world: World) -> None:
    color_for = {t: EARTH_QUAL[slot] for t, _, slot in world.traits}
    label_for = {t: lab for t, lab, _ in world.traits}

    fig, axes = plt.subplots(1, 3, figsize=figsize(FIGURE_WIDTH, 4.4), sharey=True)
    drawn = False
    for ax, (scale_label, stem, steps) in zip(axes, SCALES, strict=True):
        traj = _trajectory(stem, steps, world)
        for t, _, _ in world.traits:
            x, y = traj[t]
            if len(x) < 2:
                continue
            drawn = True
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

    if not drawn:
        plt.close(fig)
        print(f"figure7: no data for world {world.key} — skipping")
        return

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
        for t, _, _ in world.traits
    ]
    fig.subplots_adjust(top=0.80, bottom=0.22, left=0.06, right=0.99, wspace=0.06)
    fig.suptitle(
        f"Parameter scaling — VEP AUPRC across training steps · {world.label}",
        fontsize=11,
        y=0.94,
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.10),
        ncol=len(handles),
        frameon=False,
        title="variant type",
        title_fontsize=9,
        fontsize=9,
    )
    save_figure(fig, OUTPUT_DIR, f"figure7_loss_vs_traitgym_curves__{world.key}")


def build_all() -> None:
    for world in WORLDS.values():
        build(world)


if __name__ == "__main__":
    build_all()
