"""Appendix: top-down mixture-continuation tree.

Each node is a training run; edges link a continuation to the parent checkpoint
it warm-started from (dashed = branched before the parent's cooldown). Node fill
encodes the composite 6-task VEP AUPRC; the label shows the tree-path id,
mixture, score, and own/cumulative token counts. Siblings sit side-by-side so
mixtures explored at the same level are directly comparable. Lineage, weights,
and token accounting come from `figures.mixture_lineage`; metrics come from the
committed `data/data_mixture_results.csv`.

Usage:
  uv run src/figures/appendix/mixture_tree.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from figures import mixture_lineage as ml  # noqa: E402
from figures.data import load_mixture  # noqa: E402
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

NODE_W = 0.88  # node width  (x = leaf-index units)
NODE_H = 0.62  # node height (y = level units)
LEVEL_GAP = 1.0  # vertical spacing between tree levels
TREE_GAP = 0.4  # extra x gap between separate trees


def _luminance(rgba) -> float:
    r, g, b = rgba[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _own_trained_tokens(row) -> float:
    """New-portion tokens actually trained: the `tokens` tag for a finished run,
    else the tag scaled by the fraction of the OWN portion completed.

    The own portion runs from `num_train_steps - own_steps` (the warm-start /
    branch step) to `num_train_steps`, where own_steps = tokens / (batch×seq).
    A root resets, so its branch step is 0 and this reduces to plain run_progress.
    """
    tokens = float(row["tokens"])
    if row["state"] == "finished":
        return tokens
    tps = float(row["batch_size"]) * float(row["train_seq_len"])
    own_steps = tokens / tps
    total_steps = float(row["num_train_steps"])
    last_step = float(row["run_progress"]) * total_steps
    branch_step = total_steps - own_steps
    own_done = (last_step - branch_step) / own_steps if own_steps else 0.0
    return tokens * min(max(own_done, 0.0), 1.0)


def plot(df) -> None:
    present = set(df["mix"])
    extra = present - set(ml.BY_MIX)
    if extra:
        print(
            f"  WARNING: {len(extra)} finished runs absent from LINEAGE (skipped): {sorted(extra)}"
        )
    runs = [r for r in ml.LINEAGE if r.mix in present]

    # Own new-portion tokens from the tag, scaled by the fraction of the OWN
    # portion actually trained for any run that didn't finish, so the count
    # reflects tokens trained rather than the configured budget. For a continuation
    # run the W&B counter (and thus num_train_steps) spans inherited+own steps, so
    # we measure progress against the own portion only — not the absolute span,
    # which would overstate the new tokens of an early-crashed continuation.
    own = {row["mix"]: _own_trained_tokens(row) for _, row in df.iterrows()}
    # A run still in flight may not have logged any eval yet (composite is NaN).
    # Keep it in the tree as an "in progress" node, but exclude it from the color
    # scale and render it with a neutral fill.
    score = {
        row["mix"]: ml.composite_score(row)
        for _, row in df.iterrows()
        if row["mix"] in present
    }
    finite_score = {m: s for m, s in score.items() if s == s}  # drop NaN

    # Tidy top-down layout: leaves take sequential x; parents center over children.
    pos: dict[str, tuple[float, float]] = {}
    depth: dict[str, int] = {}
    x_cursor = [0.0]

    def place(run: ml.Run, d: int) -> None:
        depth[run.mix] = d
        kids = [k for k in ml.children_of(run.mix) if k.mix in present]
        if not kids:
            x = x_cursor[0]
            x_cursor[0] += 1.0
        else:
            for k in kids:
                place(k, d + 1)
            x = sum(pos[k.mix][0] for k in kids) / len(kids)
        pos[run.mix] = (x, -d * LEVEL_GAP)

    for root in (r for r in runs if r.parent is None):
        place(root, 0)
        x_cursor[0] += TREE_GAP

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    max_depth = max(depth.values())

    fig_w = max(9.0, (max(xs) - min(xs) + 1.4) * 0.80)
    fig_h = (max_depth + 1) * 1.7 + 1.3
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    norm = Normalize(min(finite_score.values()), max(finite_score.values()))
    cmap = plt.get_cmap("viridis")
    IN_PROGRESS_FILL = to_rgba("#d9d2c4")  # neutral tan for nodes without a score yet

    # Edges: parent bottom -> child top. Dashed when branched pre-cooldown.
    for run in runs:
        if run.parent not in pos:
            continue
        x0, y0 = pos[run.parent]
        x1, y1 = pos[run.mix]
        ax.plot(
            [x0, x1],
            [y0 - NODE_H / 2, y1 + NODE_H / 2],
            color="0.55",
            linewidth=0.9,
            zorder=1,
            linestyle=(0, (3, 2)) if run.branch == "pre_cooldown" else "-",
        )

    # Nodes.
    for run in runs:
        x, y = pos[run.mix]
        in_progress = run.mix not in finite_score
        color = IN_PROGRESS_FILL if in_progress else cmap(norm(score[run.mix]))
        ax.add_patch(
            FancyBboxPatch(
                (x - NODE_W / 2, y - NODE_H / 2),
                NODE_W,
                NODE_H,
                boxstyle="round,pad=0.01,rounding_size=0.06",
                facecolor=color,
                edgecolor="0.2",
                linewidth=0.5,
                zorder=2,
            )
        )
        score_line = "in progress" if in_progress else f"{score[run.mix]:.3f}"
        label = (
            f"{run.name}\n{ml.format_mixture(run.weights)}\n"
            f"{score_line}\n"
            f"{ml.format_tokens(own[run.mix])}/{ml.format_tokens(ml.cumulative_total(run.mix, own))}"
        )
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            linespacing=1.3,
            fontsize=5.8,
            color="white" if _luminance(color) < 0.5 else "black",
            zorder=3,
        )

    ax.set_xlim(min(xs) - 0.7, max(xs) + 0.7)
    ax.set_ylim(min(ys) - NODE_H, NODE_H + 0.15)
    ax.axis("off")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("composite VEP AUPRC (6-task mean)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    ax.legend(
        handles=[
            Line2D(
                [0], [0], color="0.55", linestyle="-", label="from final checkpoint"
            ),
            Line2D(
                [0],
                [0],
                color="0.55",
                linestyle=(0, (3, 2)),
                label="from pre-cooldown checkpoint",
            ),
        ],
        loc="lower right",
        fontsize=8,
        frameon=False,
    )

    fig.suptitle(
        "Mixture continuation tree — composite VEP AUPRC by run", fontsize=12, y=0.98
    )
    fig.text(
        0.01,
        0.01,
        "node: id · mixture · score · own/cumulative tokens    "
        "size tier: S≈5B  M≈18B  L≈62B new tokens",
        fontsize=7.5,
        color="0.35",
        ha="left",
        va="bottom",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    save_figure(fig, FIGURES_DIR, "mixture_tree")


def main() -> None:
    plot(load_mixture())


if __name__ == "__main__":
    main()
