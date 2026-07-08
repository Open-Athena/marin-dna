"""Figure 9 — continued-pretraining mixture shift: macro-avg VEP AUPRC vs upstream
proportion, across the four metric-worlds.

Redo of the blog's Figure 9 on the offline evals_v2 eval. Seven
``uniform_to_upstream_*`` continuations (all warm-started from the 1·L uniform run)
span upstream-heavy (U90) down to no-upstream (C50/D50); each run's FINAL HF
checkpoint is scored offline and its macro-average AUPRC plotted against the
upstream proportion of its mix, with the uniform run's score as a dotted reference.
The ⅓-mix continuations (``uniform_to_upstream_3.7`` / ``3.6.2``) are omitted — they
just repeat the uniform mixture (matching Eric's ``UPSTREAM_SWEEP``).

One parallel figure per world (M·LLR / M·Probe / S·LLR / S·Probe); the SGE worlds
plot the SGE across-gene macro instead of the Mendelian one. Renders whatever worlds
are scored (skips a world gracefully if its finals aren't on S3 yet).

Run:  uv run python -m plots.blog.figure9_upstream_mix_auprc
Out:  plots/output/blog/figure9_upstream_mix_auprc__{world.key}.{svg,png,pdf}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from marin_dna.pipelines.evals.metrics import MACRO_AVG_SUBSET
from plots.blog import _mixture as mx
from plots.blog import _mixture_lineage as ml
from plots.blog._style.figure_style import (
    FIGURE_WIDTH,
    LARGEST_MODEL_COLOR,
    X_LABEL_PAD,
    figsize,
)
from plots.blog._style.savefig import save_figure
from plots.blog._worlds import WORLDS

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "blog"

# The uniform→upstream sweep (⅓-mix 3.7 / 3.6.2 omitted — repeat the uniform mix).
UPSTREAM_SWEEP: tuple[str, ...] = (
    "uniform_to_upstream_1",  # U90
    "uniform_to_upstream_2",  # U80
    "uniform_to_upstream_3",  # U60
    "uniform_to_upstream_3.5",  # U50
    "uniform_to_upstream_3.6",  # U40
    "uniform_to_upstream_4",  # U30
    "uniform_to_upstream_5",  # U0
)
BASELINE = "uniform"


def _macro_auprc(world, mix: str) -> float | None:
    """A run's macro-average AUPRC in one world, from its FINAL HF checkpoint.

    Prefers the pipeline's ``_macro_avg_`` row (present in both the Mendelian and SGE
    metrics/probe outputs); falls back to the mean over the world's trajectory
    subsets. Returns ``None`` if the checkpoint isn't scored in this world yet."""
    try:
        df = world.read(mx.final_name(mix)).to_pandas()
    except (LookupError, FileNotFoundError, OSError):
        return None
    macro = df[df["subset"] == MACRO_AVG_SUBSET]["value"]
    if len(macro) and pd.notna(macro.iloc[0]):
        return float(macro.iloc[0])
    subsets = [s for s, _, _ in world.traits]
    vals = df[df["subset"].isin(subsets)]["value"].dropna()
    return float(vals.mean()) if len(vals) else None


def build(world) -> None:
    score = {mix: _macro_auprc(world, mix) for mix in (*UPSTREAM_SWEEP, BASELINE)}
    pts = sorted(
        (ml.BY_MIX[mix].weights.get("upstream", 0.0), score[mix])
        for mix in UPSTREAM_SWEEP
        if score[mix] is not None
    )
    if len(pts) < 2 or score[BASELINE] is None:
        print(f"figure9: world {world.key} not scored yet ({len(pts)} pts) — skipping")
        return
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    fig, ax = plt.subplots(figsize=figsize(FIGURE_WIDTH, 4.0))
    ax.axhline(score[BASELINE], color="0.4", lw=1.0, ls=":", zorder=1)
    ax.text(
        max(xs),
        score[BASELINE],
        "uniform baseline  ",
        ha="right",
        va="bottom",
        fontsize=10,
        color="0.4",
    )
    ax.plot(
        xs,
        ys,
        color=LARGEST_MODEL_COLOR,
        lw=1.8,
        marker="o",
        markersize=8,
        markerfacecolor=LARGEST_MODEL_COLOR,
        markeredgecolor=LARGEST_MODEL_COLOR,
        markeredgewidth=0.5,
        zorder=3,
    )
    ax.set_xlabel("upstream proportion in continuation mix", labelpad=X_LABEL_PAD)
    ax.set_ylabel("macro avg VEP AUPRC")
    ax.set_title(
        f"Continued pretraining from uniform mixture · {world.label}", fontsize=11
    )
    ax.grid(False)

    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR, f"figure9_upstream_mix_auprc__{world.key}")
    print(f"figure9: {world.key} rendered ({len(pts)} upstream points)")


def build_all() -> None:
    for key in ("mendelian_llr", "mendelian_probe", "sge_llr", "sge_probe"):
        build(WORLDS[key])


if __name__ == "__main__":
    build_all()
