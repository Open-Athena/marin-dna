"""Figure 9: continued-pretraining mixture shift — macro avg VEP AUPRC vs upstream proportion.

Seven `uniform_to_upstream_*` continuations (all warm-started from the 1·L
uniform run) trained mixes from upstream-heavy (U90) down to no-upstream
(C50/D50). We plot each run's final macro-average 6-task VEP AUPRC against the
upstream proportion of its mix, with the 1·L uniform run's score as a dotted
reference line. The ⅓-mix continuations (uniform_to_upstream_3.7 / 1.6·M and
uniform_to_uniform_1 / 1.7·L) are omitted — they just repeat the uniform mixture.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from figures import mixture_lineage as ml
from figures.data import save
from utils.figure_style import FIGURE_WIDTH, LARGEST_MODEL_COLOR, X_LABEL_PAD, figsize

# The uniform→upstream sweep (the ⅓-mix 1.6·M and 1.7·L are omitted).
UPSTREAM_SWEEP = (
    "uniform_to_upstream_1",  # 1.1  U90
    "uniform_to_upstream_2",  # 1.2  U80
    "uniform_to_upstream_3",  # 1.3  U60
    "uniform_to_upstream_3.5",  # 1.4  U50
    "uniform_to_upstream_3.6",  # 1.5  U40
    "uniform_to_upstream_4",  # 1.8  U30
    "uniform_to_upstream_5",  # 1.9  U0
)
BASELINE = "uniform"


def build(df: pd.DataFrame) -> None:
    score = {row["mix"]: ml.composite_score(row) for _, row in df.iterrows()}
    pts = sorted(
        (ml.BY_MIX[mix].weights.get("upstream", 0.0), score[mix])
        for mix in UPSTREAM_SWEEP
    )
    xs = [p[0] for p in pts]
    ys = [p[1] * 100.0 for p in pts]

    # Author at full FIGURE_WIDTH like every other figure so the post (which
    # displays all figures at one column width) doesn't upscale this one and
    # blow up its label text. Height is kept short for a single trend line —
    # a wide, low aspect that doesn't look oversized next to the other figures.
    fig, ax = plt.subplots(figsize=figsize(FIGURE_WIDTH, 4.0))
    baseline = score[BASELINE] * 100.0
    ax.axhline(baseline, color="0.4", lw=1.0, ls=":", zorder=1)
    ax.text(
        max(xs),
        baseline,
        "baseline (step=0)  ",
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
    ax.set_ylabel("macro avg VEP AUPRC (%)")
    ax.set_title("Continued pretraining from uniform mixture", fontsize=11)
    ax.grid(False)

    fig.tight_layout()
    save(fig, "figure9_upstream_mix_auprc")
