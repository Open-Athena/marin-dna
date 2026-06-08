"""LL trajectories, order vs family — exp255 matched comparison (issue #255).

The `val_*` functional / non-functional losses are logged every eval step on
wandb, so we compare the full **trajectories** of the two cohorts rather than a
single final-step point — the trajectory spread is the statistical context the
one-run endpoint lacked (there is no per-step SE; the runs are single-seed).

Matched region per arm (per the exp232 region->subset map):
  cds  arm: val_cds        (order dna-exp255 v4_cds_order  vs family dna-exp232 v4_cds)
  ccre arm: val_enhancer   (order ...v4_ccre_non_promoter_order vs family ...v4_ccre_non_promoter)

LL = -loss (higher = better). LL gap = LL_func - LL_nonfunc = nonfunc_loss -
func_loss (>0 = higher likelihood on functional/constrained positions). Order
arms are stitched across their relaunch runs (concat wandb history, dedupe step).

Finding: the cds trajectories overlap throughout (no cohort difference); the
enhancer LL gap shows a small, monotone late-training edge for family (functional
-LL-driven). The <=step-500 point is an early-training transient.

Output (PNG 130dpi + SVG) under plots/output/exp255_ll_trajectory/:
  exp255_ll_trajectory.{png,svg}   2x2: {LL gap, LL functional} x {val_cds, val_enhancer}

Run:  uv run python plots/exp255_ll_trajectory.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import wandb
from _exp255_style import COL_FAMILY as C_FAMILY
from _exp255_style import COL_ORDER as C_ORDER
from _exp255_style import set_style

GROUP_ORDER = "dna-exp255-v0.1"
GROUP_FAMILY = "dna-exp232-v0.1"
# arm -> (val recipe, order-run substring, family-run substring)
ARMS = {
    "cds": ("val_cds", "v4_cds_order", "v4_cds"),
    "utr3": ("val_utr3", "v4_utr3_order", "v4_utr3"),
    "ncrna": ("val_ncrna", "v4_ncrna_exon_order", "v4_ncrna_exon"),
    "tss": ("val_tss_pc", "v4_tss_region_and_utr5_order", "v4_tss_region_and_utr5"),
    "ccre": ("val_enhancer", "v4_ccre_non_promoter_order", "v4_ccre_non_promoter"),
}
MIN_STEP = (
    1000  # drop the <=500 early-training transient (order ccre just-restarted there)
)
OUT_DIR = Path(__file__).parent / "output" / Path(__file__).stem


def stitch(api: wandb.Api, group: str, sub: str, recipe: str) -> pd.DataFrame:
    """Concat every run in `group` whose name contains `sub` (order arms are split
    across relaunch runs), keep the matched-region func/nonfunc loss, dedupe by step."""
    fk, nk = f"eval/{recipe}_functional/loss", f"eval/{recipe}_nonfunctional/loss"
    frames = []
    for r in api.runs("marin", filters={"group": group}):
        if sub not in r.name:
            continue
        h = r.history(keys=[fk, nk], samples=10000, pandas=True)
        if len(h):
            frames.append(h)
    df = (
        pd.concat(frames)
        .dropna(subset=[fk, nk])
        .sort_values("_step")
        .drop_duplicates(subset=["_step"], keep="last")
    )
    df["func_ll"] = -df[fk]
    df["gap"] = df[nk] - df[fk]
    return df


def main() -> None:
    set_style()
    api = wandb.Api()
    data = {
        (arm, coh): stitch(api, grp, sub, rec)
        for arm, (rec, so, sf) in ARMS.items()
        for coh, sub, grp in (("order", so, GROUP_ORDER), ("family", sf, GROUP_FAMILY))
    }

    rows = [("gap", "LL gap (nats)"), ("func_ll", "LL functional (nats, = −loss)")]
    arms = list(ARMS)
    fig, axes = plt.subplots(
        len(rows), len(arms), figsize=(3.0 * len(arms), 7.0), squeeze=False
    )
    for ri, (key, ylab) in enumerate(rows):
        for ci, arm in enumerate(arms):
            ax = axes[ri][ci]
            for coh, color in (("family", C_FAMILY), ("order", C_ORDER)):
                d = data[(arm, coh)]
                d = d[d["_step"] >= MIN_STEP]
                ax.plot(
                    d["_step"],
                    d[key],
                    marker="o",
                    ms=4,
                    lw=1.8,
                    color=color,
                    label=f"{coh} ({'108' if coh == 'family' else '19'} sp.)",
                )
            if ri == 0:
                ax.set_title(f"{arm}  ({ARMS[arm][0]})", fontsize=11)
            if ci == 0:
                ax.set_ylabel(ylab, fontsize=10)
            ax.set_xlabel("training step", fontsize=9)
            ax.grid(alpha=0.3)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(
        h, lab, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=2, fontsize=11
    )
    fig.suptitle(
        "exp255 (#255) — matched-region LL trajectories: family (baseline) vs order\n"
        "LL = −loss (higher better); single run per arm (trajectory spread = the uncertainty cue); from step 1000 (≤500 early transient excluded)",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("png", dict(dpi=130)), ("svg", {})):
        fig.savefig(OUT_DIR / f"exp255_ll_trajectory.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"wrote exp255_ll_trajectory.png + .svg in {OUT_DIR}/")


if __name__ == "__main__":
    main()
