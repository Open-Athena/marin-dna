# Copyright The MarinDNA Authors
# SPDX-License-Identifier: Apache-2.0

"""exp284 epoch-reshuffle (ON) vs exp255 no-reshuffle (OFF) — TSS-region 0.25B.

Single-panel figures of the in-training (online, post-#266 → offline-faithful)
TSS diagonal VEP trajectories for the slice-mix reshuffle run (exp284) vs the
matched no-reshuffle baseline (exp255's ``v4_tss_region_and_utr5_order`` arm,
#256), vs training epoch. Both ran 5000 x 8192 on the same dataset/eval; the only
difference is the per-epoch slice-mix reshuffle (PR #285). Issue #284.

Writes to output/exp284_reshuffle_analysis/:
  * ll_gap.{png,svg}              val_tss_pc LL gap (nonfunc - func loss)
  * auprc_5utr.{png,svg}          mendelian 5'UTR AUPRC
  * auprc_tss_proximal.{png,svg}  mendelian tss_proximal AUPRC

Each run is plotted at its own logged eval steps (preemptions dropped the step-4000
eval for both, and OFF's final ~5000 eval).

Usage:
  uv run python plots/exp284_reshuffle_analysis.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import wandb

ON = ("dna-exp284-v0.1", "tss_region_and_utr5_order")  # reshuffle ON
OFF = ("dna-exp255-v0.1", "tss_region_and_utr5_order")  # no-reshuffle baseline (#256)

EP_ROWS = 1_965_838  # post-RC rows; epoch = step * 8192 / rows
BATCH = 8192
MIN_STEP = 500

REC = "val_tss_pc"
FK, NK = f"eval/{REC}_functional/loss", f"eval/{REC}_nonfunctional/loss"
A5 = "lm_eval/mendelian_traits_255/5_prime_UTR_variant/avg/auprc"
ATP = "lm_eval/mendelian_traits_255/tss_proximal/avg/auprc"
KEYS = [FK, NK, A5, ATP]

OUT = Path(__file__).parent / "output" / "exp284_reshuffle_analysis"
C_ON, C_OFF = "#1f6feb", "#999999"  # ON solid blue, OFF dashed gray


def set_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 11,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#5a5a5a",
            "axes.axisbelow": True,
            "axes.grid": True,
            "grid.color": "#e6e6e6",
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def traj(api: wandb.Api, group: str, sub: str) -> pd.DataFrame:
    """Stitched per-eval-step trajectory: columns _step, epoch, gap, a5, atp."""
    frames = [
        r.history(keys=KEYS, samples=10000, pandas=True)
        for r in api.runs("marin", filters={"group": group})
        if sub in r.name
    ]
    frames = [f for f in frames if len(f)]
    if not frames:
        raise SystemExit(f"no wandb runs for group={group!r} sub={sub!r}")
    df = pd.concat(frames).sort_values("_step").drop_duplicates("_step", keep="last")
    df = df[df["_step"] >= MIN_STEP].copy()
    df["epoch"] = df["_step"] * BATCH / EP_ROWS
    df["gap"] = df[NK] - df[FK]
    return df.rename(columns={A5: "a5", ATP: "atp"}).reset_index(drop=True)


def _save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)


def _abs_panel(
    on: pd.DataFrame, off: pd.DataFrame, col: str, title: str, ylab: str, name: str
) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for df, c, ls, lab in [
        (on, C_ON, "-", "reshuffle ON (exp284)"),
        (off, C_OFF, "--", "no-reshuffle OFF (exp255 #256)"),
    ]:
        d = df.dropna(subset=[col])
        ax.plot(d["epoch"], d[col], color=c, ls=ls, lw=2, marker="o", ms=5, label=lab)
    ax.set_title(title)
    ax.set_xlabel("training epochs")
    ax.set_ylabel(ylab)
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, name)


def main() -> None:
    set_style()
    api = wandb.Api()
    on, off = traj(api, *ON), traj(api, *OFF)
    _abs_panel(
        on, off, "gap", "val_tss_pc LL gap (nonfunc − func)", "LL gap (nats)", "ll_gap"
    )
    _abs_panel(on, off, "a5", "mendelian 5′UTR AUPRC", "AUPRC", "auprc_5utr")
    _abs_panel(
        on, off, "atp", "mendelian tss_proximal AUPRC", "AUPRC", "auprc_tss_proximal"
    )
    print(f"wrote 3 figures to {OUT}  | ON {len(on)} pts, OFF {len(off)} pts")


if __name__ == "__main__":
    main()
