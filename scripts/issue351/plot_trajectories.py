"""exp351 — enhancer-centered vs tiled: metric-vs-training-step trajectory plots.

Pulls per-step wandb history for the two exp351 arms and plots three trajectories
side by side (centered vs tiled overlaid):
  - distal mendelian AUPRC   lm_eval/mendelian_traits_255/distal/avg/auprc
  - val_enhancer LL-gap      LL_functional - LL_nonfunctional        (LL = -loss)
  - val_enhancer LL (func)   -eval/val_enhancer_functional/loss

Emits figure.svg (GitHub-embed artifact) + figure.png (local sanity-check) to OUTDIR.
The figures are hosted in the shared gist and embedded in issue #351 by pinned raw URL.

Run: uv run --no-sync python scripts/issue351/plot_trajectories.py [OUTDIR]
Requires a wandb API key (via ~/.netrc `api.wandb.ai` entry).
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb

RUNS = {
    "centered": "gonzalobenegas/marin/dna-exp351-zoonomia-v1-0p25b-centered-v0.1-8adcec",
    "tiled": "gonzalobenegas/marin/dna-exp351-zoonomia-v1-0p25b-tiled-v0.1-812cad",
}
DISTAL = "lm_eval/mendelian_traits_255/distal/avg/auprc"
LOSS_F = "eval/val_enhancer_functional/loss"
LOSS_NF = "eval/val_enhancer_nonfunctional/loss"

PANELS = ["distal AUPRC", "LL-gap", "LL_functional"]
PANEL_TITLE = {
    "distal AUPRC": "distal mendelian AUPRC",
    "LL-gap": "val_enhancer LL-gap",
    "LL_functional": "val_enhancer LL (functional)",
}
PANEL_YLABEL = {"distal AUPRC": "AUPRC", "LL-gap": "LL-gap", "LL_functional": "LL"}
PALETTE = {"tiled": "#4C72B0", "centered": "#DD8452"}  # tiled=control blue, centered=orange


def pull_history() -> pd.DataFrame:
    """Long-form (arm, metric, step, value) over the eval cadence, LL = -loss."""
    api = wandb.Api()
    frames = []
    for arm, path in RUNS.items():
        run = api.run(path)
        h = run.history(keys=[DISTAL, LOSS_F, LOSS_NF], samples=100000, pandas=True)
        d = h.dropna(subset=[DISTAL])[["_step", DISTAL]].rename(columns={DISTAL: "value"})
        d["metric"] = "distal AUPRC"
        v = h.dropna(subset=[LOSS_F, LOSS_NF]).copy()
        parts = [d]
        for name, series in [
            ("LL_functional", -v[LOSS_F]),
            ("LL_nonfunctional", -v[LOSS_NF]),
            ("LL-gap", (-v[LOSS_F]) - (-v[LOSS_NF])),
        ]:
            p = v[["_step"]].copy()
            p["value"] = series.values
            p["metric"] = name
            parts.append(p)
        part = pd.concat(parts, ignore_index=True)
        part["arm"] = arm
        frames.append(part)
    long = pd.concat(frames, ignore_index=True)[["arm", "metric", "_step", "value"]]
    return long.rename(columns={"_step": "step"})


def plot(long: pd.DataFrame, outdir: str) -> None:
    df = long[long["metric"].isin(PANELS)].copy()
    sns.set_theme(style="whitegrid", context="talk")
    g = sns.relplot(
        data=df, x="step", y="value", hue="arm", hue_order=["tiled", "centered"],
        col="metric", col_order=PANELS, kind="line", marker="o", markersize=8,
        linewidth=2.4, palette=PALETTE,
        facet_kws={"sharey": False, "sharex": True}, height=4.2, aspect=1.05,
    )
    g.set_titles("")
    g.set_axis_labels("training step", "")
    g.legend.set_title("arm")
    for metric, ax in g.axes_dict.items():
        ax.set_title(PANEL_TITLE[metric], fontsize=15)
        ax.set_ylabel(PANEL_YLABEL[metric], fontsize=13)
        ax.set_xlim(0, 5200)
        ax.set_xticks([0, 2000, 4000])
        sub = df[(df["metric"] == metric) & (df["arm"] == "centered")].sort_values("step")
        ax.annotate(
            f"{sub['value'].iloc[-1]:.3f}", xy=(sub["step"].iloc[-1], sub["value"].iloc[-1]),
            xytext=(-6, 6), textcoords="offset points", ha="right", fontsize=11,
            color=PALETTE["centered"], fontweight="bold",
        )
    g.figure.suptitle(
        "exp351 — enhancer-centered vs tiled, metric trajectories (0.25B, 5K×8192)",
        y=1.03, fontsize=16,
    )
    g.figure.text(
        0.5, -0.04,
        "Both arms trained to step 4999. Eval cadence irregular (preemption/resume). "
        "Centered ran ~9.7 epochs vs tiled ~3.7 at fixed compute — an epoch confound on the gaps.",
        ha="center", fontsize=9, color="0.35", wrap=True,
    )
    os.makedirs(outdir, exist_ok=True)
    for ext in ("svg", "png"):
        g.figure.savefig(os.path.join(outdir, f"exp351_trajectories.{ext}"),
                         dpi=150, bbox_inches="tight")
    print(f"wrote exp351_trajectories.{{svg,png}} to {outdir}")


if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else "scratch/exp351"
    plot(pull_history(), outdir)
