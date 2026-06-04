"""Plot QTL Pearson + grad_norm vs training step for the #259 sweep arms.

Pulls each W&B run's caqtl/dsqtl/avg QTL curve (de-duplicating the per-step
carry-forward; #259) and its grad_norm, and plots avg + caqtl + dsqtl + grad_norm
vs the global step, one line per arm. The *trajectory* (false plateaus, where arms
diverge, the WSD decay-tail bump, gradient spikes/stability) is the message — not
just the final value. Emits PNG (inline + local sanity-check) and SVG (GitHub).

  WANDB_API_KEY=... uv run --with matplotlib --extra chrombpnet python \
    scripts/chrombpnet_eval/plot_sweep_qtl.py \
    baseline=dna-exp259-sweep-baseline no-bias=dna-exp259-sweep-nobias \
    count-only=dna-exp259-sweep-countonly both=dna-exp259-sweep-both \
    adamw=dna-exp259-sweep-adamw-wd1e-2 --out-prefix .sweep_plots/sweep_qtl
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

METRICS = ["qtl_avg_pearson", "qtl_caqtl_pearson", "qtl_dsqtl_pearson"]
TITLES = ["avg (caqtl+dsqtl)/2", "caqtl (n=3173)", "dsqtl (n=309)"]


def fetch(run):
    """(qtl_curve, grad_norm_curve) for one run, indexed by global step.

    qtl is de-duped to the distinct (sparse) evals — pre-fix runs re-log the held
    value at every flush (#259); a no-op for the fixed clean runs. grad_norm is
    left dense.
    """
    df = run.history(samples=40000, pandas=True)
    g = next(c for c in df.columns if "global_step" in c)
    df = df.rename(columns={g: "step"})
    qtl = (
        df[["step", *[m for m in METRICS if m in df.columns]]]
        .dropna(subset=["qtl_avg_pearson"])
        .sort_values("step")
    )
    qtl = qtl[qtl["qtl_avg_pearson"].ne(qtl["qtl_avg_pearson"].shift())]
    gn = (
        df[["step", "grad_norm"]].dropna().sort_values("step")
        if "grad_norm" in df.columns
        else None
    )
    return qtl, gn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arms", nargs="+", help="label=wandb_run_name pairs")
    ap.add_argument("--project", default="gonzalobenegas/chrombpnet-eval")
    ap.add_argument("--out-prefix", default=".sweep_plots/sweep_qtl")
    args = ap.parse_args()

    api = wandb.Api()
    curves: dict[str, tuple] = {}
    for arm in args.arms:
        label, name = arm.split("=", 1)
        runs = api.runs(
            args.project, filters={"display_name": name}, order="-created_at"
        )
        if not runs:
            print(f"WARN: no run named {name!r}")
            continue
        run = sorted(runs, key=lambda r: r.created_at)[-1]
        qtl, gn = fetch(run)
        curves[label] = (qtl, gn)
        last = qtl.iloc[-1]
        gnmax = f"{gn['grad_norm'].max():.0f}" if gn is not None else "?"
        print(
            f"{label:14s} {len(qtl):2d} pts  step≤{int(last['step']):>5}  "
            f"caqtl={last['qtl_caqtl_pearson']:.4f} dsqtl={last['qtl_dsqtl_pearson']:.4f} "
            f"avg={last['qtl_avg_pearson']:.4f}  grad_norm_max={gnmax}"
        )

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    for ax, metric, title in zip(axes[:3], METRICS, TITLES):
        for label, (qtl, _) in curves.items():
            if metric in qtl:
                ax.plot(qtl["step"], qtl[metric], marker="o", ms=3, label=label)
        ax.set_title(title)
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("QTL Pearson (dev / train split)")
    axes[0].legend(fontsize=8, loc="lower right")
    gax = axes[3]
    for label, (_, gn) in curves.items():
        if gn is not None:
            gax.plot(gn["step"], gn["grad_norm"], lw=0.8, alpha=0.8, label=label)
    gax.set_yscale("log")
    gax.set_title("grad_norm (pre-clip)")
    gax.set_xlabel("training step")
    gax.grid(alpha=0.3, which="both")
    fig.suptitle("#259 sweep — QTL Pearson + grad_norm vs step (all-chroms WSD, N=12k)")
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    for ext in ("png", "svg"):
        path = f"{args.out_prefix}.{ext}"
        fig.savefig(path, dpi=130, bbox_inches="tight")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
