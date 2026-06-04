"""Plot QTL Pearson vs training step for the #259 sweep arms (one line per arm).

Pulls each W&B run's caqtl/dsqtl/avg QTL curve, de-duplicates the per-step
carry-forward (pre-fix runs re-log the held value at every flush; #259), and
plots avg + caqtl + dsqtl vs the global step. The *trajectory* (false plateaus,
where arms diverge, the WSD decay-tail bump) is the message — not just the final
value. Emits PNG (inline + local sanity-check) and SVG (GitHub embed).

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


def fetch_curve(run):
    """(step, caqtl, dsqtl, avg) at each distinct eval, de-duping carry-forward."""
    df = run.history(samples=20000, pandas=True)
    g = next(c for c in df.columns if "global_step" in c)
    cols = [g] + [m for m in METRICS if m in df.columns]
    sub = df[cols].dropna(subset=["qtl_avg_pearson"]).sort_values(g)
    # pre-fix runs re-log the held value every flush -> keep only rows where the
    # avg changes (the actual sparse evals). A no-op for the fixed clean runs.
    sub = sub[sub["qtl_avg_pearson"].ne(sub["qtl_avg_pearson"].shift())]
    return sub.rename(columns={g: "step"})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arms", nargs="+", help="label=wandb_run_name pairs")
    ap.add_argument("--project", default="gonzalobenegas/chrombpnet-eval")
    ap.add_argument("--out-prefix", default=".sweep_plots/sweep_qtl")
    args = ap.parse_args()

    api = wandb.Api()
    curves: dict[str, object] = {}
    for arm in args.arms:
        label, name = arm.split("=", 1)
        runs = api.runs(
            args.project, filters={"display_name": name}, order="-created_at"
        )
        if not runs:
            print(f"WARN: no run named {name!r}")
            continue
        run = sorted(runs, key=lambda r: r.created_at)[-1]
        c = fetch_curve(run)
        curves[label] = c
        last = c.iloc[-1]
        print(
            f"{label:14s} {len(c):2d} pts  step≤{int(last['step']):>5}  "
            f"caqtl={last['qtl_caqtl_pearson']:.4f} dsqtl={last['qtl_dsqtl_pearson']:.4f} "
            f"avg={last['qtl_avg_pearson']:.4f}"
        )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharex=True)
    for ax, metric, title in zip(axes, METRICS, TITLES):
        for label, c in curves.items():
            if metric in c:
                ax.plot(c["step"], c[metric], marker="o", ms=3, label=label)
        ax.set_title(title)
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("QTL Pearson (dev / train split)")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("#259 sweep — QTL Pearson vs step (all-chroms WSD, N=12k, single seed)")
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    for ext in ("png", "svg"):
        path = f"{args.out_prefix}.{ext}"
        fig.savefig(path, dpi=130, bbox_inches="tight")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
