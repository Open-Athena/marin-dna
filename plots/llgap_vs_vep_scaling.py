"""Issue #274 figure: loss-vs-VEP and LL-gap-vs-VEP across the scaling ladder.

Two side-by-side panels over the 8 `dna-bolinas-scaling-v0.5` sizes (46M→4B):
  left  — validation loss (x) vs Mendelian AUPRC (y), one series per variant type
  right — macro LL gap (x)   vs Mendelian AUPRC (y), one series per variant type
Spearman ρ per variant type is shown in each panel's legend, so the question
"does the gap track VEP better than loss?" is read off directly.

Self-contained recipe (repo convention): pulls loss + in-training VEP from W&B
and the LL gap from the evals_v2 `ll_gap` summary; if `--metrics-prefix` is
given and present, uses the official evals_v2 Mendelian AUPRC as the y-axis
instead of the in-training AUPRC. Reuses the loaders in
`scripts/issue274_scaling_correlation.py`. Emits SVG + PNG to
`plots/output/llgap_vs_vep_scaling/`.

Usage:
    uv run python plots/llgap_vs_vep_scaling.py \
        --gap-summary s3://oa-bolinas/snakemake/analysis/evals_v2/results/ll_gap/summary.parquet \
        --metrics-prefix s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

# Reuse the analysis loaders (single source of truth for schema/merge).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from issue274_scaling_correlation import (  # noqa: E402
    VARIANTS,
    load_evals_v2_vep,
    load_gap_by_size,
    pull_wandb_baseline,
)

OUT_DIR = Path(__file__).resolve().parent / "output" / "llgap_vs_vep_scaling"


def build_points(gap_summary: str, metrics_prefix: str | None, score_type: str):
    """Merged per-size frame + the AUPRC column prefix to plot (official if present)."""
    df = pull_wandb_baseline().merge(
        load_gap_by_size(gap_summary), on="size", validate="1:1"
    )
    prefix, label = "intrain_auprc_", "in-training AUPRC"
    if metrics_prefix:
        v2 = load_evals_v2_vep(metrics_prefix, score_type)
        if v2 is not None:
            df = df.merge(v2, on="size", validate="1:1")
            prefix, label = "v2_auprc_", f"evals_v2 AUPRC ({score_type})"
    return df, prefix, label


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gap-summary", required=True)
    ap.add_argument("--metrics-prefix", default=None)
    ap.add_argument("--score-type", default="minus_llr_avg")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.fonttype"] = "none"
    import matplotlib.pyplot as plt

    df, prefix, vep_label = build_points(
        args.gap_summary, args.metrics_prefix, args.score_type
    )

    predictors = [("eval_loss", "validation loss"), ("gap_macro", "macro LL gap")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    cmap = plt.get_cmap("tab10")

    for ax, (xcol, xlabel) in zip(axes, predictors):
        x = df[xcol].to_numpy(dtype=float)
        for i, v in enumerate(VARIANTS):
            ycol = f"{prefix}{v}"
            if ycol not in df.columns or df[ycol].isna().any():
                continue
            y = df[ycol].to_numpy(dtype=float)
            order = np.argsort(x)
            rho, _ = spearmanr(x, y)
            ax.plot(
                x[order],
                y[order],
                marker="o",
                ms=4,
                lw=1,
                color=cmap(i % 10),
                label=f"{v.replace('_variant', '')} (ρ={rho:+.2f})",
            )
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(vep_label)
    axes[0].set_title("loss vs VEP")
    axes[1].set_title("LL gap vs VEP")
    axes[1].legend(
        fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5), title="variant"
    )
    fig.suptitle(f"Scaling ladder (46M→4B, n=8): {vep_label} vs loss and LL gap (#274)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    svg = OUT_DIR / "figure.svg"
    fig.savefig(svg, bbox_inches="tight", dpi=200)
    fig.savefig(OUT_DIR / "figure.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {svg} (+ .png); VEP = {vep_label}")


if __name__ == "__main__":
    main()
