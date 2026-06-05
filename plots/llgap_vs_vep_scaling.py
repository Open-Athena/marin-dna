"""Issue #274 figure: overall-LL-vs-VEP and LL-gap-vs-VEP across the scaling ladder.

Two panels over the 8 `dna-bolinas-scaling-v0.5` sizes (46M→4B), for one
validation-interval region (default CDS — the weak-loss-correlation regime):
  left  — overall LL (x) vs Mendelian AUPRC (y), one series per variant type
  right — LL gap (x)     vs Mendelian AUPRC (y), one series per variant type
Pearson r per variant is shown in each legend (Pearson separates the predictors
that Spearman ties; see scripts/issue274_scaling_correlation.py). The W&B
validation loss is intentionally NOT shown — it is `lowercase_weight=0.01`
(functional-dominated), so the clean equal-weight overall LL is the baseline.

Self-contained recipe; reuses the loaders in
`scripts/issue274_scaling_correlation.py`. Uses the official evals_v2 AUPRC when
`--metrics-prefix` is given and present, else the in-training AUPRC. Emits
SVG + PNG to `plots/output/llgap_vs_vep_scaling/`.

Usage:
    uv run python plots/llgap_vs_vep_scaling.py \
        --gap-summary s3://oa-bolinas/snakemake/analysis/evals_v2/results/ll_gap/summary.parquet \
        --metrics-prefix s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics --region cds
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from issue274_scaling_correlation import (  # noqa: E402
    VARIANTS,
    load_evals_v2_vep,
    load_gap_quantities,
    pull_wandb_auprc,
)

OUT_DIR = Path(__file__).resolve().parent / "output" / "llgap_vs_vep_scaling"


def build_points(gap_summary: str, metrics_prefix: str | None, score_type: str):
    df = pull_wandb_auprc().merge(
        load_gap_quantities(gap_summary), on="size", validate="1:1"
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
    ap.add_argument(
        "--region", default="cds", choices=["cds", "upstream", "downstream", "macro"]
    )
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.fonttype"] = "none"
    import matplotlib.pyplot as plt

    df, prefix, vep_label = build_points(
        args.gap_summary, args.metrics_prefix, args.score_type
    )
    r = args.region
    predictors = [(f"LL_all_{r}", "overall LL"), (f"gap_{r}", "LL gap")]

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
            ax.plot(
                x[order],
                y[order],
                marker="o",
                ms=4,
                lw=1,
                color=cmap(i % 10),
                label=v.replace("_variant", ""),
            )
        ax.set_xlabel(f"{xlabel} ({r})")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(vep_label)
    axes[0].set_title(f"overall LL vs VEP ({r})")
    axes[1].set_title(f"LL gap vs VEP ({r})")
    axes[1].legend(
        fontsize=7,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        title="variant",
    )
    fig.suptitle(
        f"Scaling ladder (46M→4B, n=8): {vep_label} vs overall LL and LL gap — {r} (#274)"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    svg = OUT_DIR / f"figure_{r}.svg"
    fig.savefig(svg, bbox_inches="tight", dpi=200)
    fig.savefig(svg.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {svg} (+ .png); VEP = {vep_label}, region = {r}")


if __name__ == "__main__":
    main()
