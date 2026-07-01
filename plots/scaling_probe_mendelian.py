"""Issue #341 figures: scaling-ladder Mendelian VEP — frozen-embedding linear probe
vs zero-shot LLR, per consequence subset, over model size.

Reproduces the #302 iteration-10 finding through the **official** pipeline (#320
`compute_probe` + #331/#341 `compute_probe_metrics`): across the 8-rung
`dna-bolinas-scaling-v0.5` ladder (46M→4B, step-215573), does the probe's
per-chromosome-weighted Mendelian AUPRC rise ~monotonically with scale while the
zero-shot LLR peaks (~128M) then degrades? Reported **per subset** (no macro-avg).

Two figures (seaborn figure-level `relplot`):

  figure   — per-subset facet grid; each panel plots two lines over model size,
             the **linear probe** (`probe_score`) vs the **zero-shot LLR**
             (`minus_llr_avg`), scored on identical rows under the identical
             per-chrom-weighted metric (own y-scale per subset).
  missense — the `missense_variant` subset alone (the direct #302 iter10 analog).

Metric = per-chromosome-weighted AUPRC (TraitGym / #331): AUPRC within each
chromosome, size-weighted across chromosomes — a **point estimate** (no bootstrap
SE), so no error bars. The 1:9 matched positive prevalence gives a 0.10 random
baseline (dashed).

Self-contained: reads the `compute_probe_metrics` parquets per model from S3. Params
are the published final-checkpoint counts (issue #274), hardcoded. Emits SVG + PNG to
`plots/output/scaling_probe_mendelian/`.

Usage:
    uv run python plots/scaling_probe_mendelian.py \
        --metrics-prefix s3://oa-bolinas/snakemake/analysis/evals_v2/results/probe_metrics
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import polars as pl

OUT_DIR = Path(__file__).resolve().parent / "output" / "scaling_probe_mendelian"

# 8-model scaling ladder (issues #274 / #302 / #306), all at step-215573 (~84.8B tokens).
SIZES = [
    "h640-p46M",
    "h768-p76M",
    "h896-p128M",
    "h1152-p255M",
    "h1408-p476M",
    "h1920-p1B",
    "h2432-p2B",
    "h2944-p4B",
]
# Total parameter count per size (issue #274 table) — x-axis position.
PARAMS = {
    "h640-p46M": 45.9e6,
    "h768-p76M": 75.5e6,
    "h896-p128M": 128.5e6,
    "h1152-p255M": 254.8e6,
    "h1408-p476M": 475.9e6,
    "h1920-p1B": 1.12e9,
    "h2432-p2B": 2.27e9,
    "h2944-p4B": 4.02e9,
}
SIZE_LABEL = {
    "h640-p46M": "46M",
    "h768-p76M": "76M",
    "h896-p128M": "128M",
    "h1152-p255M": "255M",
    "h1408-p476M": "476M",
    "h1920-p1B": "1B",
    "h2432-p2B": "2B",
    "h2944-p4B": "4B",
}

# Prevalence baseline for the 1:9-matched Mendelian set (10% positives).
BASELINE = 0.10

# The two score types compute_probe_metrics emits, with display labels + colors.
PROBE_ST = "probe_score"
LLR_ST = "minus_llr_avg"
PROBE_LABEL = "linear probe"
LLR_LABEL = "zero-shot LLR"
LABEL_BY_ST = {PROBE_ST: PROBE_LABEL, LLR_ST: LLR_LABEL}
PALETTE = {PROBE_LABEL: "tab:blue", LLR_LABEL: "tab:red"}
HUE_ORDER = [PROBE_LABEL, LLR_LABEL]

# Subsets to plot, in facet order (missense first — the #302 focus). Restricted to
# the consequence classes the models actually see during training — coding
# (missense / synonymous), splicing, both UTRs, and TSS-proximal. Deliberately
# EXCLUDES distal / non_coding_transcript_exon / mature_miRNA (not / barely covered
# in training), so the VEP comparison stays on in-distribution consequence classes.
KEEP_SUBSETS = [
    "missense_variant",
    "synonymous_variant",
    "splicing",
    "5_prime_UTR_variant",
    "3_prime_UTR_variant",
    "tss_proximal",
]


def _model(size: str) -> str:
    return f"scaling-v0.5-{size}-step-215573"


def _disp(subset: str) -> str:
    return subset.replace("_variant", "")


def _read(path: str, what: str) -> pl.DataFrame:
    try:
        return pl.read_parquet(path)
    except Exception as e:  # fail loud, name the missing cell
        raise RuntimeError(
            f"{what}: could not read {path} — has its sky cell finished? ({e})"
        )


def load(prefix: str) -> pl.DataFrame:
    """Concatenate every rung's probe_metrics parquet, tagged with its `size` token."""
    frames = []
    for s in SIZES:
        m = _read(
            f"{prefix}/{_model(s)}/mendelian_traits.parquet", f"probe_metrics {s}"
        )
        frames.append(m.with_columns(pl.lit(s).alias("size")))
    return pl.concat(frames, how="vertical_relaxed")


def ordered_subsets(df: pl.DataFrame) -> list[str]:
    """The kept subsets that are actually present, in KEEP_SUBSETS order."""
    present = set(df["subset"].unique().to_list())
    return [s for s in KEEP_SUBSETS if s in present]


def to_long(df: pl.DataFrame) -> pd.DataFrame:
    """Tidy long-form for seaborn: one row per (size, subset, score) with a finite
    value. Restricted to KEEP_SUBSETS; drops non-finite values (a subset whose probe
    was skipped → NaN)."""
    pdf = (
        df.filter(
            pl.col("score_type").is_in([PROBE_ST, LLR_ST])
            & pl.col("subset").is_in(KEEP_SUBSETS)
        )
        .select(["size", "subset", "score_type", "value", "n_pos"])
        .to_pandas()
    )
    pdf = pdf[pdf["value"].apply(lambda v: v is not None and math.isfinite(v))].copy()
    pdf["params"] = pdf["size"].map(PARAMS)
    pdf["subset_disp"] = pdf["subset"].map(_disp)
    pdf["score"] = pdf["score_type"].map(LABEL_BY_ST)
    return pdf


def _style_axes(g) -> None:
    """Shared axis cosmetics for a relplot FacetGrid: log-x with size-label ticks,
    a dashed prevalence baseline, and a light grid."""
    x_vals = [PARAMS[s] for s in SIZES]
    x_labels = [SIZE_LABEL[s] for s in SIZES]
    g.set(xscale="log")
    for ax in g.axes.flat:
        ax.axhline(BASELINE, ls="--", lw=0.8, color="gray", alpha=0.7)
        ax.set_xticks(x_vals)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.grid(True, alpha=0.3)


def print_table(df: pl.DataFrame, subsets: list[str]) -> None:
    print("\n=== per-chrom-weighted AUPRC — probe (P) vs zero-shot LLR (Z) ===")
    wide = df.filter(pl.col("score_type").is_in([PROBE_ST, LLR_ST]))
    for subset in subsets:
        print(f"\n{subset}")
        print(f"{'size':>6} {'params':>7} | {'probe':>7} {'LLR':>7} {'Δ(P−Z)':>8}")
        for s in SIZES:
            cell = wide.filter((pl.col("size") == s) & (pl.col("subset") == subset))

            def _v(st: str) -> float:
                hit = cell.filter(pl.col("score_type") == st)
                v = hit["value"][0] if hit.height else None
                return float("nan") if v is None else float(v)

            pv, zv = _v(PROBE_ST), _v(LLR_ST)
            print(
                f"{SIZE_LABEL[s]:>6} {PARAMS[s] / 1e6:>6.0f}M | "
                f"{pv:>7.3f} {zv:>7.3f} {pv - zv:>+8.3f}"
            )


def build_grid(sns, pdf: pd.DataFrame, subsets: list[str]) -> None:
    g = sns.relplot(
        data=pdf,
        x="params",
        y="value",
        hue="score",
        hue_order=HUE_ORDER,
        palette=PALETTE,
        marker="o",
        col="subset_disp",
        col_order=[_disp(s) for s in subsets],
        col_wrap=3,
        kind="line",
        markersize=8,
        linewidth=1.8,
        facet_kws={"sharey": False},
        height=3.0,
        aspect=1.35,
    )
    _style_axes(g)
    g.set_axis_labels("model size (params, log)", "per-chrom AUPRC")
    g.set_titles("{col_name}")
    g.legend.set_title("")
    g.figure.suptitle(
        "Scaling ladder (46M→4B): Mendelian per-chrom-weighted AUPRC by subset — "
        "linear probe vs zero-shot LLR (#341)",
        y=1.02,
    )
    _save(g, "figure")


def build_missense(sns, pdf: pd.DataFrame) -> None:
    mdf = pdf[pdf["subset"] == "missense_variant"]
    g = sns.relplot(
        data=mdf,
        x="params",
        y="value",
        hue="score",
        hue_order=HUE_ORDER,
        palette=PALETTE,
        marker="o",
        kind="line",
        markersize=9,
        linewidth=2.0,
        height=5,
        aspect=1.35,
    )
    _style_axes(g)
    g.set_axis_labels("model size (params, log)", "per-chrom-weighted AUPRC")
    g.legend.set_title("")
    g.figure.suptitle(
        "Mendelian missense: linear probe vs zero-shot LLR across scale "
        "(#341, cf. #302 iter10)",
        y=1.02,
    )
    _save(g, "missense")


def _save(g, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    svg = OUT_DIR / f"{stem}.svg"
    g.savefig(svg, dpi=200, bbox_inches="tight")
    g.savefig(svg.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {svg} (+ .png)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--metrics-prefix",
        default="s3://oa-bolinas/snakemake/analysis/evals_v2/results/probe_metrics",
    )
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.fonttype"] = "none"
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")

    df = load(args.metrics_prefix)
    subsets = ordered_subsets(df)
    pdf = to_long(df)

    print_table(df, subsets)
    build_grid(sns, pdf, subsets)
    build_missense(sns, pdf)


if __name__ == "__main__":
    main()
