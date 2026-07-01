"""Issue #341 figures: scaling-ladder Mendelian VEP — frozen-embedding linear probe
vs zero-shot LLR, per consequence subset, over model size.

Reproduces the #302 iteration-10 finding through the **official** pipeline (#320
`compute_probe` + #331/#341 `compute_probe_metrics`): across the 8-rung
`dna-bolinas-scaling-v0.5` ladder (46M→4B, step-215573), does the probe's
per-chromosome-weighted Mendelian AUPRC rise ~monotonically with scale while the
zero-shot LLR peaks (~128M) then degrades? Reported **per subset** (no macro-avg).

Two figures:

  figure   — per-subset small-multiples grid; each panel plots two curves over size,
             the **linear probe** (`probe_score`) vs the **zero-shot LLR**
             (`minus_llr_avg`), scored on identical rows under the identical
             per-chrom-weighted metric.
  missense — the `missense_variant` subset alone (the direct #302 iter10 analog).

Metric = per-chromosome-weighted AUPRC (TraitGym / #331): AUPRC within each
chromosome, size-weighted across chromosomes — a **point estimate** (no bootstrap
SE), so unlike the native-AUPRC recipes there are no error bars. The 1:9 matched
positive prevalence gives a 0.10 random baseline (dashed).

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

# The two score types compute_probe_metrics emits, styled.
PROBE_ST = "probe_score"
LLR_ST = "minus_llr_avg"
SCORE_STYLE = {
    PROBE_ST: {"label": "linear probe", "color": "tab:blue", "marker": "o"},
    LLR_ST: {"label": "zero-shot LLR", "color": "tab:red", "marker": "s"},
}

# Preferred panel order (missense first — the #302 focus); any other subset present
# is appended alphabetically.
PREFERRED_SUBSETS = [
    "missense_variant",
    "synonymous_variant",
    "splicing",
    "5_prime_UTR_variant",
    "3_prime_UTR_variant",
    "tss_proximal",
    "distal",
    "non_coding_transcript_exon_variant",
    "mature_miRNA_variant",
]


def _model(size: str) -> str:
    return f"scaling-v0.5-{size}-step-215573"


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
    present = set(df["subset"].unique().to_list())
    ordered = [s for s in PREFERRED_SUBSETS if s in present]
    ordered += sorted(present - set(ordered))
    return ordered


def series(
    df: pl.DataFrame, subset: str, score_type: str
) -> tuple[list[float], list[float]]:
    """(params, values) over SIZES for one (subset, score_type), dropping non-finite
    values (a subset whose probe was skipped → NaN `probe_score`)."""
    xs: list[float] = []
    ys: list[float] = []
    for s in SIZES:
        hit = df.filter(
            (pl.col("size") == s)
            & (pl.col("subset") == subset)
            & (pl.col("score_type") == score_type)
        )
        if hit.height == 0:
            continue
        assert hit.height == 1, (
            f"{_model(s)} {subset} [{score_type}]: expected 1 row, got {hit.height}"
        )
        v = hit["value"][0]
        if v is not None and math.isfinite(v):
            xs.append(PARAMS[s])
            ys.append(float(v))
    return xs, ys


def _n_pos(df: pl.DataFrame, subset: str) -> int:
    """Positives in a subset (subset-level, score-independent) — for the panel title."""
    hit = df.filter((pl.col("size") == SIZES[-1]) & (pl.col("subset") == subset))
    return int(hit["n_pos"][0]) if hit.height else 0


def draw_panel(ax, df: pl.DataFrame, subset: str, x_ticks: list[float]) -> None:
    """Probe vs zero-shot LLR over size, for one subset."""
    for st in (PROBE_ST, LLR_ST):
        sty = SCORE_STYLE[st]
        xs, ys = series(df, subset, st)
        ax.plot(
            xs,
            ys,
            marker=sty["marker"],
            ms=5,
            lw=1.6,
            color=sty["color"],
            label=sty["label"],
        )
    ax.axhline(BASELINE, ls="--", lw=0.8, color="gray", alpha=0.7)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([SIZE_LABEL[s] for s in SIZES], fontsize=8)
    ax.set_title(
        f"{subset.replace('_variant', '')}  (n_pos≈{_n_pos(df, subset)})", fontsize=11
    )


def print_table(df: pl.DataFrame, subsets: list[str]) -> None:
    print("\n=== per-chrom-weighted AUPRC — probe (P) vs zero-shot LLR (Z) ===")
    for subset in subsets:
        print(f"\n{subset}")
        print(f"{'size':>6} {'params':>7} | {'probe':>7} {'LLR':>7} {'Δ(P−Z)':>8}")
        for s in SIZES:
            p, _ = series(df.filter(pl.col("size") == s), subset, PROBE_ST)
            z, _ = series(df.filter(pl.col("size") == s), subset, LLR_ST)
            pv = p[0] if p else float("nan")
            zv = z[0] if z else float("nan")
            print(
                f"{SIZE_LABEL[s]:>6} {PARAMS[s] / 1e6:>6.0f}M | "
                f"{pv:>7.3f} {zv:>7.3f} {pv - zv:>+8.3f}"
            )


def build_grid(plt, df: pl.DataFrame, subsets: list[str], x_ticks: list[float]) -> None:
    ncol = 3
    nrow = math.ceil(len(subsets) / ncol)
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), sharex=True, layout="constrained"
    )
    axes = axes.ravel()
    for ax, subset in zip(axes, subsets):
        draw_panel(ax, df, subset, x_ticks)
        ax.set_ylabel("per-chrom AUPRC", fontsize=9)
    for ax in axes[len(subsets) :]:  # hide unused cells
        ax.set_visible(False)
    axes[0].legend(fontsize=9, loc="best")
    fig.suptitle(
        "Scaling ladder (46M→4B, n=8): Mendelian per-chrom-weighted AUPRC by subset — "
        "linear probe vs zero-shot LLR (#341)"
    )
    fig.supxlabel(
        "model size (params, log).   Metric = per-chromosome-weighted AUPRC (TraitGym / #331), "
        "point estimate (no bootstrap SE).   Probe = nested-LOOC concat_ref_delta on frozen "
        "last-layer embeddings (#320); zero-shot = −LLR (FWD+RC).   Dashed = 0.10 (1:9 prevalence).",
        fontsize=8,
        color="dimgray",
    )
    _save(fig, plt, "figure")


def build_missense(plt, df: pl.DataFrame, x_ticks: list[float]) -> None:
    subset = "missense_variant"
    fig, ax = plt.subplots(figsize=(6.5, 5), layout="constrained")
    draw_panel(ax, df, subset, x_ticks)
    ax.set_ylabel("per-chrom-weighted AUPRC", fontsize=11)
    ax.set_xlabel("model size (params, log)")
    ax.legend(fontsize=10, loc="best")
    ax.set_title(
        "Mendelian missense: linear probe vs zero-shot LLR across scale (#341, cf. #302 iter10)",
        fontsize=11,
    )
    fig.supxlabel(
        "Per-chrom-weighted AUPRC (point estimate).  Probe = nested-LOOC concat_ref_delta on "
        "frozen embeddings (#320).  Dashed = 0.10 (1:9 prevalence).",
        fontsize=8,
        color="dimgray",
    )
    _save(fig, plt, "missense")


def _save(fig, plt, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    svg = OUT_DIR / f"{stem}.svg"
    fig.savefig(svg, dpi=200)
    fig.savefig(svg.with_suffix(".png"), dpi=200)
    plt.close(fig)
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
    import matplotlib.pyplot as plt

    df = load(args.metrics_prefix)
    subsets = ordered_subsets(df)
    x_ticks = [PARAMS[s] for s in SIZES]

    print_table(df, subsets)
    build_grid(plt, df, subsets, x_ticks)
    build_missense(plt, df, x_ticks)


if __name__ == "__main__":
    main()
