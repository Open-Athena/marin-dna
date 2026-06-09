"""Issue #306 figure: AUPRC vs model size on SGE vs Mendelian, for missense & splicing.

Two panels over the 8 `dna-bolinas-scaling-v0.5` sizes (46M→4B, step-215573):
  Panel A — missense_variant
  Panel B — splicing

Within each panel, two lines on **two independent y-axes** (shared log-x = params):
  left  (blue)  — Mendelian AUPRC: official evals_v2, pooled within-consequence,
                  matched 1:9 cluster bootstrap (reused from #274, not recomputed).
  right (red)   — SGE AUPRC: macro-averaged across genes (`accession == _macro_avg_`).

The two AUPRCs are constructed differently (Mendelian = pooled/matched; SGE =
unmatched per-gene macro) and are **not level-comparable** — the dual axes make
explicit that we compare the *shapes / trends with scale*, never the vertical gap
between the lines. Both use `minus_llr_avg` (signed −LLR, FWD+RC mean).

Self-contained: reads the two evals_v2 metric parquets per model from S3. Params
are the published final-checkpoint counts (issue #274 table), hardcoded so the
recipe needs no W&B. Emits SVG + PNG to `plots/output/scaling_sge_vs_mendelian/`.

Usage:
    uv run python plots/scaling_sge_vs_mendelian.py \
        --metrics-prefix s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

OUT_DIR = Path(__file__).resolve().parent / "output" / "scaling_sge_vs_mendelian"

# 8-model scaling ladder (issues #274 / #306), all at step-215573 (~84.8B tokens).
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
CONSEQUENCES = ["missense_variant", "splicing"]

MEN_COLOR, SGE_COLOR = "tab:blue", "tab:red"


def _model(size: str) -> str:
    return f"scaling-v0.5-{size}-step-215573"


def _read(path: str, what: str) -> pl.DataFrame:
    try:
        return pl.read_parquet(path)
    except Exception as e:  # fail loud, name the missing cell
        raise RuntimeError(
            f"{what}: could not read {path} — has its sky cell finished? ({e})"
        )


def load_mendelian(prefix: str, score_type: str) -> dict[tuple[str, str], dict]:
    """(size, consequence) -> {value, se}. Pooled within-consequence AUPRC (#274)."""
    out: dict[tuple[str, str], dict] = {}
    for s in SIZES:
        m = _read(f"{prefix}/{_model(s)}/mendelian_traits.parquet", f"mendelian {s}")
        m = m.filter(pl.col("score_type") == score_type)
        for c in CONSEQUENCES:
            hit = m.filter(pl.col("subset") == c)
            assert hit.height == 1, (
                f"{_model(s)} mendelian {c}: expected 1 row, got {hit.height}"
            )
            out[(s, c)] = {"value": hit["value"][0], "se": hit["se"][0]}
    return out


def load_sge(prefix: str, score_type: str) -> dict[tuple[str, str], dict]:
    """(size, consequence) -> {value, se, n_genes}. Macro across genes (_macro_avg_)."""
    out: dict[tuple[str, str], dict] = {}
    for s in SIZES:
        m = _read(f"{prefix}/{_model(s)}/sge.parquet", f"sge {s}")
        m = m.filter(
            (pl.col("accession") == "_macro_avg_")
            & (pl.col("score_type") == score_type)
        )
        for c in CONSEQUENCES:
            hit = m.filter(pl.col("subset") == c)
            assert hit.height == 1, (
                f"{_model(s)} sge {c}: expected 1 macro row, got {hit.height}"
            )
            out[(s, c)] = {
                "value": hit["value"][0],
                "se": hit["se"][0],
                "n_genes": int(hit["n"][0]),
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--metrics-prefix",
        default="s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics",
    )
    ap.add_argument("--score-type", default="minus_llr_avg")
    args = ap.parse_args()

    men = load_mendelian(args.metrics_prefix, args.score_type)
    sge = load_sge(args.metrics_prefix, args.score_type)

    # Console table (for the issue comment).
    print(
        f"score_type={args.score_type}  (Mendelian pooled-within-consequence | SGE macro-across-genes)"
    )
    print(
        f"{'size':>6} {'params':>8} | {'men_miss':>9} {'sge_miss':>9} ({'g':>2}) | {'men_splice':>10} {'sge_splice':>10} ({'g':>2})"
    )
    for s in SIZES:
        mm, sm = men[(s, "missense_variant")], sge[(s, "missense_variant")]
        mp, sp = men[(s, "splicing")], sge[(s, "splicing")]
        print(
            f"{SIZE_LABEL[s]:>6} {PARAMS[s] / 1e6:>7.0f}M | {mm['value']:>9.3f} {sm['value']:>9.3f} ({sm['n_genes']:>2}) "
            f"| {mp['value']:>10.3f} {sp['value']:>10.3f} ({sp['n_genes']:>2})"
        )

    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.fonttype"] = "none"
    import matplotlib.pyplot as plt

    x = [PARAMS[s] for s in SIZES]
    xt_labels = [SIZE_LABEL[s] for s in SIZES]

    # constrained layout auto-spaces the twin-axis labels/ticks between the two
    # panels (the inner SGE/Mendelian labels would otherwise collide in the
    # centre); the legend goes *outside* the panels so it can't overlap the data.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
    handles = None
    gene_note = {}
    for ax, c in zip(axes, CONSEQUENCES):
        # Mendelian — left axis (blue)
        h_men = ax.errorbar(
            x,
            [men[(s, c)]["value"] for s in SIZES],
            yerr=[men[(s, c)]["se"] for s in SIZES],
            marker="o",
            ms=5,
            lw=1.5,
            color=MEN_COLOR,
            capsize=3,
        )
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(xt_labels, fontsize=8)
        ax.tick_params(axis="x", which="minor", bottom=False)
        ax.set_xlabel("model size (params)")
        ax.set_ylabel("Mendelian AUPRC", color=MEN_COLOR)
        ax.tick_params(axis="y", labelcolor=MEN_COLOR)
        ax.grid(True, alpha=0.3)

        # SGE — right axis (red)
        ax2 = ax.twinx()
        h_sge = ax2.errorbar(
            x,
            [sge[(s, c)]["value"] for s in SIZES],
            yerr=[sge[(s, c)]["se"] for s in SIZES],
            marker="s",
            ms=5,
            lw=1.5,
            color=SGE_COLOR,
            capsize=3,
        )
        ax2.set_ylabel("SGE AUPRC (macro)", color=SGE_COLOR)
        ax2.tick_params(axis="y", labelcolor=SGE_COLOR)

        ng = sorted({sge[(s, c)]["n_genes"] for s in SIZES})
        gene_note[c] = f"{ng[0]}" if len(ng) == 1 else f"{ng[0]}–{ng[-1]}"
        ax.set_title(c.replace("_variant", ""))
        if handles is None:
            handles = [h_men, h_sge]

    fig.suptitle(
        f"Scaling ladder (46M→4B, n=8): Mendelian vs SGE AUPRC by model size — "
        f"{args.score_type}, FWD+RC (#306)"
    )
    # Single legend outside the axes at the bottom, with the caveat as its title —
    # exactly one element on top (suptitle) and one at the bottom (legend), so
    # nothing can overlap the lines, the suptitle, the inner axis labels, or each
    # other (constrained_layout reserves room for all of it).
    leg = fig.legend(
        handles,
        ["Mendelian (left axis)", "SGE (right axis)"],
        loc="outside lower center",
        ncol=2,
        frameon=True,
        title=(
            "Independent y-axes — compare shapes/trends with scale, not the vertical gap.\n"
            f"SGE = macro across genes ({gene_note['missense_variant']} missense / "
            f"{gene_note['splicing']} splicing), unmatched   ·   "
            "Mendelian = pooled within-consequence, matched 1:9"
        ),
    )
    leg.get_title().set_fontsize(8)
    leg.get_title().set_color("dimgray")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    svg = OUT_DIR / "figure.svg"
    fig.savefig(svg, dpi=200)
    fig.savefig(svg.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"\nwrote {svg} (+ .png)")


if __name__ == "__main__":
    main()
