"""Issue #306 figure: scaling-ladder native AUPRC vs model size, per dataset × subset.

One figure over the 8 `dna-bolinas-scaling-v0.5` sizes (46M→4B, step-215573): a
**dataset × subset grid** — 3 rows (Mendelian, complex traits, SGE) × 2 cols
(missense, splicing) = 6 panels, each on its **own native AUPRC scale** (missense
and splicing never share an axis; no cross-consequence mixing).

Scores (FWD+RC): Mendelian & SGE use −LLR (`minus_llr_avg`); complex uses |LLR|
(`abs_llr_avg`, its score_protocol). SGE is macro-averaged across genes; Mendelian
and complex are pooled within-consequence, matched 1:9 (baseline 0.10).
**Error bars are ±1 SE (bootstrap)** — drawn capless (a dispersion indicator, not a
bounded interval).

Caveat: complex *splicing* rests on only ~19 match-groups (vs ~250 missense) → its
AUPRC has a very wide SE and is read with caution.

Self-contained: reads the evals_v2 metric parquets per model from S3. Params are
the published final-checkpoint counts (issue #274), hardcoded so the recipe needs
no W&B. Emits SVG + PNG to `plots/output/scaling_sge_vs_mendelian/`.

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
CONSEQUENCES = ["missense_variant", "splicing"]  # columns
DATASETS = ["Mendelian", "Complex", "SGE"]  # rows

DATASET_COLOR = {"Mendelian": "tab:blue", "Complex": "tab:green", "SGE": "tab:red"}
CONSEQ_MARKER = {"missense_variant": "o", "splicing": "s"}

# complex_traits scores with the abs_llr protocol (magnitude); mendelian & sge use minus_llr.
COMPLEX_SCORE_TYPE = "abs_llr_avg"


def _model(size: str) -> str:
    return f"scaling-v0.5-{size}-step-215573"


def _read(path: str, what: str) -> pl.DataFrame:
    try:
        return pl.read_parquet(path)
    except Exception as e:  # fail loud, name the missing cell
        raise RuntimeError(
            f"{what}: could not read {path} — has its sky cell finished? ({e})"
        )


def load_matched(
    prefix: str, dataset: str, score_type: str
) -> dict[tuple[str, str], dict]:
    """Matched-pair (mendelian/complex) AUPRC: (size, consequence) -> {value, se, n_groups}.

    Pooled within-consequence AUPRC, matched 1:9, cluster bootstrap over match_groups.
    """
    out: dict[tuple[str, str], dict] = {}
    for s in SIZES:
        m = _read(f"{prefix}/{_model(s)}/{dataset}.parquet", f"{dataset} {s}")
        m = m.filter(pl.col("score_type") == score_type)
        for c in CONSEQUENCES:
            hit = m.filter(pl.col("subset") == c)
            assert hit.height == 1, (
                f"{_model(s)} {dataset} {c}: expected 1 row, got {hit.height}"
            )
            out[(s, c)] = {
                "value": hit["value"][0],
                "se": hit["se"][0],
                "n_groups": int(hit["n_groups"][0]),
            }
    return out


def load_sge(prefix: str, score_type: str) -> dict[tuple[str, str], dict]:
    """SGE macro across genes: (size, consequence) -> {value, se, n_genes} (_macro_avg_)."""
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


def _series(d: dict, c: str) -> tuple[list[float], list[float]]:
    """(values, ses) over SIZES for consequence c."""
    return ([d[(s, c)]["value"] for s in SIZES], [d[(s, c)]["se"] for s in SIZES])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--metrics-prefix",
        default="s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics",
    )
    ap.add_argument("--score-type", default="minus_llr_avg", help="for mendelian & sge")
    args = ap.parse_args()

    men = load_matched(args.metrics_prefix, "mendelian_traits", args.score_type)
    cpx = load_matched(args.metrics_prefix, "complex_traits", COMPLEX_SCORE_TYPE)
    sge = load_sge(args.metrics_prefix, args.score_type)
    data = {"Mendelian": men, "Complex": cpx, "SGE": sge}

    # Console table — native AUPRC (M=missense, S=splicing).
    print(
        f"mendelian/sge={args.score_type} | complex={COMPLEX_SCORE_TYPE}   (native AUPRC ± SE)"
    )
    print(
        f"{'size':>6} {'params':>7} | {'men_M':>7} {'cpx_M':>7} {'sge_M':>7} "
        f"| {'men_S':>7} {'cpx_S':>7} {'sge_S':>7}"
    )
    for s in SIZES:
        row = [SIZE_LABEL[s], f"{PARAMS[s] / 1e6:.0f}M"]
        for c in CONSEQUENCES:
            row += [f"{data[d][(s, c)]['value']:.3f}" for d in DATASETS]
        print(
            f"{row[0]:>6} {row[1]:>7} | {row[2]:>7} {row[3]:>7} {row[4]:>7} "
            f"| {row[5]:>7} {row[6]:>7} {row[7]:>7}"
        )
    cpx_spl_n = cpx[(SIZES[0], "splicing")]["n_groups"]
    sge_g = {c: sge[(SIZES[0], c)]["n_genes"] for c in CONSEQUENCES}

    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.fonttype"] = "none"
    import matplotlib.pyplot as plt

    x = [PARAMS[s] for s in SIZES]
    xt_labels = [SIZE_LABEL[s] for s in SIZES]

    # Native AUPRC grid: rows = dataset, cols = consequence. Each panel its own
    # scale (missense / splicing never share an axis). Error bars capless ±1 SE.
    fig, axes = plt.subplots(
        len(DATASETS),
        len(CONSEQUENCES),
        figsize=(9.5, 11),
        sharex=True,
        layout="constrained",
    )
    for ri, dsname in enumerate(DATASETS):
        for ci, c in enumerate(CONSEQUENCES):
            ax = axes[ri, ci]
            vals, ses = _series(data[dsname], c)
            ax.errorbar(
                x,
                vals,
                yerr=ses,
                marker=CONSEQ_MARKER[c],
                ms=5,
                lw=1.5,
                color=DATASET_COLOR[dsname],
                elinewidth=1,
            )
            ax.set_xscale("log")
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="x", which="minor", bottom=False)
            if ri == 0:  # column header = consequence
                ax.set_title(c.replace("_variant", ""), fontsize=12)
            if ci == 0:  # row label = dataset
                ax.set_ylabel(f"{dsname}\nAUPRC", fontsize=11)
            else:
                ax.set_ylabel("AUPRC")
    for ax in axes[-1]:  # x labels on the bottom row (sharex hides the upper rows')
        ax.set_xticks(x)
        ax.set_xticklabels(xt_labels, fontsize=8)
        ax.set_xlabel("model size (params)")

    fig.suptitle(
        "Scaling ladder (46M→4B, n=8): native AUPRC by dataset × subset — FWD+RC (#306)"
    )
    fig.supxlabel(
        "Error bars = ±1 SE (bootstrap).   Scores: Mendelian & SGE = −LLR, complex = |LLR|.   "
        "Mendelian/complex = pooled within-consequence, matched 1:9 (baseline 0.10); "
        f"SGE = macro across genes ({sge_g['missense_variant']} missense / {sge_g['splicing']} splicing).\n"
        f"Caveat: complex splicing rests on only ~{cpx_spl_n} match-groups → very wide SE; read its trend cautiously.",
        fontsize=8,
        color="dimgray",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    svg = OUT_DIR / "figure.svg"
    fig.savefig(svg, dpi=200)
    fig.savefig(svg.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"\nwrote {svg} (+ .png)")


if __name__ == "__main__":
    main()
