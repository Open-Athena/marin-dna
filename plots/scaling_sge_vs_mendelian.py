"""Issue #306 figure: AUPRC vs model size on Mendelian / complex traits / SGE.

Two panels over the 8 `dna-bolinas-scaling-v0.5` sizes (46M→4B, step-215573):
  Panel A — missense_variant
  Panel B — splicing

Lines (shared log-x = params; two independent y-axes per panel):
  left  — matched 1:9 AUPRC (baseline 0.10; same construction, comparable):
            • Mendelian (blue, −LLR), and
            • Complex traits (green, |LLR|) on both panels. NOTE complex
              *splicing* rests on only ~19 match-groups (vs ~250 for missense),
              so it carries a wide CI and is read with caution (caveat on figure).
  right — SGE AUPRC (red): macro-averaged across genes (`accession == _macro_avg_`, −LLR).

The matched-pair (left) and per-gene-macro SGE (right) AUPRCs are constructed
differently and are **not level-comparable** — the dual axes make explicit that
we compare *shapes / trends with scale*, never the vertical gap. All scores FWD+RC.

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
CONSEQUENCES = ["missense_variant", "splicing"]

MEN_COLOR, COMPLEX_COLOR, SGE_COLOR = "tab:blue", "tab:green", "tab:red"
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


def load_complex(prefix: str, score_type: str) -> dict[tuple[str, str], dict]:
    """(size, consequence) -> {value, se, n_groups}. Matched 1:9 AUPRC, abs_llr.

    Complex *splicing* has very few match-groups (~19 vs ~250 for missense), so
    its AUPRC carries a wide CI — plotted on both panels but caveated.
    """
    out: dict[tuple[str, str], dict] = {}
    for s in SIZES:
        m = _read(f"{prefix}/{_model(s)}/complex_traits.parquet", f"complex {s}")
        m = m.filter(pl.col("score_type") == score_type)
        for c in CONSEQUENCES:
            hit = m.filter(pl.col("subset") == c)
            assert hit.height == 1, (
                f"{_model(s)} complex {c}: expected 1 row, got {hit.height}"
            )
            out[(s, c)] = {
                "value": hit["value"][0],
                "se": hit["se"][0],
                "n_groups": int(hit["n_groups"][0]),
            }
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
    ap.add_argument("--score-type", default="minus_llr_avg", help="for mendelian & sge")
    args = ap.parse_args()

    men = load_mendelian(args.metrics_prefix, args.score_type)
    cpx = load_complex(args.metrics_prefix, COMPLEX_SCORE_TYPE)
    sge = load_sge(args.metrics_prefix, args.score_type)

    # Console table (for the issue comment).
    print(
        f"mendelian/sge={args.score_type} | complex={COMPLEX_SCORE_TYPE} "
        "(left=matched 1:9 AUPRC | SGE=macro across genes)"
    )
    hdr = ("size", "params", "men_M", "cpx_M", "sge_M", "men_S", "cpx_S", "sge_S")
    print(
        f"{hdr[0]:>6} {hdr[1]:>7} | {hdr[2]:>7} {hdr[3]:>7} {hdr[4]:>7} | "
        f"{hdr[5]:>7} {hdr[6]:>7} {hdr[7]:>7}"
    )
    for s in SIZES:
        v = lambda d, c: d[(s, c)]["value"]  # noqa: E731
        print(
            f"{SIZE_LABEL[s]:>6} {PARAMS[s] / 1e6:>6.0f}M | "
            f"{v(men, 'missense_variant'):>7.3f} {v(cpx, 'missense_variant'):>7.3f} {v(sge, 'missense_variant'):>7.3f} | "
            f"{v(men, 'splicing'):>7.3f} {v(cpx, 'splicing'):>7.3f} {v(sge, 'splicing'):>7.3f}"
        )

    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.fonttype"] = "none"
    import matplotlib.pyplot as plt

    x = [PARAMS[s] for s in SIZES]
    xt_labels = [SIZE_LABEL[s] for s in SIZES]
    cpx_spl_n = cpx[(SIZES[0], "splicing")]["n_groups"]  # constant across models

    # constrained layout auto-spaces the twin-axis labels/ticks between the two
    # panels (the inner labels would otherwise collide in the centre); the legend
    # goes *outside* the panels so it can't overlap the data.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
    h_men_ref = h_cpx_ref = h_sge_ref = None
    gene_note = {}
    for ax, c in zip(axes, CONSEQUENCES):
        # Left axis — matched 1:9 AUPRC (neutral/black: hosts Mendelian + Complex).
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
        h_cpx = ax.errorbar(
            x,
            [cpx[(s, c)]["value"] for s in SIZES],
            yerr=[cpx[(s, c)]["se"] for s in SIZES],
            marker="^",
            ms=5,
            lw=1.5,
            color=COMPLEX_COLOR,
            capsize=3,
        )
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(xt_labels, fontsize=8)
        ax.tick_params(axis="x", which="minor", bottom=False)
        ax.set_xlabel("model size (params)")
        ax.set_ylabel("AUPRC — matched 1:9")
        ax.grid(True, alpha=0.3)

        # Right axis — SGE macro across genes (red).
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
        h_men_ref, h_cpx_ref, h_sge_ref = h_men, h_cpx, h_sge

    fig.suptitle(
        "Scaling ladder (46M→4B, n=8): AUPRC by model size — "
        "Mendelian / complex / SGE, FWD+RC (#306)"
    )
    # One legend outside the axes at the bottom, caveat as its title — exactly one
    # element on top (suptitle) and one at the bottom (legend), so nothing overlaps
    # the lines, the suptitle, the inner axis labels, or each other.
    leg = fig.legend(
        [h_men_ref, h_cpx_ref, h_sge_ref],
        [
            "Mendelian (left, −LLR)",
            "Complex traits (left, |LLR|)",
            "SGE (right, macro across genes)",
        ],
        loc="outside lower center",
        ncol=3,
        frameon=True,
        title=(
            "Independent y-axes — compare shapes/trends with scale, not the vertical gap.   "
            "Left = matched 1:9 AUPRC (baseline 0.10); "
            f"right = SGE macro across genes ({gene_note['missense_variant']} missense / "
            f"{gene_note['splicing']} splicing), unmatched.\n"
            f"Caveat: complex splicing rests on only ~{cpx_spl_n} match-groups → wide CI; "
            "read its trend cautiously."
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
