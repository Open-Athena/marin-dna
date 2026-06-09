"""issue #302 / #306 — iteration 17: characterizing SGE vs Mendelian missense, and testing
whether allele frequency explains the benchmark-specific degradation.

#306: the missense scale-degradation replicates on Mendelian (ClinVar) but NOT on SGE
(saturation genome editing): SGE missense AUPRC *rises* 0.285->0.384.

Feature characterization (what GB asked): the NEGATIVE sets are built completely differently.
Mendelian benigns are matched COMMON population variants (100% in gnomAD, 35% AF>1%); SGE
neutrals come from a saturation screen and are ~all rare/novel (0.8% in gnomAD). Tempting
mechanism: "the model over-calls common-tolerated benigns, which only Mendelian has."

BUT we TEST that and it FAILS: dropping the common-AF benigns leaves the Mendelian degradation
fully intact (rare-only negatives still 0.54->0.43). And AUROC(pos vs common-neg)=0.89 >
AUROC(pos vs rare-neg)=0.82 — common benigns are the EASY ones globally. So AF is not the
cause; the degradation is the conserved-tolerated-benign over-call (iter3/5/8), and the
cross-benchmark difference is one of constraint/matching, not allele frequency.

  Panel A: AF composition of pos & neg, Mendelian vs SGE (the characterization).
  Panel B: AUPRC across scale — Mendelian full vs rare-only-negatives vs SGE. Both Mendelian
           curves degrade (AF removed -> still degrades); SGE rises. (Compare shapes, not
           levels: prevalence differs across the three.)

Run: OMP_NUM_THREADS=2 uv run --group genome-s3 python scripts/issue302/iter17_sge_af_mechanism.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score as ap

SCORES_S3 = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores"
OUT_S3 = "s3://oa-bolinas/analysis/issue302/probe_out"
OUT = Path("scratch/issue302/figs")
KEY = ["chrom", "pos", "ref", "alt"]
COMMON_AF = 0.01
LADDER = [
    ("scaling-v0.5-h640-p46M-step-215573", 46),
    ("scaling-v0.5-h768-p76M-step-215573", 76),
    ("scaling-v0.5-h896-p128M-step-215573", 128),
    ("scaling-v0.5-h1152-p255M-step-215573", 255),
    ("scaling-v0.5-h1408-p476M-step-215573", 476),
    ("scaling-v0.5-h1920-p1B-step-215573", 1120),
    ("scaling-v0.5-h2432-p2B-step-215573", 2270),
    ("scaling-v0.5-h2944-p4B-step-215573", 4020),
]


def _mll(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("chrom").cast(str),
        (-(pl.col("llr_fwd") + pl.col("llr_rc")) / 2).alias("mll"),
    )


def _ap(pos: np.ndarray, neg: np.ndarray) -> float:
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return ap(y, np.r_[pos, neg])


def main() -> None:
    from datasets import load_dataset

    sge_af = (
        pl.from_pandas(load_dataset("bolinas-dna/evals_sge", split="train").to_pandas())
        .with_columns(pl.col("chrom").cast(str))
        .select([*KEY, "author_gnomad_af"])
    )

    rows = []
    comp = None
    for sdir, params in LADDER:
        men = (
            _mll(pl.read_parquet(f"{SCORES_S3}/{sdir}/mendelian_traits.parquet"))
            .filter(pl.col("subset") == "missense_variant")
            .with_columns(pl.col("AF").fill_null(0.0))
        )
        sge = (
            _mll(pl.read_parquet(f"{SCORES_S3}/{sdir}/sge.parquet"))
            .filter(pl.col("subset") == "missense_variant")
            .join(sge_af, on=KEY, how="left")
            .with_columns(pl.col("author_gnomad_af").fill_null(0.0))
        )
        mp = men.filter(pl.col("label") == 1)["mll"].to_numpy()
        mnR = men.filter((pl.col("label") == 0) & (pl.col("AF") <= COMMON_AF))[
            "mll"
        ].to_numpy()
        mn = men.filter(pl.col("label") == 0)["mll"].to_numpy()
        sp = sge.filter(pl.col("label") == 1)["mll"].to_numpy()
        sn = sge.filter(pl.col("label") == 0)["mll"].to_numpy()
        rows.append(
            {
                "params": params,
                "MEN_full": _ap(mp, mn),
                "MEN_rareNeg": _ap(mp, mnR),
                "SGE": _ap(sp, sn),
            }
        )
        if comp is None:

            def afstats(df, af):
                return {
                    "in_gnomad": float((df[af] > 0).mean()) * 100,
                    "common": float((df[af] > COMMON_AF).mean()) * 100,
                }

            comp = {
                "MEN_pos": afstats(men.filter(pl.col("label") == 1), "AF"),
                "MEN_neg": afstats(men.filter(pl.col("label") == 0), "AF"),
                "SGE_pos": afstats(
                    sge.filter(pl.col("label") == 1), "author_gnomad_af"
                ),
                "SGE_neg": afstats(
                    sge.filter(pl.col("label") == 0), "author_gnomad_af"
                ),
            }
        print(
            f"{params:>5}M | MEN full={rows[-1]['MEN_full']:.3f} rare-neg={rows[-1]['MEN_rareNeg']:.3f} | SGE={rows[-1]['SGE']:.3f}"
        )

    res = pl.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.write_parquet("scratch/issue302/sge_af_mechanism.parquet")
    c = res.to_pandas()
    print(f"\nAF composition: {comp}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.7))
    # Panel A — AF composition of pos & neg, both benchmarks.
    groups = ["MEN_pos", "MEN_neg", "SGE_pos", "SGE_neg"]
    labels = ["Mendelian\npos", "Mendelian\nneg", "SGE\npos", "SGE\nneg"]
    cols = ["#c0392b", "#7f1d1d", "#2980b9", "#1b3a5b"]
    x = np.arange(len(groups))
    ax[0].bar(x, [comp[g]["in_gnomad"] for g in groups], color=cols)
    for i, g in enumerate(groups):
        ax[0].text(
            i,
            comp[g]["in_gnomad"] + 1.5,
            f"{comp[g]['in_gnomad']:.0f}%\n({comp[g]['common']:.0f}% common)",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels, fontsize=9)
    ax[0].set_ylabel("% observed in gnomAD")
    ax[0].set_ylim(0, 115)
    ax[0].set_title(
        "Only the Mendelian NEGATIVES are common population variants\n(matched benigns); everything else is rare/novel"
    )
    ax[0].grid(alpha=0.3, axis="y")
    # Panel B — drop-common counterfactual.
    ax[1].plot(
        c["params"],
        c["MEN_full"],
        "D-",
        color="tab:red",
        lw=2.5,
        label="Mendelian (full negatives)",
    )
    ax[1].plot(
        c["params"],
        c["MEN_rareNeg"],
        "o--",
        color="tab:orange",
        lw=2,
        label="Mendelian (common benigns REMOVED)",
    )
    ax[1].plot(
        c["params"],
        c["SGE"],
        "s-",
        color="tab:blue",
        lw=2.5,
        label="SGE (all-rare negatives)",
    )
    ax[1].set_xscale("log")
    ax[1].set_xlabel("params (M, log)")
    ax[1].set_ylabel("missense AUPRC (minus_llr)")
    ax[1].set_title(
        "AF is NOT the cause: removing common benigns leaves\nthe Mendelian degradation intact; SGE still rises"
    )
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.suptitle(
        "SGE vs Mendelian missense: the negatives differ in AF, but AF does not explain the benchmark-specific degradation",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT / "sge_af_mechanism.png", dpi=130, bbox_inches="tight")
    fig.savefig(OUT / "sge_af_mechanism.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / 'sge_af_mechanism'}.{{png,svg}}")

    import s3fs

    fs = s3fs.S3FileSystem()
    for p in (
        "scratch/issue302/sge_af_mechanism.parquet",
        str(OUT / "sge_af_mechanism.png"),
        str(OUT / "sge_af_mechanism.svg"),
    ):
        fs.put(p, f"{OUT_S3.replace('s3://', '')}/{Path(p).name}")
    print(f"  uploaded -> {OUT_S3}")


if __name__ == "__main__":
    main()
