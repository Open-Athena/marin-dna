"""issue #302 — iteration 39: the mechanism as a CONTINUOUS axis (sharpening iter33's threshold).
Define each site's MAMMAL-SPECIFICITY = z(phyloP_241m) - z(phyloP_100v): high = conserved in
mammals but relaxed across vertebrates (the 'trap' — looks constrained to a cross-species
likelihood, yet tolerated). Question: does the model over-call benigns in PROPORTION to mammal-
specificity, and does that proportionality STEEPEN with scale?

Per ladder model, on missense: (1) Spearman(benign minus_llr, mammal_specificity) across scale;
(2) binned over-call — mean within-model score-percentile of benigns per mammal-specificity
quintile, one line per model. mammal_specificity is model-independent (phyloP only) so the x-axis
is fixed across the ladder. CPU; reads S3.
Run: OMP_NUM_THREADS=2 uv run --group genome-s3 python scripts/issue302/iter39_mammal_specificity.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import rankdata, spearmanr

SCORES = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores"
CB = "s3://oa-bolinas/snakemake/conservation_eval/results/mendelian_traits"
OUT_S3 = "s3://oa-bolinas/analysis/issue302/probe_out"
OUT = Path("scratch/issue302/figs")
KEY = ["chrom", "pos", "ref", "alt"]
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


def _track(name):
    return (
        pl.read_parquet(f"{CB}/{name}_train.parquet")
        .with_columns(pl.col("chrom").cast(str))
        .unique(subset=KEY)
        .select([*KEY, pl.col("score").alias(name)])
    )


def _z(x):
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)


def main() -> None:
    p241, p100 = _track("phyloP_241m"), _track("phyloP_100v")
    rows, curves = [], {}
    NB = 5
    edges = None
    for sdir, params in LADDER:
        m = (
            pl.read_parquet(f"{SCORES}/{sdir}/mendelian_traits.parquet")
            .filter(pl.col("subset") == "missense_variant")
            .with_columns(
                pl.col("chrom").cast(str),
                (-(pl.col("llr_fwd") + pl.col("llr_rc")) / 2).alias("mll"),
            )
            .join(p241, on=KEY, how="left")
            .join(p100, on=KEY, how="left")
            .drop_nulls(["phyloP_241m", "phyloP_100v", "mll"])
        )
        lab = m["label"].to_numpy()
        mll = m["mll"].to_numpy()
        spec = _z(m["phyloP_241m"].to_numpy()) - _z(m["phyloP_100v"].to_numpy())
        pct = rankdata(mll) / len(
            mll
        )  # within-model score percentile (high=pathogenic-ranked)
        bmask = lab == 0
        rho = spearmanr(mll[bmask], spec[bmask]).statistic
        # also control: among benigns, partial — corr with raw phyloP_241m (mammal conservation alone)
        rho_241 = spearmanr(mll[bmask], m["phyloP_241m"].to_numpy()[bmask]).statistic
        rows.append(
            {"params": params, "rho_spec": float(rho), "rho_241": float(rho_241)}
        )
        if edges is None:
            edges = np.quantile(spec[bmask], np.linspace(0, 1, NB + 1))
        binidx = np.clip(np.digitize(spec[bmask], edges[1:-1]), 0, NB - 1)
        curves[params] = [float(np.mean(pct[bmask][binidx == b])) for b in range(NB)]
        print(
            f"{params:>5}M | Spearman(benign mll, mammal-specificity)={rho:+.3f}  (vs raw phyloP_241m={rho_241:+.3f})"
        )

    res = pl.DataFrame(rows)
    res.write_parquet("scratch/issue302/mammal_specificity.parquet")
    c = res.to_pandas()
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))
    pp = [p for _, p in LADDER]
    qcol = plt.get_cmap("coolwarm")
    qlabels = [
        "Q1 (vertebrate-conserved)",
        "Q2",
        "Q3 (middle)",
        "Q4",
        "Q5 (mammal-specific 'trap')",
    ]
    for b in range(NB):
        traj = [curves[p][b] for p in pp]
        ax[0].plot(
            pp,
            traj,
            "o-",
            color=qcol(b / (NB - 1)),
            lw=2.4 if b in (0, NB - 1) else 1.4,
            label=qlabels[b],
        )
    ax[0].axvline(128, ls=":", color="gray", lw=1)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("params (M, log)")
    ax[0].set_ylabel("benign over-call (mean within-model score percentile)")
    ax[0].set_title(
        "The scale-driven over-call is SPECIFIC to the mammal-specific\nbenigns: Q5 rises 0.51\u21920.56; vertebrate-conserved Q1 is flat/down"
    )
    ax[0].legend(fontsize=7.5, title="mammal-specificity quintile")
    ax[0].grid(alpha=0.3)
    ax[1].plot(
        c["params"],
        c["rho_spec"],
        "o-",
        color="tab:red",
        lw=2.5,
        label="vs mammal-specificity\n(z241 − z100v)",
    )
    ax[1].plot(
        c["params"],
        c["rho_241"],
        "s--",
        color="gray",
        lw=2,
        label="vs raw mammal phyloP_241m",
    )
    ax[1].axvline(128, ls=":", color="gray", lw=1)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("params (M, log)")
    ax[1].set_ylabel("Spearman(benign minus_llr, axis)")
    ax[1].set_title(
        "The benign over-call keys on mammal-specificity\nMORE with scale (rises past the ~128M peak)"
    )
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.suptitle(
        "The over-call grows with MAMMAL-SPECIFICITY and with SCALE (modest but monotonic): the scale effect is concentrated in the mammal-conserved-but-vertebrate-relaxed benigns",
        y=1.02,
        fontsize=9.5,
    )
    fig.tight_layout()
    fig.savefig(OUT / "mammal_specificity.png", dpi=130, bbox_inches="tight")
    fig.savefig(OUT / "mammal_specificity.svg", bbox_inches="tight")
    plt.close(fig)
    print(
        f"  wrote {OUT / 'mammal_specificity'}  | rho_spec 46M={c['rho_spec'].iloc[0]:.3f} -> 4B={c['rho_spec'].iloc[-1]:.3f}"
    )

    import s3fs

    fs = s3fs.S3FileSystem()
    for p in (str(OUT / "mammal_specificity.png"), str(OUT / "mammal_specificity.svg")):
        fs.put(p, f"{OUT_S3.replace('s3://', '')}/{Path(p).name}")
    print(f"  uploaded -> {OUT_S3}")


if __name__ == "__main__":
    main()
