"""issue #302 — iteration 37: does adding the MSA signal (GPN-Star) actually FIX the missense
scale-degradation, and is our gLM ADDITIVE to MSA? The body claims "the remedy must inject
MSA/structure/human info, not a conservation knob" — this demonstrates it directly, non-circularly
(GPN-Star is an independent model's score, not a label-derived feature).

For each ladder model: standardize our minus_llr and GPN-Star-V on the shared missense set, take a
50/50 z-score ensemble, and compute missense AUPRC across scale for: ours alone / GPN-Star alone
(constant) / ensemble. Finding: a naive 50/50 ensemble stays BELOW GPN-Star alone at every scale and still
degrades (attenuated) — for Mendelian missense the single-sequence gLM is not additive to the
MSA model; GPN-Star alone dominates. (iter5 saw the same supervised: 4B+features 0.463 < GPN 0.682.)

CPU; reads S3 (our scores) + gist (GPN-Star-V). Writes fig to S3.
Run: OMP_NUM_THREADS=2 uv run --group genome-s3 python scripts/issue302/iter37_ensemble_fix.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score as ap

SCORES = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores"
GPN_V = "https://gist.githubusercontent.com/gonzalobenegas/db282f89aa00244fbb7437dce0f069ef/raw/02484d50d9bfd80337e313652b26f98a9362b6b1/bolinas_mendelian_traits_GPN-Star-V.parquet"
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


def _z(x):
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)


def main() -> None:
    gpn = (
        pl.read_parquet(GPN_V)
        .with_columns(
            [pl.col("chrom").cast(str), (-pl.col("llr_calibrated")).alias("gpn")]
        )
        .select([*KEY, "gpn"])
    )
    rows = []
    gpn_auprc = None
    for sdir, params in LADDER:
        m = (
            pl.read_parquet(f"{SCORES}/{sdir}/mendelian_traits.parquet")
            .filter(pl.col("subset") == "missense_variant")
            .with_columns(
                pl.col("chrom").cast(str),
                (-(pl.col("llr_fwd") + pl.col("llr_rc")) / 2).alias("ours"),
            )
            .join(gpn, on=KEY, how="inner")
            .drop_nulls(["ours", "gpn"])
        )
        y = m["label"].to_numpy()
        o = m["ours"].to_numpy()
        g = m["gpn"].to_numpy()
        ens = _z(o) + _z(g)
        if gpn_auprc is None:
            gpn_auprc = float(ap(y, g))  # GPN-Star is model-independent -> constant
        rows.append(
            {
                "params": params,
                "ours": float(ap(y, o)),
                "ensemble": float(ap(y, ens)),
                "gpn": gpn_auprc,
                "n": len(y),
                "prev": float(y.mean()),
            }
        )
        print(
            f"{params:>5}M | ours={rows[-1]['ours']:.3f}  ensemble(ours+GPN)={rows[-1]['ensemble']:.3f}  (GPN alone={gpn_auprc:.3f})"
        )

    res = pl.DataFrame(rows)
    res.write_parquet("scratch/issue302/ensemble_fix.parquet")
    c = res.to_pandas()
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.plot(
        c["params"],
        c["ours"],
        "o-",
        color="tab:red",
        lw=2.5,
        label="our gLM alone (minus_llr) — degrades",
    )
    ax.plot(
        c["params"],
        c["ensemble"],
        "D-",
        color="tab:purple",
        lw=2.5,
        label="our gLM + GPN-Star (z-ensemble)",
    )
    ax.axhline(
        gpn_auprc,
        ls="--",
        color="tab:green",
        lw=2,
        label=f"GPN-Star alone ({gpn_auprc:.3f}, model-independent)",
    )
    ax.axvline(128, ls=":", color="gray", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("our-model params (M, log)")
    ax.set_ylabel("Mendelian missense AUPRC")
    o_drop = c["ours"].max() - c["ours"].iloc[-1]
    e_drop = c["ensemble"].max() - c["ensemble"].iloc[-1]
    ax.set_title(
        f"Our single-seq gLM is NOT additive to MSA on missense:\na 50/50 ensemble stays BELOW GPN-Star alone at every scale and still degrades\n(ours peak→4B {o_drop:+.3f} · ensemble {e_drop:+.3f} · GPN-Star alone dominates)"
    )
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "ensemble_fix.png", dpi=130, bbox_inches="tight")
    fig.savefig(OUT / "ensemble_fix.svg", bbox_inches="tight")
    plt.close(fig)
    print(
        f"  wrote {OUT / 'ensemble_fix'}  | ours drop={o_drop:+.3f} ensemble drop={e_drop:+.3f}"
    )

    import s3fs

    fs = s3fs.S3FileSystem()
    for p in (str(OUT / "ensemble_fix.png"), str(OUT / "ensemble_fix.svg")):
        fs.put(p, f"{OUT_S3.replace('s3://', '')}/{Path(p).name}")
    print(f"  uploaded -> {OUT_S3}")


if __name__ == "__main__":
    main()
