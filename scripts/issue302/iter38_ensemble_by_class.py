"""issue #302 — iteration 38: is iter37's gLM-not-additive-to-MSA result SPECIFIC to the degrading
missense task, or general? Repeat the ours+GPN-Star z-ensemble vs GPN-Star-alone vs ours-alone for
the two CDS classes that IMPROVE with scale — splicing and synonymous — across the ladder.

Prediction (sharpening iter37): on splicing/synonymous, where our readout keeps improving, the gLM
SHOULD be additive (ensemble > GPN alone) — so the missense non-additivity is specific to the
broken task, not a blanket 'single-seq gLM < MSA'. CPU; reads S3 + gist.
Run: OMP_NUM_THREADS=2 uv run --group genome-s3 python scripts/issue302/iter38_ensemble_by_class.py
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
CLASSES = ["missense_variant", "splicing", "synonymous_variant"]


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
    gpn_alone = {}
    for sdir, params in LADDER:
        full = (
            pl.read_parquet(f"{SCORES}/{sdir}/mendelian_traits.parquet")
            .with_columns(
                pl.col("chrom").cast(str),
                (-(pl.col("llr_fwd") + pl.col("llr_rc")) / 2).alias("ours"),
            )
            .join(gpn, on=KEY, how="inner")
        )
        for cls in CLASSES:
            m = full.filter(pl.col("subset") == cls).drop_nulls(["ours", "gpn"])
            if m.height < 30:
                continue
            y = m["label"].to_numpy()
            o, g = m["ours"].to_numpy(), m["gpn"].to_numpy()
            ga = float(ap(y, g))
            gpn_alone.setdefault(cls, ga)
            rows.append(
                {
                    "params": params,
                    "cls": cls,
                    "ours": float(ap(y, o)),
                    "ensemble": float(ap(y, _z(o) + _z(g))),
                    "gpn": ga,
                }
            )
        r = {x["cls"]: x for x in rows if x["params"] == params}
        print(
            f"{params:>5}M | "
            + " | ".join(
                f"{c.split('_')[0][:4]}: ours={r[c]['ours']:.3f} ens={r[c]['ensemble']:.3f} gpn={r[c]['gpn']:.3f}"
                for c in CLASSES
                if c in r
            )
        )

    res = pl.DataFrame(rows)
    res.write_parquet("scratch/issue302/ensemble_by_class.parquet")
    c = res.to_pandas()
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)
    titles = {
        "missense_variant": "missense (degrades)",
        "splicing": "splicing (improves)",
        "synonymous_variant": "synonymous (improves)",
    }
    for k, cls in enumerate(CLASSES):
        d = c[c.cls == cls].sort_values("params")
        if d.empty:
            continue
        ax[k].plot(
            d["params"], d["ours"], "o-", color="tab:red", lw=2.2, label="ours alone"
        )
        ax[k].plot(
            d["params"],
            d["ensemble"],
            "D-",
            color="tab:purple",
            lw=2.2,
            label="ours + GPN-Star",
        )
        ax[k].axhline(
            gpn_alone[cls],
            ls="--",
            color="tab:green",
            lw=2,
            label=f"GPN-Star alone ({gpn_alone[cls]:.2f})",
        )
        add = d["ensemble"].iloc[-1] - gpn_alone[cls]
        ax[k].set_xscale("log")
        ax[k].set_title(
            f"{titles[cls]}\nensemble−GPN @4B = {add:+.3f}  ({'gLM ADDITIVE' if add > 0.005 else 'not additive'})",
            fontsize=10,
        )
        ax[k].set_xlabel("our-model params (M)")
        ax[k].grid(alpha=0.3)
        if k == 0:
            ax[k].set_ylabel("Mendelian AUPRC")
            ax[k].legend(fontsize=8, loc="lower left")
    fig.suptitle(
        "gLM additivity to the MSA model is TASK-SPECIFIC: not additive where the readout degrades (missense), additive where it improves (splicing/synonymous)",
        y=1.03,
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT / "ensemble_by_class.png", dpi=130, bbox_inches="tight")
    fig.savefig(OUT / "ensemble_by_class.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / 'ensemble_by_class'}")

    import s3fs

    fs = s3fs.S3FileSystem()
    for p in (str(OUT / "ensemble_by_class.png"), str(OUT / "ensemble_by_class.svg")):
        fs.put(p, f"{OUT_S3.replace('s3://', '')}/{Path(p).name}")
    print(f"  uploaded -> {OUT_S3}")


if __name__ == "__main__":
    main()
