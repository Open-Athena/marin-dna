"""issue #302 — iteration 21: are the confidently-over-called benigns (FPs) in RECENTLY-
RELAXED regions — constrained in mammals but de-constrained on the human/primate lineage
(GB's hypothesis, HAR-style)? Or are they conserved all the way down to primates and merely
tolerated within humans?

Test with the conservation_eval tracks at different clade depths:
  mammal:    phyloP_241m / phyloP_470m, phastCons_470m
  vertebrate (DEEPER, older): phyloP_100v
  PRIMATE (the recent-lineage test): phastCons_43p   (only primate track we have; phastCons,
                                                       element-based, saturates -> insensitive
                                                       to per-base recent relaxation)
Groups within the high-LLR (confident) set: TP (true pathogenic), FP (over-called benign);
plus B0 (correctly-low benign) as a floor. Score = minus_llr_avg, 4B. Reads/writes S3. CPU.

Run: OMP_NUM_THREADS=2 uv run --group genome-s3 python scripts/issue302/iter21_conservation_depth.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

KEY = ["chrom", "pos", "ref", "alt"]
CB = "s3://oa-bolinas/snakemake/conservation_eval/results/mendelian_traits"
S = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores/scaling-v0.5-h2944-p4B-step-215573/mendelian_traits.parquet"
OUT_S3 = "s3://oa-bolinas/analysis/issue302/probe_out"
OUT = Path("scratch/issue302/figs")
TRACKS = ["phyloP_241m", "phyloP_470m", "phyloP_100v", "phastCons_470m", "phastCons_43p"]


def _cons(name):
    d = pl.read_parquet(f"{CB}/{name}_train.parquet").with_columns(pl.col("chrom").cast(str)).unique(subset=KEY)
    return d.select([*KEY, pl.col("score").alias(name)])


def main() -> None:
    m = pl.read_parquet(S).filter(pl.col("subset") == "missense_variant").with_columns(
        pl.col("chrom").cast(str), (-(pl.col("llr_fwd") + pl.col("llr_rc")) / 2).alias("mll")
    )
    for t in TRACKS:
        m = m.join(_cons(t), on=KEY, how="left")
    mm = m.to_pandas()
    posmed = np.median(mm.loc[mm.label == 1, "mll"])
    masks = {
        "TP\n(path, hi-LLR)": ((mm.label == 1) & (mm.mll >= posmed)).values,
        "FP\n(benign, hi-LLR)": ((mm.label == 0) & (mm.mll >= posmed)).values,
        "B0\n(benign, lo-LLR)": ((mm.label == 0) & (mm.mll < posmed)).values,
    }
    colors = {"TP\n(path, hi-LLR)": "tab:red", "FP\n(benign, hi-LLR)": "tab:orange", "B0\n(benign, lo-LLR)": "tab:blue"}

    # AUROC(TP vs FP) per track
    tp, fp = masks["TP\n(path, hi-LLR)"], masks["FP\n(benign, hi-LLR)"]
    aurocs = {}
    for t in TRACKS:
        sel = (tp | fp) & ~mm[t].isna().values
        aurocs[t] = roc_auc_score(mm.label.values[sel], mm[t].values[sel])
    print("AUROC(TP vs FP) by depth:", {t: round(v, 3) for t, v in aurocs.items()})
    for g, msk in masks.items():
        print(f"  {g.split(chr(10))[0]:>3} n={msk.sum():>4} | " + "  ".join(f"{t.split('_')[1]}={np.nanmedian(mm[t].values[msk]):.2f}" for t in TRACKS))

    rows = [{"group": g.replace("\n", " "), "track": t, "median": float(np.nanmedian(mm[t].values[msk]))} for g, msk in masks.items() for t in TRACKS]
    pl.DataFrame(rows).write_parquet("scratch/issue302/conservation_depth.parquet")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    gnames = list(masks)
    xg = np.arange(len(gnames))
    # Panel A — phyloP: mammal (470m) vs vertebrate (100v, DEEPER)
    w = 0.38
    for k, (t, lab) in enumerate([("phyloP_470m", "mammal (470sp)"), ("phyloP_100v", "vertebrate (100sp, deeper)")]):
        vals = [np.nanmedian(mm[t].values[masks[g]]) for g in gnames]
        ax[0].bar(xg + (k - 0.5) * w, vals, w, label=lab, color=["#1b9e77", "#7570b3"][k])
    ax[0].set_xticks(xg)
    ax[0].set_xticklabels(gnames, fontsize=8)
    ax[0].set_ylabel("median phyloP")
    ax[0].set_title(f"phyloP by depth — FPs drop at the DEEP (vertebrate) level\n(deep phyloP best separates TP vs FP: AUROC {aurocs['phyloP_100v']:.3f} > mammal {aurocs['phyloP_470m']:.3f})")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, axis="y")
    # Panel B — phastCons: mammal (470m) vs PRIMATE (43p) — the recent-relaxation test
    for k, (t, lab) in enumerate([("phastCons_470m", "mammal (470sp)"), ("phastCons_43p", "primate (43sp)")]):
        vals = [np.nanmedian(mm[t].values[masks[g]]) for g in gnames]
        ax[1].bar(xg + (k - 0.5) * w, vals, w, label=lab, color=["#1b9e77", "#d95f02"][k])
    ax[1].set_xticks(xg)
    ax[1].set_xticklabels(gnames, fontsize=8)
    ax[1].set_ylabel("median phastCons")
    ax[1].set_title("phastCons mammal vs PRIMATE — the FPs stay primate-conserved\n(NOT recently relaxed; primate separates TP/FP worst: AUROC %.3f)" % aurocs["phastCons_43p"])
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, axis="y")
    fig.suptitle("Are the over-called benigns in recently-relaxed regions? No — they stay conserved to the primate level; they're just less DEEPLY (vertebrate) conserved than true pathogenics", y=1.03, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "conservation_depth.png", dpi=130, bbox_inches="tight")
    fig.savefig(OUT / "conservation_depth.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / 'conservation_depth'}.{{png,svg}}")

    import s3fs

    fs = s3fs.S3FileSystem()
    for p in ("scratch/issue302/conservation_depth.parquet", str(OUT / "conservation_depth.png"), str(OUT / "conservation_depth.svg")):
        fs.put(p, f"{OUT_S3.replace('s3://', '')}/{Path(p).name}")
    print(f"  uploaded -> {OUT_S3}")


if __name__ == "__main__":
    main()
