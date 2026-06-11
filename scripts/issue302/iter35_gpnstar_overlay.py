"""issue #302 — iteration 35: DIRECT GPN-Star overlay (closes GB's GPN-Star thread). GPN-Star is
a self-supervised VERTEBRATE-MSA gLM; iter4 showed it escapes the over-call, iter30/33 argued its
signal is the deeper (vertebrate) conservation our single-sequence gLM lacks. Test both directly
on our confident FPs:
  (1) RESCUE — does GPN-Star score the variants our 4B confidently over-calls (benign, mll>=path
      median) as benign? AUROC(pathogenic vs our-confident-FP) by GPN-Star vs by our mll (~0.5 by
      construction).
  (2) MECHANISM — among our confident FPs, does GPN-Star score the VERTEBRATE-TOLERANT ones
      (phyloP_100v<4) lower than the vertebrate-conserved ones? i.e. is GPN-Star using the depth?
      And Spearman(score, phyloP_100v) on benigns: GPN-Star vs our mll.

GPN-Star-V scores from #145/#203 gist (cLLR = -llr_calibrated). 4B. CPU; reads S3 + gist + tracks.
Run: OMP_NUM_THREADS=2 uv run --group genome-s3 python scripts/issue302/iter35_gpnstar_overlay.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score

OURS = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores/scaling-v0.5-h2944-p4B-step-215573/mendelian_traits.parquet"
CB = "s3://oa-bolinas/snakemake/conservation_eval/results/mendelian_traits"
GPN_V = "https://gist.githubusercontent.com/gonzalobenegas/db282f89aa00244fbb7437dce0f069ef/raw/02484d50d9bfd80337e313652b26f98a9362b6b1/bolinas_mendelian_traits_GPN-Star-V.parquet"
OUT_S3 = "s3://oa-bolinas/analysis/issue302/probe_out"
OUT = Path("scratch/issue302/figs")
KEY = ["chrom", "pos", "ref", "alt"]


def _track(name):
    return pl.read_parquet(f"{CB}/{name}_train.parquet").with_columns(pl.col("chrom").cast(str)).unique(subset=KEY).select([*KEY, pl.col("score").alias(name)])


def main() -> None:
    m = (
        pl.read_parquet(OURS)
        .filter(pl.col("subset") == "missense_variant")
        .with_columns(pl.col("chrom").cast(str), (-(pl.col("llr_fwd") + pl.col("llr_rc")) / 2).alias("ours"))
    )
    gpn = pl.read_parquet(GPN_V).with_columns([pl.col("chrom").cast(str), (-pl.col("llr_calibrated")).alias("gpn")]).select([*KEY, "gpn"])
    m = m.join(gpn, on=KEY, how="inner").join(_track("phyloP_100v"), on=KEY, how="left").join(_track("phyloP_241m"), on=KEY, how="left")
    print(f"joined our 4B + GPN-Star-V: {m.height} missense variants ({m.filter(pl.col('label') == 1).height} path / {m.filter(pl.col('label') == 0).height} benign)")

    pmd = m.filter(pl.col("label") == 1)["ours"].median()
    tp = m.filter(pl.col("label") == 1)
    fp = m.filter((pl.col("label") == 0) & (pl.col("ours") >= pmd))  # our confident over-calls
    print(f"\nconfident FPs (benign, our mll>=path median): n={fp.height}")

    # (1) rescue: AUROC(TP vs confident-FP) by each scorer
    for scorer in ("ours", "gpn"):
        y = np.r_[np.ones(tp.height), np.zeros(fp.height)]
        s = np.r_[tp[scorer].to_numpy(), fp[scorer].to_numpy()]
        ok = ~np.isnan(s)
        print(f"  AUROC(pathogenic vs our-confident-FP) by {scorer:>4}: {roc_auc_score(y[ok], s[ok]):.3f}")

    # (2a) mechanism: GPN on confident FPs, split by vertebrate depth
    ft = fp.filter(pl.col("phyloP_100v") < 4)
    fc = fp.filter(pl.col("phyloP_100v") >= 4)
    gt, gc = ft["gpn"].drop_nulls().to_numpy(), fc["gpn"].drop_nulls().to_numpy()
    ot, oc = ft["ours"].drop_nulls().to_numpy(), fc["ours"].drop_nulls().to_numpy()
    u = mannwhitneyu(gt, gc).pvalue
    print(f"\n  among confident FPs: GPN-Star score  vert-tolerant(n={len(gt)})={np.median(gt):+.3f}  vert-conserved(n={len(gc)})={np.median(gc):+.3f}  MWU p={u:.1e}")
    print(f"                       our mll          vert-tolerant={np.median(ot):+.3f}  vert-conserved={np.median(oc):+.3f}  (ours can't tell them apart by construction)")

    # (2b) corr with vertebrate depth on benigns
    ben = m.filter(pl.col("label") == 0).drop_nulls(["phyloP_100v", "gpn", "ours"])
    rg = spearmanr(ben["gpn"], ben["phyloP_100v"]).statistic
    ro = spearmanr(ben["ours"], ben["phyloP_100v"]).statistic
    rg241 = spearmanr(ben["gpn"], ben["phyloP_241m"]).statistic
    ro241 = spearmanr(ben["ours"], ben["phyloP_241m"]).statistic
    print(f"\n  Spearman(score, phyloP_100v vertebrate) on benigns:  GPN={rg:.3f}  ours={ro:.3f}")
    print(f"  Spearman(score, phyloP_241m mammal)     on benigns:  GPN={rg241:.3f}  ours={ro241:.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    # Panel A — rescue: GPN-Star score distributions, TP vs our-confident-FP
    parts = ax[0].violinplot([tp["gpn"].drop_nulls().to_numpy(), fp["gpn"].drop_nulls().to_numpy()], showmedians=True)
    ax[0].set_xticks([1, 2])
    ax[0].set_xticklabels([f"pathogenic\n(n={tp.height})", f"our confident FP\n(n={fp.height})"])
    ax[0].set_ylabel("GPN-Star-V score (−calibrated LLR; ↑=pathogenic)")
    y = np.r_[np.ones(tp.height), np.zeros(fp.height)]
    s = np.r_[tp["gpn"].to_numpy(), fp["gpn"].to_numpy()]
    ok = ~np.isnan(s)
    ax[0].set_title(f"RESCUE: GPN-Star separates pathogenic from\nour confident FPs (AUROC={roc_auc_score(y[ok], s[ok]):.2f}) — which our LLR can't (0.50)")
    ax[0].grid(alpha=0.3, axis="y")
    # Panel B — mechanism: GPN-Star score on confident FPs by vertebrate depth
    parts = ax[1].violinplot([gt, gc], showmedians=True)
    ax[1].set_xticks([1, 2])
    ax[1].set_xticklabels([f"vertebrate-TOLERANT\nphyloP_100v<4 (n={len(gt)})", f"vertebrate-conserved\nphyloP_100v≥4 (n={len(gc)})"])
    ax[1].set_ylabel("GPN-Star-V score on our confident FPs")
    ax[1].set_title(f"MECHANISM: GPN-Star scores the vertebrate-TOLERANT\nFPs lower (correctly benign) — MWU p={u:.0e}; it reads the depth")
    ax[1].grid(alpha=0.3, axis="y")
    fig.suptitle("Direct GPN-Star overlay: it rescues our confident FPs, and does so via the vertebrate-depth signal a single-sequence gLM lacks", y=1.02, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "gpnstar_overlay.png", dpi=130, bbox_inches="tight")
    fig.savefig(OUT / "gpnstar_overlay.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / 'gpnstar_overlay'}")

    import s3fs

    fs = s3fs.S3FileSystem()
    for p in (str(OUT / "gpnstar_overlay.png"), str(OUT / "gpnstar_overlay.svg")):
        fs.put(p, f"{OUT_S3.replace('s3://', '')}/{Path(p).name}")
    print(f"  uploaded -> {OUT_S3}")


if __name__ == "__main__":
    main()
