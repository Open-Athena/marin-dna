"""issue #302 — iteration 41: reconcile the EARLY gene-age clue (iter3: degradation concentrates
in OLD/deeply-conserved genes) with the LATE position-depth clue (iter33/39: degradation is the
VERTEBRATE-TOLERANT/shallow-position benigns). Tension or independent axes?

Per benign missense variant, define the recruitment Δ = z(mll_4B) − z(mll_128M) (how much MORE the
benign is over-called at 4B than at the 128M peak). Regress Δ on gene-age (Liebeskind modeAge, MYA)
and position mammal-specificity (z(phyloP_241m) − z(phyloP_100v)). Questions:
  (1) Are gene-age and mammal-specificity correlated (is iter3 just a proxy for iter33)?
  (2) In a joint standardized regression, do BOTH independently predict the recruitment, or does one
      subsume the other? FINDING: gene-age does NOT independently predict variant-level recruitment
      (univariate -0.003, partial -0.018); it collapses once position conservation (phyloP_241m) is
      controlled. So iter3's old-gene effect is a CONFOUND/proxy for position conservation, not a
      distinct gene-level memorization axis. The recruitment is position-level (phyloP_241m + mammal-spec).

Gene age cached at scratch/issue302/ensg_to_age.parquet (join on exon_closest_pc_gene_id). CPU; S3.
Run: OMP_NUM_THREADS=2 uv run --group genome-s3 python scripts/issue302/iter41_age_vs_depth.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import spearmanr

SC = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores"
CB = "s3://oa-bolinas/snakemake/conservation_eval/results/mendelian_traits"
OUT_S3 = "s3://oa-bolinas/analysis/issue302/probe_out"
OUT = Path("scratch/issue302/figs")
KEY = ["chrom", "pos", "ref", "alt"]


def _track(name):
    return pl.read_parquet(f"{CB}/{name}_train.parquet").with_columns(pl.col("chrom").cast(str)).unique(subset=KEY).select([*KEY, pl.col("score").alias(name)])


def _z(x):
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)


def _load(tag):
    return (
        pl.read_parquet(f"{SC}/scaling-v0.5-{tag}-step-215573/mendelian_traits.parquet")
        .filter(pl.col("subset") == "missense_variant")
        .with_columns(pl.col("chrom").cast(str), (-(pl.col("llr_fwd") + pl.col("llr_rc")) / 2).alias("mll"))
    )


def _zmll(tag):
    m = _load(tag)
    return m.with_columns(((pl.col("mll") - pl.col("mll").mean()) / pl.col("mll").std()).alias(f"z_{tag}")).select([*KEY, "label", f"z_{tag}", "exon_closest_pc_gene_id"])


def main() -> None:
    a = _zmll("h896-p128M")
    b = _zmll("h2944-p4B")
    age = pl.read_parquet("scratch/issue302/ensg_to_age.parquet").select([pl.col("ensg").alias("exon_closest_pc_gene_id"), "age_mya"])
    m = (
        a.join(b.select([*KEY, "z_h2944-p4B"]), on=KEY, how="inner")
        .join(_track("phyloP_241m"), on=KEY, how="left")
        .join(_track("phyloP_100v"), on=KEY, how="left")
        .join(age, on="exon_closest_pc_gene_id", how="left")
        .filter(pl.col("label") == 0)
        .drop_nulls(["phyloP_241m", "phyloP_100v", "age_mya", "z_h896-p128M", "z_h2944-p4B"])
    )
    n = m.height
    delta = (m["z_h2944-p4B"] - m["z_h896-p128M"]).to_numpy()  # recruitment with scale
    gene_age = np.log10(m["age_mya"].to_numpy())
    spec = _z(m["phyloP_241m"].to_numpy()) - _z(m["phyloP_100v"].to_numpy())
    p241 = m["phyloP_241m"].to_numpy()
    print(f"benign missense with all features: n={n}")

    # (1) are the two axes correlated?
    r_axes = spearmanr(gene_age, spec).statistic
    r_age_241 = spearmanr(gene_age, p241).statistic
    print(f"\n(1) Spearman(gene-age, mammal-specificity) = {r_axes:+.3f}; Spearman(gene-age, raw phyloP_241m) = {r_age_241:+.3f} (old genes ~ more conserved positions => the proxy)")

    # univariate Spearman of recruitment with each
    r_age = spearmanr(delta, gene_age).statistic
    r_spec = spearmanr(delta, spec).statistic
    r_241 = spearmanr(delta, p241).statistic
    print(f"\n(2) recruitment Δ (z_4B − z_128M) univariate Spearman:")
    print(f"      vs gene-age          = {r_age:+.3f}")
    print(f"      vs mammal-specificity= {r_spec:+.3f}")
    print(f"      vs raw phyloP_241m   = {r_241:+.3f}")

    # joint standardized OLS: which survive?
    X = np.column_stack([_z(gene_age), _z(spec), _z(p241)])
    X1 = np.column_stack([np.ones(n), X])
    coef, *_ = np.linalg.lstsq(X1, _z(delta), rcond=None)
    # std errors
    resid = _z(delta) - X1 @ coef
    sigma2 = (resid @ resid) / (n - X1.shape[1])
    cov = sigma2 * np.linalg.inv(X1.T @ X1)
    se = np.sqrt(np.diag(cov))
    names = ["intercept", "gene-age", "mammal-specificity", "raw phyloP_241m"]
    print(f"\n    joint standardized OLS (Δ ~ gene-age + mammal-specificity + phyloP_241m), n={n}:")
    for nm, c, s in zip(names, coef, se):
        z = c / s
        print(f"      {nm:<20} β={c:+.3f}  (SE {s:.3f}, z={z:+.1f}{'  *' if abs(z) > 2 else ''})")

    # partial: gene-age controlling for mammal-specificity (+241)
    def partial(y, x, ctrl):
        cx = np.column_stack([np.ones(n)] + [_z(c) for c in ctrl])
        rx = x - cx @ np.linalg.lstsq(cx, x, rcond=None)[0]
        ry = y - cx @ np.linalg.lstsq(cx, y, rcond=None)[0]
        return spearmanr(rx, ry).statistic
    pr_age = partial(_z(delta), _z(gene_age), [spec, p241])
    pr_spec = partial(_z(delta), _z(spec), [gene_age, p241])
    print(f"\n    partial Spearman(Δ, gene-age | mammal-spec, phyloP241) = {pr_age:+.3f}")
    print(f"    partial Spearman(Δ, mammal-spec | gene-age, phyloP241) = {pr_spec:+.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    # Panel A — recruitment vs DISCRETE gene-age category (modeAge maps to fixed MYA values)
    amya = m["age_mya"].to_numpy()
    cats = sorted(set(amya.tolist()))
    means = [float(np.mean(delta[amya == a])) for a in cats]
    ncat = [int(np.sum(amya == a)) for a in cats]
    ax[0].bar(range(len(cats)), means, color="tab:brown")
    ax[0].set_xticks(range(len(cats)))
    ax[0].set_xticklabels([f"{a:.0f}\n(n={nc})" for a, nc in zip(cats, ncat)], fontsize=7)
    ax[0].set_xlabel("gene age (MYA, discrete modeAge → older →)")
    ax[0].set_ylabel("benign recruitment Δ (z_4B − z_128M)")
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_title(f"Gene-age does NOT predict variant-level recruitment\n(Spearman {r_age:+.3f}, flat) — iter3's old-gene effect is a confound")
    ax[0].grid(alpha=0.3, axis="y")
    # Panel B — the two axes are independent + both survive
    labels = ["gene-age", "mammal-\nspecificity", "raw\nphyloP_241m"]
    bx = np.arange(3)
    ax[1].bar(bx - 0.2, [r_age, r_spec, r_241], 0.4, label="univariate", color="lightsteelblue")
    ax[1].bar(bx + 0.2, [pr_age, pr_spec, np.nan], 0.4, label="partial (others controlled)", color="tab:blue")
    ax[1].set_xticks(bx)
    ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_ylabel("Spearman with recruitment Δ")
    ax[1].set_title(f"Recruitment is POSITION-level: phyloP_241m + mammal-spec\nsurvive; gene-age collapses to ~0 when position is controlled")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, axis="y")
    fig.suptitle("Resolving iter3 ↔ iter33: the scale recruitment is driven by POSITION conservation depth, NOT gene age — iter3's old-gene effect was a proxy for position-level conservation", y=1.02, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(OUT / "age_vs_depth.png", dpi=130, bbox_inches="tight")
    fig.savefig(OUT / "age_vs_depth.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / 'age_vs_depth'}")

    import s3fs
    fs = s3fs.S3FileSystem()
    for p in (str(OUT / "age_vs_depth.png"), str(OUT / "age_vs_depth.svg")):
        fs.put(p, f"{OUT_S3.replace('s3://', '')}/{Path(p).name}")
    print(f"  uploaded -> {OUT_S3}")


if __name__ == "__main__":
    main()
