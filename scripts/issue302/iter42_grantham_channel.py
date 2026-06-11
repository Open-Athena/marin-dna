"""issue #302 — iteration 42: is the amino-acid-severity (Grantham) channel an INDEPENDENT driver
of the missense degradation, or also subsumed by conservation? iter5 found a Grantham channel that
"turns on at scale" (partial Spearman(benign score, Grantham | phyloP) ~0.07@128M -> ~0.21@4B). Two
tests, parallel to iter41:
  (1) reproduce the scale-onset across the full ladder (per-model partial Spearman, Grantham | phyloP_241m);
  (2) does Grantham independently predict the per-benign recruitment Δ = z(mll@4B) − z(mll@128M),
      controlling for position conservation (phyloP_241m) and mammal-specificity?

Grantham from cached aa_ref/aa_alt (scratch/issue302/myvariant_aa.parquet) via iter5's validated
property formula. Titles filled from the data (no pre-judged conclusions). CPU; reads S3 + cache.
Run: OMP_NUM_THREADS=2 uv run --group genome-s3 python scripts/issue302/iter42_grantham_channel.py
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
LADDER = [
    ("h640-p46M", 46),
    ("h768-p76M", 76),
    ("h896-p128M", 128),
    ("h1152-p255M", 255),
    ("h1408-p476M", 476),
    ("h1920-p1B", 1120),
    ("h2432-p2B", 2270),
    ("h2944-p4B", 4020),
]
GP = {
    "A": (0.0, 8.1, 31),
    "R": (0.65, 10.5, 124),
    "N": (1.33, 11.6, 56),
    "D": (1.38, 13.0, 54),
    "C": (2.75, 5.5, 55),
    "Q": (0.89, 10.5, 85),
    "E": (0.92, 12.3, 83),
    "G": (0.74, 9.0, 3),
    "H": (0.58, 10.4, 96),
    "I": (0.0, 5.2, 111),
    "L": (0.0, 4.9, 111),
    "K": (0.33, 11.3, 119),
    "M": (0.0, 5.7, 105),
    "F": (0.0, 5.0, 132),
    "P": (0.39, 8.0, 32.5),
    "S": (1.42, 9.2, 32),
    "T": (0.71, 8.6, 61),
    "W": (0.13, 5.4, 170),
    "Y": (0.2, 6.2, 136),
    "V": (0.0, 5.9, 84),
}


def grantham(a, b):
    if a not in GP or b not in GP:
        return None
    if a == b:
        return 0.0
    (ca, pa, va), (cb, pb, vb) = GP[a], GP[b]
    return 50.723 * np.sqrt(
        1.833 * (ca - cb) ** 2 + 0.1018 * (pa - pb) ** 2 + 0.000399 * (va - vb) ** 2
    )


def _track(name):
    return (
        pl.read_parquet(f"{CB}/{name}_train.parquet")
        .with_columns(pl.col("chrom").cast(str))
        .unique(subset=KEY)
        .select([*KEY, pl.col("score").alias(name)])
    )


def _z(x):
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)


def _load(tag):
    return (
        pl.read_parquet(f"{SC}/scaling-v0.5-{tag}-step-215573/mendelian_traits.parquet")
        .filter(pl.col("subset") == "missense_variant")
        .with_columns(
            pl.col("chrom").cast(str),
            (-(pl.col("llr_fwd") + pl.col("llr_rc")) / 2).alias("mll"),
        )
        .select([*KEY, "label", "mll"])
    )


def partial_sp(y, x, ctrl):
    n = len(y)
    cx = np.column_stack([np.ones(n), _z(ctrl)])
    rx = x - cx @ np.linalg.lstsq(cx, x, rcond=None)[0]
    ry = y - cx @ np.linalg.lstsq(cx, y, rcond=None)[0]
    return spearmanr(rx, ry).statistic


def main() -> None:
    aa = (
        pl.read_parquet("scratch/issue302/myvariant_aa.parquet")
        .with_columns(pl.col("chrom").cast(str))
        .drop_nulls(["aa_ref", "aa_alt"])
    )
    gr = [grantham(r, a) for r, a in zip(aa["aa_ref"], aa["aa_alt"])]
    aa = (
        aa.with_columns(pl.Series("grantham", gr))
        .drop_nulls("grantham")
        .select([*KEY, "grantham"])
    )
    p241, p100 = _track("phyloP_241m"), _track("phyloP_100v")

    # (1) per-model partial Spearman(benign mll, grantham | phyloP_241m) across the ladder
    rows = []
    for tag, params in LADDER:
        m = (
            _load(tag)
            .join(aa, on=KEY, how="inner")
            .join(p241, on=KEY, how="left")
            .drop_nulls(["grantham", "phyloP_241m"])
            .filter(pl.col("label") == 0)
        )
        mll = m["mll"].to_numpy()
        g = m["grantham"].to_numpy()
        ps = partial_sp(_z(mll), _z(g), m["phyloP_241m"].to_numpy())
        raw = spearmanr(mll, g).statistic
        rows.append({"params": params, "partial": float(ps), "raw": float(raw)})
        print(
            f"{params:>5}M | partial Spearman(benign mll, grantham | phyloP_241m)={ps:+.3f}  (raw {raw:+.3f})"
        )

    # (2) recruitment regression with grantham
    a = _load("h896-p128M").with_columns(
        ((pl.col("mll") - pl.col("mll").mean()) / pl.col("mll").std()).alias("z128")
    )
    b = (
        _load("h2944-p4B")
        .with_columns(
            ((pl.col("mll") - pl.col("mll").mean()) / pl.col("mll").std()).alias("z4b")
        )
        .select([*KEY, "z4b"])
    )
    m = (
        a.join(b, on=KEY, how="inner")
        .join(aa, on=KEY, how="left")
        .join(p241, on=KEY, how="left")
        .join(p100, on=KEY, how="left")
        .filter(pl.col("label") == 0)
        .drop_nulls(["grantham", "phyloP_241m", "phyloP_100v", "z128", "z4b"])
    )
    n = m.height
    delta = (m["z4b"] - m["z128"]).to_numpy()
    g = m["grantham"].to_numpy()
    spec = _z(m["phyloP_241m"].to_numpy()) - _z(m["phyloP_100v"].to_numpy())
    p241v = m["phyloP_241m"].to_numpy()
    X1 = np.column_stack([np.ones(n), _z(p241v), _z(spec), _z(g)])
    coef, *_ = np.linalg.lstsq(X1, _z(delta), rcond=None)
    resid = _z(delta) - X1 @ coef
    se = np.sqrt(
        np.diag(((resid @ resid) / (n - X1.shape[1])) * np.linalg.inv(X1.T @ X1))
    )
    names = ["intercept", "phyloP_241m", "mammal-specificity", "grantham"]
    print(f"\n  recruitment Δ regression with grantham (n={n}):")
    for nm, c, s in zip(names, coef, se):
        print(
            f"    {nm:<20} β={c:+.3f} (z={c / s:+.1f}{'  *' if abs(c / s) > 2 else ''})"
        )
    uni_g = spearmanr(delta, g).statistic
    par_g = partial_sp(_z(delta), _z(g), p241v)
    print(
        f"    univariate Spearman(Δ, grantham)={uni_g:+.3f} ; partial(|phyloP_241m)={par_g:+.3f}"
    )

    res = pl.DataFrame(rows)
    res.write_parquet("scratch/issue302/grantham_channel.parquet")
    c = res.to_pandas()
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(
        c["params"],
        c["partial"],
        "o-",
        color="tab:purple",
        lw=2.5,
        label="partial | phyloP_241m",
    )
    ax[0].plot(c["params"], c["raw"], "s--", color="gray", lw=1.8, label="raw")
    ax[0].axvline(128, ls=":", color="gray", lw=1)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("params (M, log)")
    ax[0].set_ylabel("Spearman(benign mll, Grantham)")
    ax[0].set_title(
        f"(1) The AA-severity channel across scale\npartial(|phyloP) {c['partial'].iloc[2]:+.2f}@128M → {c['partial'].iloc[-1]:+.2f}@4B"
    )
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    bx = np.arange(3)
    ax[1].bar(bx, coef[1:], color=["tab:red", "tab:orange", "tab:purple"])
    for i, (cc, s) in enumerate(zip(coef[1:], se[1:])):
        ax[1].text(
            i,
            cc + 0.004 * np.sign(cc),
            f"{cc:+.3f}\nz={cc / s:+.1f}",
            ha="center",
            va="bottom" if cc > 0 else "top",
            fontsize=8,
        )
    ax[1].set_xticks(bx)
    ax[1].set_xticklabels(
        ["phyloP_241m", "mammal-\nspecificity", "grantham"], fontsize=8
    )
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_ylabel("standardized β (recruitment Δ)")
    ax[1].set_title(
        f"(2) Does Grantham drive recruitment independently?\njoint β grantham={coef[3]:+.3f} (z={coef[3] / se[3]:+.1f}); univariate {uni_g:+.2f}"
    )
    ax[1].grid(alpha=0.3, axis="y")
    fig.suptitle(
        "The amino-acid-severity (Grantham) channel: scale-onset + whether it independently drives the recruitment",
        y=1.02,
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT / "grantham_channel.png", dpi=130, bbox_inches="tight")
    fig.savefig(OUT / "grantham_channel.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / 'grantham_channel'}")

    import s3fs

    fs = s3fs.S3FileSystem()
    for p in (str(OUT / "grantham_channel.png"), str(OUT / "grantham_channel.svg")):
        fs.put(p, f"{OUT_S3.replace('s3://', '')}/{Path(p).name}")
    print(f"  uploaded -> {OUT_S3}")


if __name__ == "__main__":
    main()
