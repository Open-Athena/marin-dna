"""issue #314 iter1 — parsimony slice: within the *good* recipe region, does each
extra knob earn its keep?

A good protocol should be as simple as possible. From the global attribution the
winning region is window pooling (center/entire_window) × a difference combo
(delta/sum_absdiff/abs_delta) × strong L2 (C<0.1). *Within that region* this asks,
from the already-saved random-search AUPRCs (no refit): does **center** beat
**entire_window** (center costs an extra width hyperparameter)? does **PCA** beat
**no-PCA**? does the **StandardScaler** beat no-scaler? Each as a subset-centered
marginal (mean over the other good-region choices and all subsets), so a knob that
doesn't move AUPRC can be dropped for free.

Run:
  uv run --group genome-s3 python scripts/issue314/iter1_parsimony.py \
    --in s3://oa-bolinas/analysis/issue314/iter1_search/exp135-1B-m5.1 --model exp135-1B-m5.1
"""

import argparse

import polars as pl

GOOD_POOL = ["center", "entire_window"]
GOOD_COMBO = ["delta", "sum_absdiff", "abs_delta"]


def _marg(df: pl.DataFrame, factor: str) -> pl.DataFrame:
    ce = df.with_columns(
        (pl.col("auprc") - pl.col("auprc").mean().over("subset")).alias("ce")
    )
    return (
        ce.group_by(factor)
        .agg(
            pl.col("ce").mean().alias("mean"),
            (pl.col("ce").std() / pl.len().sqrt()).alias("se"),
            pl.len().alias("n"),
        )
        .sort("mean", descending=True)
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    df = pl.read_parquet(f"{args.inp}/auprc.parquet").with_columns(
        pl.when(pl.col("n_pca") == 0)
        .then(pl.lit("none"))
        .otherwise(pl.col("n_pca").cast(pl.Utf8))
        .alias("n_pca")
    )
    good = df.filter(
        (pl.col("rep_kind") == "pool")
        & pl.col("pooling").is_in(GOOD_POOL)
        & pl.col("combo").is_in(GOOD_COMBO)
        & (pl.col("c") < 0.1)
    )
    print(f"\n########## {args.model} — parsimony within good region "
          f"(window pool × diff combo × C<0.1, {len(good)} cells) ##########")
    # has_pca collapses 8/32/64/128/256 → "PCA" vs "none" for the binary question
    good = good.with_columns(
        pl.when(pl.col("n_pca") == "none").then(pl.lit("no-PCA"))
        .otherwise(pl.lit("PCA")).alias("has_pca")
    )
    for fac in ["pooling", "has_pca", "n_pca", "scaler", "combo"]:
        g = _marg(good, fac)
        spread = float(g["mean"].max() - g["mean"].min())
        print(f"\n  {fac}  (spread = {spread:.3f}):")
        for r in g.iter_rows(named=True):
            print(f"    {str(r[fac]):14s} {r['mean']:+.3f} ± {r['se']:.3f}  (n={r['n']})")

    # simplest recipe within good region: entire_window + no-PCA, best combo/scaler by avg
    simplest = good.filter(
        (pl.col("pooling") == "entire_window") & (pl.col("has_pca") == "no-PCA")
    )
    if len(simplest):
        ce = simplest.with_columns(
            (pl.col("auprc") - pl.col("auprc").mean().over("subset")).alias("ce")
        )
        top = (
            ce.group_by("combo", "scaler")
            .agg(pl.col("ce").mean().alias("mce"), pl.len().alias("n"))
            .sort("mce", descending=True)
        )
        print("\n  simplest family (entire_window + no-PCA) — combo×scaler by centered ΔAUPRC:")
        for r in top.iter_rows(named=True):
            print(f"    {r['combo']:12s} scaler={int(r['scaler'])}  {r['mce']:+.3f}  (n={r['n']})")


if __name__ == "__main__":
    main()
