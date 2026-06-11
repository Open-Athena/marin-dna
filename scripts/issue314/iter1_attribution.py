"""issue #314 iter1 — factorial attribution from the random-search results.

Reads the point-AUPRC table from ``iter1_sweep`` and reports, per subset, the
**marginal main-effect** of each design choice — pooling extent, ref/alt combo,
rep-kind, StandardScaler, PCA-k, C — *averaged over all other choices* (the
"averaged across all combinations" statement), each with the **spread** (how much
that choice moves AUPRC) so the factors rank by leverage. Plus the best configs.
Fast and re-runnable; operates on the saved AUPRCs, no re-fitting. (Paired
significance is a separate step on the saved OOF predictions.)

Run:
  uv run --group genome-s3 python scripts/issue314/iter1_attribution.py \
    --in s3://oa-bolinas/analysis/issue314/iter1_search/exp135-1B-m5.1
"""

import argparse

import polars as pl

C_EDGES = [-3, -1, 0, 1, 2]  # log10(C) bin edges: <0.1, 0.1–1, 1–10, 10–100


def _c_bin(df: pl.DataFrame) -> pl.DataFrame:
    lab = pl.col("c").log10()
    return df.with_columns(
        pl.when(lab < -1)
        .then(pl.lit("C<0.1"))
        .when(lab < 0)
        .then(pl.lit("C0.1-1"))
        .when(lab < 1)
        .then(pl.lit("C1-10"))
        .otherwise(pl.lit("C10-100"))
        .alias("c_bin")
    )


def _marginal(d: pl.DataFrame, factor: str) -> pl.DataFrame:
    return (
        d.group_by(factor)
        .agg(
            pl.col("auprc").mean().alias("mean"),
            (pl.col("auprc").std() / pl.len().sqrt()).alias("se"),
            pl.len().alias("n"),
        )
        .sort("mean", descending=True)
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True, help="iter1_sweep --out prefix")
    ap.add_argument("--subset", default=None, help="default: loop all subsets")
    args = ap.parse_args()

    df = _c_bin(pl.read_parquet(f"{args.inp}/auprc.parquet"))
    subsets = [args.subset] if args.subset else sorted(df["subset"].unique().to_list())
    # pool-only factors vs all-rep factors
    fac_scope = {
        "pooling": "pool",
        "combo": "pool",
        "rep_kind": "all",
        "scaler": "all",
        "n_pca": "all",
        "c_bin": "all",
    }

    leverage_rows = []
    for s in subsets:
        d = df.filter(pl.col("subset") == s)
        pool = d.filter(pl.col("rep_kind") == "pool")
        print(f"\n===== {s}  ({len(d)} cells) =====")
        print("top configs:")
        for r in d.sort("auprc", descending=True).head(5).iter_rows(named=True):
            print(
                f"  {r['rep']:24s} scaler={int(r['scaler'])} pca={r['n_pca']:>3} "
                f"C={r['c']:.3g}  AUPRC={r['auprc']:.3f}"
            )
        spreads = []
        for fac, scope in fac_scope.items():
            g = _marginal(pool if scope == "pool" else d, fac)
            spread = float(g["mean"].max() - g["mean"].min())
            spreads.append((fac, spread))
            print(f"\n  main effect — {fac}  (spread = {spread:.3f}):")
            for r in g.iter_rows(named=True):
                print(
                    f"    {str(r[fac]):16s} {r['mean']:.3f} ± {r['se']:.3f}  (n={r['n']})"
                )
        print("\n  factor leverage (by spread):")
        for fac, sp in sorted(spreads, key=lambda x: -x[1]):
            print(f"    {fac:10s} {sp:.3f}")
            leverage_rows.append({"subset": s, "factor": fac, "spread": sp})

    pl.DataFrame(leverage_rows).write_parquet(f"{args.inp}/factor_leverage.parquet")
    print(f"\nwrote {args.inp}/factor_leverage.parquet")


if __name__ == "__main__":
    main()
