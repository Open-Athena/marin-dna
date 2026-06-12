"""issue #314 iter1 — consolidate the random-search sweep into the post tables.

Two deliverables from the saved artifacts (no re-fitting):

1. **Per-subset best-probe vs zero-shot LLR** — for each Mendelian subset, take the
   single best config (max OOF point-AUPRC), then on its *saved OOF predictions*
   compute AUPRC ± cluster-bootstrap SE and the **paired** probe−LLR delta (resampling
   ``match_group``, two-sided p). The bootstrap was deferred out of the sweep loop; this
   runs it only on the 1 winning config/subset, so it's cheap.
2. **Global factorial attribution** — the user's "averaged across all the other
   combinations" question. AUPRC is **centered within each subset** (subtract that
   subset's grand mean over configs) so easy subsets don't dominate, then the marginal
   mean of each design choice (pooling extent, ref/alt combo, rep-kind, StandardScaler,
   PCA-k, C-bin) is taken across all configs and subsets. **Leverage** = spread
   (max−min) of a factor's level means → ranks which choices matter.

Run:
  uv run --group genome-s3 python scripts/issue314/iter1_consolidate.py \
    --in s3://oa-bolinas/analysis/issue314/iter1_search/exp135-1B-m5.1 --model exp135-1B-m5.1
"""

import argparse

import numpy as np
import polars as pl

from marin_dna.pipelines.evals.metrics import (
    auprc_with_bootstrap_se,
    paired_metric_delta_bootstrap,
)

POOL_FACTORS = ("pooling", "combo")  # defined only for rep_kind == "pool"
ALL_FACTORS = ("rep_kind", "scaler", "n_pca", "c_bin")


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


def _marginal_centered(df: pl.DataFrame, factor: str) -> pl.DataFrame:
    """Subset-centered marginal mean of ``factor`` (mean over all other choices and
    subsets, after removing each subset's grand mean)."""
    centered = df.with_columns(
        (pl.col("auprc") - pl.col("auprc").mean().over("subset")).alias("ce")
    )
    return (
        centered.group_by(factor)
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

    df = _c_bin(pl.read_parquet(f"{args.inp}/auprc.parquet"))
    df = df.with_columns(
        pl.when(pl.col("n_pca") == 0)
        .then(pl.lit("none"))
        .otherwise(pl.col("n_pca").cast(pl.Utf8))
        .alias("n_pca")
    )
    subsets = sorted(df["subset"].unique().to_list())

    # Single best-on-AVERAGE recipe: the config_id (shared across subsets, since
    # sample_configs is subset-independent) with the highest mean *within-subset-
    # centered* AUPRC. This picks ONE recipe by its cross-subset average — a fair,
    # pre-committed test — as opposed to the per-subset argmax (selection-biased,
    # optimistic), which we keep only as a labeled ceiling.
    centered = df.with_columns(
        (pl.col("auprc") - pl.col("auprc").mean().over("subset")).alias("ce")
    )
    by_cfg = (
        centered.group_by("config_id")
        .agg(pl.col("ce").mean().alias("mce"), pl.col("rep").first(),
             pl.col("scaler").first(), pl.col("n_pca").first(), pl.col("c").first())
        .sort("mce", descending=True)
    )
    avg_cfg = by_cfg.row(0, named=True)
    avg_ci = avg_cfg["config_id"]
    print(
        f"\nbest-on-average recipe (config {avg_ci}): {avg_cfg['rep']} "
        f"scaler={int(avg_cfg['scaler'])} pca={avg_cfg['n_pca']} C={avg_cfg['c']:.3g}"
    )

    # ---- per subset: LLR baseline, fixed best-avg recipe, oracle ceiling ----
    print(f"\n########## {args.model} — per-subset AUPRC ± SE, probe vs zero-shot LLR ##########")
    print(
        f"{'subset':22s} {'n':>5} {'pos':>4}  {'LLR':>12}  "
        f"{'fixed recipe':>12} {'Δ':>14} {'p':>6}   {'oracle':>12} (rep)"
    )
    rows = []
    for s in subsets:
        d = df.filter(pl.col("subset") == s)
        orc = d.sort("auprc", descending=True).row(0, named=True)
        oof = pl.read_parquet(f"{args.inp}/oof_{s}.parquet")
        y = oof["label"].to_numpy().astype(int)
        mg = oof["match_group"].to_numpy()
        llr = (-(oof["llr_fwd"] + oof["llr_rc"]) / 2).to_numpy()  # minus_llr_avg
        llr_st = auprc_with_bootstrap_se(y, llr, mg, rng=0)
        # fixed best-on-average recipe
        fx = oof[f"oof_{avg_ci}"].to_numpy()
        fx_st = auprc_with_bootstrap_se(y, fx, mg, rng=0)
        fx_dl = paired_metric_delta_bootstrap(y, fx, llr, mg, rng=0)
        fsig = "✓" if fx_dl["p_two_sided"] < 0.05 and fx_dl["delta"] > 0 else ""
        # oracle ceiling (per-subset argmax — optimistic)
        orc_st = auprc_with_bootstrap_se(y, oof[f"oof_{orc['config_id']}"].to_numpy(), mg, rng=0)
        print(
            f"{s:22s} {len(y):>5} {int(y.sum()):>4}  "
            f"{llr_st['value']:.3f}±{llr_st['se']:.3f}  "
            f"{fx_st['value']:.3f}±{fx_st['se']:.3f} "
            f"{fx_dl['delta']:+.3f}±{fx_dl['se']:.3f} {fx_dl['p_two_sided']:.3f}{fsig}   "
            f"{orc_st['value']:.3f}±{orc_st['se']:.3f} ({orc['rep']})"
        )
        rows.append(
            {
                "subset": s, "n": len(y), "n_pos": int(y.sum()),
                "llr_auprc": llr_st["value"], "llr_se": llr_st["se"],
                "fixed_auprc": fx_st["value"], "fixed_se": fx_st["se"],
                "fixed_delta": fx_dl["delta"], "fixed_delta_se": fx_dl["se"],
                "fixed_p": fx_dl["p_two_sided"],
                "oracle_auprc": orc_st["value"], "oracle_rep": orc["rep"],
            }
        )
    pl.DataFrame(rows).write_parquet(f"{args.inp}/best_vs_llr.parquet")

    # ---- 2. global subset-centered attribution ----
    print(f"\n########## {args.model} — global attribution (subset-centered ΔAUPRC) ##########")
    pool = df.filter(pl.col("rep_kind") == "pool")
    leverage = []
    for fac in POOL_FACTORS + ALL_FACTORS:
        src = pool if fac in POOL_FACTORS else df
        g = _marginal_centered(src, fac)
        spread = float(g["mean"].max() - g["mean"].min())
        leverage.append((fac, spread))
        print(f"\n  {fac}  (leverage spread = {spread:.3f}):")
        for r in g.iter_rows(named=True):
            print(f"    {str(r[fac]):14s} {r['mean']:+.3f} ± {r['se']:.3f}  (n={r['n']})")
    print("\n  factor leverage (by spread, ↓):")
    for fac, sp in sorted(leverage, key=lambda x: -x[1]):
        print(f"    {fac:10s} {sp:.3f}")
    print(f"\nwrote {args.inp}/best_vs_llr.parquet")


if __name__ == "__main__":
    main()
