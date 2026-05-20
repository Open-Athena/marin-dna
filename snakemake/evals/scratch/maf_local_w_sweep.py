"""Local equal-WIDTH MAF_bin sweep on complex_traits + eqtl.

Within each categorical match group (chrom × consequence × 4 closest_*_gene_id),
divide the JOINT (pos+neg) MAF range into n equal-width bins. Two variants:
  - lin:  equal-width on linear MAF.    bin edges = linspace(min_MAF, max_MAF, n+1)
  - log:  equal-width on log10(MAF).    bin edges = linspace(log10_min, log10_max, n+1)

Replaces the global MAF_bin in the categorical match key. MAF stays continuous
for cdist tie-break within each sub-bucket.

Sweep n_bins ∈ {2, 4, 8, 16, 32} for both width variants.

Different from quantile bins (iter 28): local-width has variable count per bin
(rare-MAF buckets may end up empty in some groups), but the bin EDGES are by
absolute MAF distance, which should produce neutral cdist matching within
each bucket (no quantile-induced reverse-skew where pos clusters at the low
edge of the upper buckets).
"""

import os

for var in (
    "POLARS_MAX_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(var, "1")

import gc
import time

import boto3
import polars as pl
from scipy.stats import binomtest

from marin_dna.pipelines.evals.matching import match_features


CAT_BASE = [
    "chrom",
    "consequence_final",
    "tss_closest_pc_gene_id",
    "tss_closest_nc_gene_id",
    "exon_closest_pc_gene_id",
    "exon_closest_nc_gene_id",
]
CONT_BASE = [
    "distance_tss_pc",
    "distance_tss_nc",
    "distance_exon_pc",
    "distance_exon_nc",
    "MAF",
]
N_BINS_GRID = [2, 4, 8, 16, 32]


def _safe_idx(value: pl.Expr, lo: pl.Expr, width: pl.Expr, n_bins: int) -> pl.Expr:
    """((value - lo) / width).floor() → Int64, with fallback for width==0 or
    NaN (polars evaluates both when/then branches; need strict=False to keep
    the (value-lo)/0 -> NaN cast from blowing up).
    """
    return (
        ((value - lo) / width)
        .floor()
        .cast(pl.Int64, strict=False)
        .fill_null(0)
        .clip(0, n_bins - 1)
    )


def add_local_lin_bin(df: pl.DataFrame, n_bins: int) -> pl.DataFrame:
    """Equal-width linear MAF bins per group, joint pos+neg ref."""
    df = df.with_columns(
        pl.col("MAF").min().over(CAT_BASE).alias("_lo"),
        pl.col("MAF").max().over(CAT_BASE).alias("_hi"),
    )
    width = (pl.col("_hi") - pl.col("_lo")) / n_bins
    bin_idx = _safe_idx(pl.col("MAF"), pl.col("_lo"), width, n_bins)
    df = df.with_columns(pl.format("b{}", bin_idx).alias("MAF_local_bin"))
    return df.drop(["_lo", "_hi"])


def add_local_log_bin(df: pl.DataFrame, n_bins: int) -> pl.DataFrame:
    """Equal-width log10-MAF bins per group, joint pos+neg ref."""
    log_maf = pl.col("MAF").clip(1e-10, 1.0).log10()
    df = df.with_columns(
        log_maf.min().over(CAT_BASE).alias("_lo"),
        log_maf.max().over(CAT_BASE).alias("_hi"),
    )
    width = (pl.col("_hi") - pl.col("_lo")) / n_bins
    bin_idx = _safe_idx(log_maf, pl.col("_lo"), width, n_bins)
    df = df.with_columns(pl.format("b{}", bin_idx).alias("MAF_local_bin"))
    return df.drop(["_lo", "_hi"])


def pa_p(V: pl.DataFrame, feature: str) -> tuple[float, int, float]:
    pos = (
        V.filter(pl.col("label"))
        .select(["match_group", feature])
        .rename({feature: "pos"})
    )
    neg = (
        V.filter(~pl.col("label"))
        .select(["match_group", feature])
        .rename({feature: "neg"})
    )
    paired = pos.join(neg, on="match_group", how="inner")
    n = paired.height
    if n == 0:
        return float("nan"), 0, float("nan")
    diff = (paired["pos"] - paired["neg"]).to_numpy()
    wins = int((diff > 0).sum())
    losses = int((diff < 0).sum())
    ties = n - wins - losses
    pa = (wins + 0.5 * ties) / n
    decisive = wins + losses
    if decisive == 0:
        return pa, n, float("nan")
    p = binomtest(wins, decisive, p=0.5, alternative="two-sided").pvalue
    return pa, n, p


s3 = boto3.client("s3", region_name="us-east-2")

for dataset in ("complex_traits", "eqtl"):
    print(f"\n{'#' * 70}\n# {dataset}\n{'#' * 70}", flush=True)
    src = f"snakemake/evals/results/{dataset}/dataset_all.parquet"
    local_in = f"/tmp/{dataset}_da.parquet"
    if not os.path.exists(local_in):
        print(f"  downloading {src}...", flush=True)
        s3.download_file("oa-bolinas", src, local_in)
    df_full = pl.read_parquet(local_in)
    # Drop NaN/null MAF rows up-front. Avoids NaN-cast errors in the local-bin
    # computation. The iter-26 global scheme handled these via bin_feature's
    # null branch ("OOR" bin) — same effective behaviour, slightly fewer rows.
    df_full = df_full.filter(pl.col("MAF").is_finite() & pl.col("MAF").is_not_null())
    n_pos = df_full.filter(pl.col("label")).height
    print(f"  loaded {df_full.height} rows ({n_pos} pos)", flush=True)

    subset_pos = (
        df_full.filter(pl.col("label"))
        .group_by("consequence_group")
        .len()
        .rename({"len": "n_pos"})
    )
    subsets = subset_pos.sort("n_pos", descending=True)["consequence_group"].to_list()

    pair_table: dict[str, dict[str, int]] = {s: {} for s in subsets}
    pa_table: dict[str, dict[str, float]] = {s: {} for s in subsets}
    p_table: dict[str, dict[str, float]] = {s: {} for s in subsets}
    totals: dict[str, int] = {}

    scheme_names = (
        ["no_bin"] + [f"lin{n}" for n in N_BINS_GRID] + [f"log{n}" for n in N_BINS_GRID]
    )
    for scheme in scheme_names:
        t0 = time.time()
        if scheme == "no_bin":
            V = df_full
            cat = CAT_BASE
        else:
            n_b = int(scheme.replace("lin", "").replace("log", ""))
            if scheme.startswith("lin"):
                V = add_local_lin_bin(df_full, n_b)
            else:
                V = add_local_log_bin(df_full, n_b)
            cat = CAT_BASE + ["MAF_local_bin"]

        matched = match_features(
            V.filter(pl.col("label")),
            V.filter(~pl.col("label")),
            CONT_BASE,
            cat,
            k=1,
        )
        if scheme != "no_bin":
            del V
            gc.collect()

        n_total = matched.filter(pl.col("label")).height
        totals[scheme] = n_total
        for s in subsets:
            sub = matched.filter(pl.col("consequence_group") == s)
            pair_table[s][scheme] = sub.filter(pl.col("label")).height
            pa, _, p = pa_p(sub, "MAF")
            pa_table[s][scheme] = pa
            p_table[s][scheme] = p
        print(
            f"  scheme={scheme:7s}  total={n_total:6d}  ({time.time() - t0:5.1f}s)",
            flush=True,
        )

    print(f"\n--- {dataset}: matched pair count per (subset, scheme) ---")
    print(f"{'subset':25s}" + "".join(f"  {s:>7}" for s in scheme_names))
    for s in subsets:
        n_pos_s = subset_pos.filter(pl.col("consequence_group") == s)["n_pos"][0]
        line = f"{s[:24]:25s}"
        for sch in scheme_names:
            line += f"  {pair_table[s][sch]:7d}"
        line += f"   (of {n_pos_s} pos)"
        print(line)
    line = f"{'TOTAL':25s}"
    for sch in scheme_names:
        line += f"  {totals[sch]:7d}"
    print(line)

    print(f"\n--- {dataset}: MAF pairwise-accuracy (PA) per (subset, scheme) ---")
    print(f"{'subset':25s}" + "".join(f"  {s:>7}" for s in scheme_names))
    for s in subsets:
        line = f"{s[:24]:25s}"
        for sch in scheme_names:
            pa = pa_table[s][sch]
            line += f"  {pa:7.3f}" if pa == pa else "      NaN"
        print(line)

    print(
        f"\n--- {dataset}: MAF p-value per (subset, scheme) — '★' = Bonferroni-significant ---"
    )
    bonf = 0.05 / len(subsets)
    print(
        f"  Bonferroni threshold (per scheme, MAF only across {len(subsets)} subsets) = {bonf:.2e}"
    )
    print(f"{'subset':25s}" + "".join(f"  {s:>7}" for s in scheme_names))
    for s in subsets:
        line = f"{s[:24]:25s}"
        for sch in scheme_names:
            p = p_table[s][sch]
            if p != p:
                line += "      NaN"
            else:
                star = "★" if p < bonf else " "
                line += f"  {p:6.0e}{star}"
        print(line)
