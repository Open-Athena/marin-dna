"""Sweep MAF_bin schemes (general — same scheme across all subsets) on
complex_traits + eqtl to map the Pareto frontier between MAF-leakage bias
and pair retention. Reads cached dataset_all parquets from /tmp.

Schemes scanned (coarse → fine):
  0.  no_bin   → just MAF in continuous, no categorical
  1.  3bin     → [0, 0.01, 0.05, 0.5]                 (rare / low / common)
  2.  5bin    → [0, 0.001, 0.01, 0.05, 0.2, 0.5]
  3.  10bin   → log-spaced 10 buckets across [0, 0.5]
  4.  20bin   → iter-24 production scheme (MAF_BIN_EDGES)

For each (dataset, scheme) prints:
  (a) per-subset matched-pair count
  (b) per-subset MAF PA + binomial p
  (c) total matched pairs

Output is structured so we can read off the Pareto curve at a glance.
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

from marin_dna.pipelines.evals.matching import (
    MAF_BIN_EDGES,
    bin_feature,
    match_features,
)


SCHEMES: dict[str, list[float] | None] = {
    "no_bin": None,
    "3bin": [0, 0.01, 0.05, 0.5],
    "5bin": [0, 0.001, 0.01, 0.05, 0.2, 0.5],
    "10bin": [0, 0.0005, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.5],
    "20bin": MAF_BIN_EDGES,  # iter 24
}

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
    print(
        f"  loaded {df_full.height} rows ({df_full.filter(pl.col('label')).height} pos)",
        flush=True,
    )

    # Pre-compute per-subset positive counts (the cap on matched pairs).
    subset_pos = (
        df_full.filter(pl.col("label"))
        .group_by("consequence_group")
        .len()
        .rename({"len": "n_pos"})
    )
    subsets = subset_pos.sort("n_pos", descending=True)["consequence_group"].to_list()

    # --- 3 tables, columns = schemes, rows = subsets ---
    pair_table: dict[str, dict[str, int]] = {s: {} for s in subsets}
    pa_table: dict[str, dict[str, float]] = {s: {} for s in subsets}
    p_table: dict[str, dict[str, float]] = {s: {} for s in subsets}
    totals: dict[str, int] = {}

    for scheme_name, edges in SCHEMES.items():
        t0 = time.time()
        cont = CONT_BASE.copy()
        cat = CAT_BASE.copy()
        if edges is None:
            V = df_full
        else:
            V = df_full.with_columns(
                bin_feature("MAF", edges, right_closed=True).alias("MAF_bin")
            )
            cat.append("MAF_bin")

        matched = match_features(
            V.filter(pl.col("label")),
            V.filter(~pl.col("label")),
            cont,
            cat,
            k=1,
        )
        del V
        gc.collect()

        n_total = matched.filter(pl.col("label")).height
        totals[scheme_name] = n_total
        for s in subsets:
            sub = matched.filter(pl.col("consequence_group") == s)
            n_pos = sub.filter(pl.col("label")).height
            pair_table[s][scheme_name] = n_pos
            pa, _, p = pa_p(sub, "MAF")
            pa_table[s][scheme_name] = pa
            p_table[s][scheme_name] = p
        elapsed = time.time() - t0
        print(
            f"  scheme={scheme_name:7s}  total={n_total:6d}  ({elapsed:5.1f}s)",
            flush=True,
        )

    schemes = list(SCHEMES.keys())
    print(f"\n--- {dataset}: matched pair count per (subset, scheme) ---")
    print(f"{'subset':25s}" + "".join(f"  {s:>7}" for s in schemes))
    for s in subsets:
        n_pos = subset_pos.filter(pl.col("consequence_group") == s)["n_pos"][0]
        line = f"{s[:24]:25s}"
        for sch in schemes:
            line += f"  {pair_table[s][sch]:7d}"
        line += f"   (of {n_pos} pos)"
        print(line)
    line = f"{'TOTAL':25s}"
    for sch in schemes:
        line += f"  {totals[sch]:7d}"
    print(line)

    print(f"\n--- {dataset}: MAF pairwise-accuracy (PA) per (subset, scheme) ---")
    print(f"{'subset':25s}" + "".join(f"  {s:>7}" for s in schemes))
    for s in subsets:
        line = f"{s[:24]:25s}"
        for sch in schemes:
            pa = pa_table[s][sch]
            line += f"  {pa:7.3f}" if pa == pa else "      NaN"
        print(line)

    print(
        f"\n--- {dataset}: MAF p-value per (subset, scheme) — '★' = Bonferroni-significant ---"
    )
    n_cells = len(subsets)
    bonf = 0.05 / n_cells
    print(
        f"  Bonferroni threshold (per scheme, MAF only across {n_cells} subsets) = {bonf:.2e}"
    )
    print(f"{'subset':25s}" + "".join(f"  {s:>7}" for s in schemes))
    for s in subsets:
        line = f"{s[:24]:25s}"
        for sch in schemes:
            p = p_table[s][sch]
            if p != p:  # NaN
                line += "      NaN"
            else:
                star = "★" if p < bonf else " "
                line += f"  {p:6.0e}{star}"
        print(line)
