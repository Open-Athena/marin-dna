"""Subset-specific MAF_bin sweep on complex_traits + eqtl.

Each row's MAF_bin is computed using a scheme that depends on its
consequence_group, on the theory that:
  - subsets with big leak + large pos counts can afford fine bins
  - subsets with small pos counts get coarser bins (recover pairs)

Schemes tested:
  uniform_20bin = iter-26 reference: all subsets get the iter-24 20-edge scheme
  uniform_10bin = iter-27 reference: all subsets get the 10-edge scheme
  tiered_v1     = big-leak/big subsets → 20bin, medium → 10bin, small → 5bin
  tiered_v2     = more aggressive coarsening on small / non-leaking subsets
  tiered_v3     = uniform 10bin everywhere except missense/synon/splicing →
                  5bin (cushion the missense pair-count drop in complex)
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

# Bin granularity tiers (right-closed, log-spaced).
BINS_20 = MAF_BIN_EDGES  # iter 24 — 20 buckets
BINS_10 = [
    0.0,
    0.0005,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.15,
    0.2,
    0.5,
]  # 10 buckets
BINS_5 = [0.0, 0.001, 0.01, 0.05, 0.2, 0.5]  # 5 buckets
BINS_3 = [0.0, 0.01, 0.05, 0.5]  # 3 buckets


# Subset → bin edges. Subsets not listed get no MAF_bin column entry (label
# 'NONE' for every row of that subset, which acts like "no MAF bin in this
# subset" since 'NONE' shares with itself across subsets but match_features'
# categorical also includes consequence_final, so cross-subset collision is
# blocked).
SCHEMES: dict[str, dict[str, list[float]]] = {
    "uniform_20bin": {  # iter 26
        s: BINS_20
        for s in (
            "distal",
            "tss_proximal",
            "non_coding_transcript_exon_variant",
            "3_prime_UTR_variant",
            "5_prime_UTR_variant",
            "missense_variant",
            "synonymous_variant",
            "splicing",
            "mature_miRNA_variant",
            "stop_retained_variant",
            "coding_sequence_variant",
        )
    },
    "uniform_10bin": {  # iter 27
        s: BINS_10
        for s in (
            "distal",
            "tss_proximal",
            "non_coding_transcript_exon_variant",
            "3_prime_UTR_variant",
            "5_prime_UTR_variant",
            "missense_variant",
            "synonymous_variant",
            "splicing",
            "mature_miRNA_variant",
            "stop_retained_variant",
            "coding_sequence_variant",
        )
    },
    "tiered_v1": {
        # Big-leak / big subsets → 20bin
        "distal": BINS_20,
        "tss_proximal": BINS_20,
        "non_coding_transcript_exon_variant": BINS_20,
        # Medium → 10bin
        "3_prime_UTR_variant": BINS_10,
        "5_prime_UTR_variant": BINS_10,
        "missense_variant": BINS_10,
        # Small → 5bin
        "synonymous_variant": BINS_5,
        "splicing": BINS_5,
        "mature_miRNA_variant": BINS_5,
        "stop_retained_variant": BINS_5,
        "coding_sequence_variant": BINS_5,
    },
    "tiered_v2": {
        # Aggressive coarsening, more retention bias
        "distal": BINS_20,
        "tss_proximal": BINS_20,
        "non_coding_transcript_exon_variant": BINS_10,
        "3_prime_UTR_variant": BINS_5,
        "5_prime_UTR_variant": BINS_5,
        "missense_variant": BINS_5,
        "synonymous_variant": BINS_3,
        "splicing": BINS_3,
        "mature_miRNA_variant": BINS_3,
        "stop_retained_variant": BINS_3,
        "coding_sequence_variant": BINS_3,
    },
    "tiered_v3": {
        # 10bin everywhere except small-pop subsets (missense/synon/splicing → 5bin)
        "distal": BINS_10,
        "tss_proximal": BINS_10,
        "non_coding_transcript_exon_variant": BINS_10,
        "3_prime_UTR_variant": BINS_10,
        "5_prime_UTR_variant": BINS_10,
        "missense_variant": BINS_5,
        "synonymous_variant": BINS_5,
        "splicing": BINS_5,
        "mature_miRNA_variant": BINS_5,
        "stop_retained_variant": BINS_5,
        "coding_sequence_variant": BINS_5,
    },
}


def add_subset_bin(df: pl.DataFrame, scheme: dict[str, list[float]]) -> pl.DataFrame:
    """Apply per-subset MAF_bin via consequence_group lookup."""
    expr = pl.lit("UNKNOWN")
    for subset, edges in scheme.items():
        # bin_feature returns 'b0'..'b{n-1}' or 'OOR'; tag with subset so labels
        # don't collide across subsets within polars' partition_by (even though
        # consequence_final already separates them, defensively prefixing keeps
        # the bin labels self-describing in any inspection of the matched table).
        bin_expr = pl.format(
            "{}:{}", pl.lit(subset[:8]), bin_feature("MAF", edges, right_closed=True)
        )
        expr = (
            pl.when(pl.col("consequence_group") == subset)
            .then(bin_expr)
            .otherwise(expr)
        )
    return df.with_columns(expr.alias("MAF_bin"))


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

    for scheme_name, scheme in SCHEMES.items():
        t0 = time.time()
        V = add_subset_bin(df_full, scheme)
        cat = CAT_BASE + ["MAF_bin"]

        matched = match_features(
            V.filter(pl.col("label")),
            V.filter(~pl.col("label")),
            CONT_BASE,
            cat,
            k=1,
        )
        del V
        gc.collect()

        n_total = matched.filter(pl.col("label")).height
        totals[scheme_name] = n_total
        for s in subsets:
            sub = matched.filter(pl.col("consequence_group") == s)
            pair_table[s][scheme_name] = sub.filter(pl.col("label")).height
            pa, _, p = pa_p(sub, "MAF")
            pa_table[s][scheme_name] = pa
            p_table[s][scheme_name] = p
        print(
            f"  scheme={scheme_name:18s}  total={n_total:6d}  ({time.time() - t0:5.1f}s)",
            flush=True,
        )

    schemes = list(SCHEMES.keys())
    print(f"\n--- {dataset}: matched pair count per (subset, scheme) ---")
    print(f"{'subset':25s}" + "".join(f"  {s:>14}" for s in schemes))
    for s in subsets:
        n_pos_s = subset_pos.filter(pl.col("consequence_group") == s)["n_pos"][0]
        line = f"{s[:24]:25s}"
        for sch in schemes:
            line += f"  {pair_table[s][sch]:14d}"
        line += f"   (of {n_pos_s} pos)"
        print(line)
    line = f"{'TOTAL':25s}"
    for sch in schemes:
        line += f"  {totals[sch]:14d}"
    print(line)

    print(f"\n--- {dataset}: MAF pairwise-accuracy (PA) per (subset, scheme) ---")
    print(f"{'subset':25s}" + "".join(f"  {s:>14}" for s in schemes))
    for s in subsets:
        line = f"{s[:24]:25s}"
        for sch in schemes:
            pa = pa_table[s][sch]
            line += f"  {pa:14.3f}" if pa == pa else "             NaN"
        print(line)

    print(
        f"\n--- {dataset}: MAF p-value per (subset, scheme) — '★' = Bonferroni-significant ---"
    )
    bonf = 0.05 / len(subsets)
    print(f"  Bonferroni threshold = {bonf:.2e}")
    print(f"{'subset':25s}" + "".join(f"  {s:>14}" for s in schemes))
    for s in subsets:
        line = f"{s[:24]:25s}"
        for sch in schemes:
            p = p_table[s][sch]
            if p != p:
                line += "             NaN"
            else:
                star = "★" if p < bonf else " "
                line += f"        {p:6.0e}{star}"
        print(line)
