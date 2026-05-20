"""Iter 31 — hybrid MAF_bin: per-subset choice between global-edge bins and
local equal-width log10 bins.

Iter 30 picked tiered_v1 as the complex_traits Pareto winner (1212 pairs,
0 Bonf-significant MAF leaks). Iter 29 picked log8 as the eqtl winner
(6707 pairs, 0 leaks; the only scheme that closes eqtl/distal). Goal of
iter 31: combine the two — keep tiered_v1's coarser bins for small subsets
(missense / synon / splicing recover their pair counts) but replace the
20bin scheme on the big high-leak subsets with **log local 8** so the
eqtl/distal residual is closed.

Schemes:
  tiered_v1                = iter-30 winner, global edges only
  tiered_log8_big          = log_local_8 for {distal, tss_prox, ncRNA},
                              tiered_v1 elsewhere
  tiered_log8_distal_only  = log_local_8 for distal only, tiered_v1 elsewhere
  tiered_log8_global       = log_local_8 for everything (sanity reference)
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

BINS_20 = MAF_BIN_EDGES
BINS_10 = [0.0, 0.0005, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.5]
BINS_5 = [0.0, 0.001, 0.01, 0.05, 0.2, 0.5]
BINS_3 = [0.0, 0.01, 0.05, 0.5]

LOG_LOCAL_N = 8
LOG_LOCAL = "log_local"  # sentinel string in scheme dicts

# Subset → bin choice. Either global edges (list) or LOG_LOCAL (use local-log-8).
SCHEMES: dict[str, dict[str, list[float] | str]] = {
    "tiered_v1": {
        "distal": BINS_20,
        "tss_proximal": BINS_20,
        "non_coding_transcript_exon_variant": BINS_20,
        "3_prime_UTR_variant": BINS_10,
        "5_prime_UTR_variant": BINS_10,
        "missense_variant": BINS_10,
        "synonymous_variant": BINS_5,
        "splicing": BINS_5,
        "mature_miRNA_variant": BINS_5,
        "stop_retained_variant": BINS_5,
        "coding_sequence_variant": BINS_5,
    },
    "tiered_log8_big": {
        # Big subsets get log_local_8 (closes eqtl/distal asymptote)
        "distal": LOG_LOCAL,
        "tss_proximal": LOG_LOCAL,
        "non_coding_transcript_exon_variant": LOG_LOCAL,
        # Mid + small same as tiered_v1
        "3_prime_UTR_variant": BINS_10,
        "5_prime_UTR_variant": BINS_10,
        "missense_variant": BINS_10,
        "synonymous_variant": BINS_5,
        "splicing": BINS_5,
        "mature_miRNA_variant": BINS_5,
        "stop_retained_variant": BINS_5,
        "coding_sequence_variant": BINS_5,
    },
    "tiered_log8_distal_only": {
        # Only distal gets log_local; rest is tiered_v1
        "distal": LOG_LOCAL,
        "tss_proximal": BINS_20,
        "non_coding_transcript_exon_variant": BINS_20,
        "3_prime_UTR_variant": BINS_10,
        "5_prime_UTR_variant": BINS_10,
        "missense_variant": BINS_10,
        "synonymous_variant": BINS_5,
        "splicing": BINS_5,
        "mature_miRNA_variant": BINS_5,
        "stop_retained_variant": BINS_5,
        "coding_sequence_variant": BINS_5,
    },
    "tiered_log8_global": {  # sanity ref: log_local_8 everywhere
        s: LOG_LOCAL
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
}


def add_hybrid_bin(
    df: pl.DataFrame, scheme: dict[str, list[float] | str]
) -> pl.DataFrame:
    """Per-row MAF_bin column, choosing between global edges and log_local_8
    based on row's consequence_group.
    """
    needs_log_local = any(v == LOG_LOCAL for v in scheme.values())
    if needs_log_local:
        log_maf = pl.col("MAF").clip(1e-10, 1.0).log10()
        df = df.with_columns(
            log_maf.min().over(CAT_BASE).alias("_lo"),
            log_maf.max().over(CAT_BASE).alias("_hi"),
        )
        width = (pl.col("_hi") - pl.col("_lo")) / LOG_LOCAL_N
        log_local_idx = (
            ((log_maf - pl.col("_lo")) / width)
            .floor()
            .cast(pl.Int64, strict=False)
            .fill_null(0)
            .clip(0, LOG_LOCAL_N - 1)
        )
        log_local_label = pl.format("ll:{}", log_local_idx)

    expr = pl.lit("UNKNOWN")
    for subset, val in scheme.items():
        if val == LOG_LOCAL:
            bin_expr = log_local_label
        else:
            bin_expr = pl.format(
                "{}:{}",
                pl.lit(subset[:8]),
                bin_feature("MAF", val, right_closed=True),
            )
        expr = (
            pl.when(pl.col("consequence_group") == subset)
            .then(bin_expr)
            .otherwise(expr)
        )

    df = df.with_columns(expr.alias("MAF_bin"))
    if needs_log_local:
        df = df.drop(["_lo", "_hi"])
    return df


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
    # Drop NaN/null MAF (1.5% of complex_traits) — the log_local computation
    # would propagate NaN otherwise.
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

    for scheme_name, scheme in SCHEMES.items():
        t0 = time.time()
        V = add_hybrid_bin(df_full, scheme)
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
            f"  scheme={scheme_name:25s}  total={n_total:6d}  ({time.time() - t0:5.1f}s)",
            flush=True,
        )

    schemes = list(SCHEMES.keys())
    print(f"\n--- {dataset}: matched pair count per (subset, scheme) ---")
    print(f"{'subset':25s}" + "".join(f"  {s:>23}" for s in schemes))
    for s in subsets:
        n_pos_s = subset_pos.filter(pl.col("consequence_group") == s)["n_pos"][0]
        line = f"{s[:24]:25s}"
        for sch in schemes:
            line += f"  {pair_table[s][sch]:23d}"
        line += f"   (of {n_pos_s} pos)"
        print(line)
    line = f"{'TOTAL':25s}"
    for sch in schemes:
        line += f"  {totals[sch]:23d}"
    print(line)

    print(f"\n--- {dataset}: MAF pairwise-accuracy (PA) per (subset, scheme) ---")
    print(f"{'subset':25s}" + "".join(f"  {s:>23}" for s in schemes))
    for s in subsets:
        line = f"{s[:24]:25s}"
        for sch in schemes:
            pa = pa_table[s][sch]
            line += f"  {pa:23.3f}" if pa == pa else "                      NaN"
        print(line)

    print(
        f"\n--- {dataset}: MAF p-value per (subset, scheme) — '★' = Bonferroni-significant ---"
    )
    bonf = 0.05 / len(subsets)
    print(f"  Bonferroni threshold = {bonf:.2e}")
    print(f"{'subset':25s}" + "".join(f"  {s:>23}" for s in schemes))
    for s in subsets:
        line = f"{s[:24]:25s}"
        for sch in schemes:
            p = p_table[s][sch]
            if p != p:
                line += "                      NaN"
            else:
                star = "★" if p < bonf else " "
                line += f"                {p:6.0e}{star}"
        print(line)
