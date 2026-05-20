"""Local-quantile MAF_bin sweep on complex_traits + eqtl.

Replaces the global MAF_bin (iter 24's 20-edge log-spaced scheme) with a
LOCAL quantile bin: within each categorical match group
(chrom × consequence_final × 4 closest_*_gene_id), assign each row a quantile
bucket of MAF.

Two reference variants:
  - joint:  quantiles computed on pos+neg jointly. Each bucket has
            ~1/n_q of the GROUP rows.
  - neg:    quantiles computed on negatives only. Each bucket has
            exactly 1/n_q of the NEG rows; pos rows fall into whichever
            bucket their MAF lands in. Better-behaved for retention since
            no bucket is empty of negs by construction.

The bin replaces the global MAF_bin in the categorical match key. MAF stays
continuous for cdist tie-break within each sub-group. Mendelian skipped —
no MAF column.

Sweep n_quantiles ∈ {2, 4, 8, 16, 32} for both reference variants.
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
N_QUANTILES_GRID = [2, 4, 8, 16, 32]


def add_local_q_bin_joint(df: pl.DataFrame, n_q: int) -> pl.DataFrame:
    """Joint pos+neg quantile binning — each bucket holds 1/n_q of group rows."""
    labels = [f"q{i}" for i in range(n_q)]
    return df.with_columns(
        pl.col("MAF")
        .qcut(quantiles=n_q, labels=labels, allow_duplicates=True)
        .cast(pl.String)
        .over(CAT_BASE)
        .alias("MAF_local_q_bin")
    )


def add_local_q_bin_neg(df: pl.DataFrame, n_q: int) -> pl.DataFrame:
    """Neg-only quantile bins. Edges from neg distribution per group; pos rows
    bucketed by where their MAF falls in those edges. Each bucket has exactly
    1/n_q of the NEG rows by construction (so retention should be high).
    """
    neg = df.filter(~pl.col("label"))
    levels = [(i + 1) / n_q for i in range(n_q - 1)]  # n_q - 1 internal edges
    edges = neg.group_by(CAT_BASE).agg(
        [pl.col("MAF").quantile(q).alias(f"_e{i}") for i, q in enumerate(levels)]
    )
    df = df.join(edges, on=CAT_BASE, how="left")
    bucket = pl.lit(f"q{n_q - 1}")
    for i in reversed(range(n_q - 1)):
        bucket = (
            pl.when(pl.col("MAF") <= pl.col(f"_e{i}"))
            .then(pl.lit(f"q{i}"))
            .otherwise(bucket)
        )
    df = df.with_columns(bucket.alias("MAF_local_q_bin"))
    return df.drop([f"_e{i}" for i in range(n_q - 1)])


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

    schemes: dict[str, dict] = {}
    pair_table: dict[str, dict[str, int]] = {s: {} for s in subsets}
    pa_table: dict[str, dict[str, float]] = {s: {} for s in subsets}
    p_table: dict[str, dict[str, float]] = {s: {} for s in subsets}
    totals: dict[str, int] = {}
    times: dict[str, float] = {}

    # No-bin baseline (for reference / Pareto comparison).
    scheme_names = (
        ["no_bin"]
        + [f"j_q{n}" for n in N_QUANTILES_GRID]  # joint-quantile
        + [f"n_q{n}" for n in N_QUANTILES_GRID]  # neg-only quantile
    )
    for scheme in scheme_names:
        t0 = time.time()
        if scheme == "no_bin":
            V = df_full
            cat = CAT_BASE
        else:
            n_q = int(scheme.split("_q")[1])
            if scheme.startswith("j"):
                V = add_local_q_bin_joint(df_full, n_q)
            else:
                V = add_local_q_bin_neg(df_full, n_q)
            cat = CAT_BASE + ["MAF_local_q_bin"]

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
        elapsed = time.time() - t0
        times[scheme] = elapsed
        print(
            f"  scheme={scheme:7s}  total={n_total:6d}  ({elapsed:5.1f}s)", flush=True
        )

    # --- print 3 tables ---
    print(f"\n--- {dataset}: matched pair count per (subset, scheme) ---")
    print(f"{'subset':25s}" + "".join(f"  {s:>7}" for s in scheme_names))
    for s in subsets:
        n_pos = subset_pos.filter(pl.col("consequence_group") == s)["n_pos"][0]
        line = f"{s[:24]:25s}"
        for sch in scheme_names:
            line += f"  {pair_table[s][sch]:7d}"
        line += f"   (of {n_pos} pos)"
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
