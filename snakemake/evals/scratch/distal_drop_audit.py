"""Why did the distal pair count drop 1551 → 672?

Hypothesis: with TSS bins active for distal, the categorical match key
(chrom × consequence × 4 closest_gene_ids × MAF_bin × 4 dist_bins) is so
fine-grained that many positive bin-combos have ZERO matching negatives —
so match_features hits `if neg_group_pl is None: continue` and drops them
silently (warning goes to stderr, not the sky log).

Audit on the local /tmp dataset_all parquet:
  1. Apply the same bin transform.
  2. For DISTAL positives only: count how many positive bin-combos have
     0 matching negatives in the same bin-combo.
  3. Break down losses by (which bin column is the culprit).
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

import polars as pl

from marin_dna.pipelines.evals.matching import (
    BIN_NA,
    EXON_DIST_BIN_EDGES,
    MAF_BIN_EDGES,
    TSS_DIST_BIN_EDGES,
    bin_feature,
)

CAT = [
    "chrom",
    "consequence_final",
    "tss_closest_pc_gene_id",
    "tss_closest_nc_gene_id",
    "exon_closest_pc_gene_id",
    "exon_closest_nc_gene_id",
    "distance_tss_pc_bin",
    "distance_tss_nc_bin",
    "distance_exon_pc_bin",
    "distance_exon_nc_bin",
    "MAF_bin",
]


def add_bins(V: pl.DataFrame) -> pl.DataFrame:
    return V.with_columns(
        pl.when(pl.col("consequence_group").is_in(["tss_proximal", "distal"]))
        .then(bin_feature("distance_tss_pc", TSS_DIST_BIN_EDGES))
        .otherwise(pl.lit(BIN_NA))
        .alias("distance_tss_pc_bin"),
        pl.when(pl.col("consequence_group").is_in(["tss_proximal", "distal"]))
        .then(bin_feature("distance_tss_nc", TSS_DIST_BIN_EDGES))
        .otherwise(pl.lit(BIN_NA))
        .alias("distance_tss_nc_bin"),
        pl.when(pl.col("consequence_group") == "splicing")
        .then(bin_feature("distance_exon_pc", EXON_DIST_BIN_EDGES))
        .otherwise(pl.lit(BIN_NA))
        .alias("distance_exon_pc_bin"),
        pl.when(pl.col("consequence_group") == "splicing")
        .then(bin_feature("distance_exon_nc", EXON_DIST_BIN_EDGES))
        .otherwise(pl.lit(BIN_NA))
        .alias("distance_exon_nc_bin"),
        bin_feature("MAF", MAF_BIN_EDGES, right_closed=True).alias("MAF_bin"),
    )


df = pl.read_parquet("/tmp/complex_traits_da.parquet")
print(f"dataset_all: {df.height}, pos={df.filter(pl.col('label')).height}")
V = add_bins(df)

distal = V.filter(pl.col("consequence_group") == "distal")
pos = distal.filter(pl.col("label"))
neg = distal.filter(~pl.col("label"))
print(f"distal: pos={pos.height} neg={neg.height}")

# 1. Bin distribution of distal positives.
print("\n--- distal positive distance_tss_pc bin counts ---")
print(pos.group_by("distance_tss_pc_bin").len().sort("distance_tss_pc_bin"))
print("\n--- distal positive distance_tss_nc bin counts ---")
print(pos.group_by("distance_tss_nc_bin").len().sort("distance_tss_nc_bin"))

# 2. How many distal positives have 0 matching distal negs in their full bin combo?
pos_keys = pos.select(CAT).unique()
neg_keys = neg.select(CAT).unique().with_columns(neg_combo=pl.lit(True))
pos_with_neg = pos_keys.join(neg_keys, on=CAT, how="left")
n_combos_total = pos_keys.height
n_combos_no_neg = pos_with_neg.filter(pl.col("neg_combo").is_null()).height
print(
    f"\n--- POS combo coverage --- "
    f"total={n_combos_total}  with_neg={n_combos_total - n_combos_no_neg}  "
    f"NO_neg={n_combos_no_neg}"
)

# How many positive ROWS map to a no-neg combo?
pos_with_neg_rows = pos.join(neg_keys, on=CAT, how="left")
n_pos_lost = pos_with_neg_rows.filter(pl.col("neg_combo").is_null()).height
print(
    f"--- POS rows lost (combo has 0 negs): {n_pos_lost} / {pos.height}"
    f"  = {n_pos_lost / pos.height * 100:.1f}%"
)

# 3. Drop one categorical column at a time and re-check, to see which is the
#    bottleneck.
print("\n--- single-column drop experiments (pos rows that gain a matching neg) ---")
baseline_lost = n_pos_lost
for drop_col in CAT:
    cat_drop = [c for c in CAT if c != drop_col]
    neg_keys_d = neg.select(cat_drop).unique().with_columns(_has=pl.lit(True))
    pos_d = pos.join(neg_keys_d, on=cat_drop, how="left")
    lost_d = pos_d.filter(pl.col("_has").is_null()).height
    rescued = baseline_lost - lost_d
    print(f"  drop {drop_col:>34}  lost: {lost_d:4d}  (rescued {rescued:+4d})")
