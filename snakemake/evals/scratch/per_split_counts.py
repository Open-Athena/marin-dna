"""Per-split, per-consequence_group positive counts for mendelian_traits at AF>0.001.

Uses the count-only k=1 gene-matched approximation:
  per match-group, kept = min(n_pos, n_neg).
"""

import polars as pl
import yaml

from marin_dna.pipelines.evals.trait_intervals import build_dataset

base = "s3://oa-bolinas/snakemake/evals/results"
COORDINATES = ["chrom", "pos", "ref", "alt"]
GROUP_COLS = [
    "chrom",
    "consequence_final",
    "tss_closest_gene_id",
    "exon_closest_gene_id",
]
EXCLUDE_SUBSETS = ["mature_miRNA_variant", "coding_sequence_variant"]

CHROMS = [str(i) for i in range(1, 23)] + ["X", "Y"]
SPLIT_CHROMS = {"train": CHROMS[::2], "test": CHROMS[1::2]}

with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)

positives = pl.read_parquet(f"{base}/mendelian_traits/positives.parquet").with_columns(
    label=pl.lit(True)
)
exon = pl.read_parquet(f"{base}/intervals/exon.parquet")
tss = pl.read_parquet(f"{base}/intervals/tss.parquet")

common = (
    pl.scan_parquet(f"{base}/gnomad/all.parquet")
    .filter((pl.col("AN") >= cfg["gnomad_min_AN"]) & (pl.col("AF") > 0.001))
    .collect()
).with_columns(label=pl.lit(False))
print(f"common at AF>0.001: {common.height} variants", flush=True)

V = pl.concat([positives, common], how="diagonal_relaxed")
V = build_dataset(
    V,
    exon,
    tss,
    cfg["exclude_consequences"],
    cfg["exon_proximal_dist"],
    cfg["tss_proximal_dist"],
    cfg["consequence_groups"],
).filter(~pl.col("consequence_group").is_in(EXCLUDE_SUBSETS))

# k=1 count-only: per (group_cols, consequence_group) take min(n_pos, n_neg)
pos_groups = (
    V.filter(pl.col("label"))
    .group_by(GROUP_COLS + ["consequence_group"])
    .agg(pl.len().alias("n_pos"))
)
neg_groups = (
    V.filter(~pl.col("label")).group_by(GROUP_COLS).agg(pl.len().alias("n_neg"))
)
joined = (
    pos_groups.join(neg_groups, on=GROUP_COLS, how="left")
    .with_columns(pl.col("n_neg").fill_null(0))
    .with_columns(n_kept=pl.min_horizontal("n_pos", "n_neg"))
)

per_split = []
for split_name, chroms in SPLIT_CHROMS.items():
    counts = (
        joined.filter(pl.col("chrom").is_in(chroms))
        .group_by("consequence_group")
        .agg(pl.col("n_kept").sum().alias(split_name))
    )
    per_split.append(counts)

# Also full
total = joined.group_by("consequence_group").agg(pl.col("n_kept").sum().alias("total"))

out = (
    total.join(per_split[0], on="consequence_group", how="full", coalesce=True)
    .join(per_split[1], on="consequence_group", how="full", coalesce=True)
    .fill_null(0)
    .sort("total", descending=True)
)
print(out)

# Total / non-missense rows
for cols in [["total", "train", "test"]]:
    t = out.select([pl.col(c).sum() for c in cols])
    nm = out.filter(pl.col("consequence_group") != "missense_variant").select(
        [pl.col(c).sum() for c in cols]
    )
    print("total:           ", dict(t.row(0, named=True)))
    print("total_nonmissense:", dict(nm.row(0, named=True)))
