"""Fast AF sweep — count gene-matched k=1 positive survivors per subset.

Skips match_features (slow per-group cdist loop). For k=1, every
(chrom, consequence_final, tss_closest_gene_id, exon_closest_gene_id)
group with N_pos positives and N_neg ≥ 1 negatives contributes
min(N_pos, N_neg) surviving positives. That's two group_bys + a join.

Logs progress to /tmp/af_sweep.log so it can be tailed remotely.
"""

import os
import time
import psutil
import polars as pl
import yaml

from marin_dna.pipelines.evals.trait_intervals import build_dataset

LOG_PATH = "/tmp/af_sweep.log"
_log_fh = open(LOG_PATH, "w")


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    _log_fh.write(line + "\n")
    _log_fh.flush()


proc = psutil.Process(os.getpid())


def mem_gb():
    return proc.memory_info().rss / 1024**3


base = "s3://oa-bolinas/snakemake/evals/results"
COORDINATES = ["chrom", "pos", "ref", "alt"]
GROUP_COLS = [
    "chrom",
    "consequence_final",
    "tss_closest_gene_id",
    "exon_closest_gene_id",
]

with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)

positives = pl.read_parquet(f"{base}/mendelian_traits/positives.parquet").with_columns(
    label=pl.lit(True)
)
exon = pl.read_parquet(f"{base}/intervals/exon.parquet")
tss = pl.read_parquet(f"{base}/intervals/tss.parquet")
gnomad = f"{base}/gnomad/all.parquet"

log(f"positives: {positives.height} rows, mem {mem_gb():.2f} GB")

af_thresholds = [0.001, 0.005, 0.01, 0.05]  # 0.001 first (most informative)
results_per_subset = {}
totals = {}

for af_min in af_thresholds:
    t = time.time()
    common = (
        pl.scan_parquet(gnomad)
        .filter((pl.col("AN") >= cfg["gnomad_min_AN"]) & (pl.col("AF") > af_min))
        .collect()
    ).with_columns(label=pl.lit(False))
    log(
        f"=== AF > {af_min}: {common.height} common variants (filter took {time.time() - t:.1f}s, mem {mem_gb():.2f} GB) ==="
    )

    t = time.time()
    V = pl.concat([positives, common], how="diagonal_relaxed")
    V = build_dataset(
        V,
        exon,
        tss,
        cfg["exclude_consequences"],
        cfg["exon_proximal_dist"],
        cfg["tss_proximal_dist"],
        cfg["consequence_groups"],
    )
    log(
        f"  build_dataset: {time.time() - t:.1f}s, mem {mem_gb():.2f} GB, rows={V.height}, pos={V.filter(pl.col('label')).height}"
    )

    # Count-only k=1 gene-match: per group, min(n_pos, n_neg).
    t = time.time()
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
    by_subset = joined.group_by("consequence_group").agg(
        pl.col("n_kept").sum().alias(f"af_{af_min}")
    )
    total_kept = joined.select(pl.col("n_kept").sum()).item()
    log(f"  count_kept (k=1, gene): {time.time() - t:.1f}s, total_kept={total_kept}")
    log(str(by_subset.sort(f"af_{af_min}", descending=True)))
    results_per_subset[af_min] = by_subset
    totals[af_min] = total_kept

    del common, V, pos_groups, neg_groups, joined

# Combine
out = results_per_subset[af_thresholds[0]]
for af in af_thresholds[1:]:
    out = out.join(
        results_per_subset[af], on="consequence_group", how="full", coalesce=True
    )
log("=== gene-matched k=1 positive count by gnomAD common-AF threshold ===")
log(str(out.sort(f"af_{af_thresholds[0]}", descending=True)))
log(f"totals: {totals}")
