"""Re-match: pull dataset_all parquets from S3, apply the iter-33 production
matching design, save dataset_unsplit locally (NO S3 / HF push).

Mirrors the `*_traits_dataset` / `eqtl_dataset` rules in the smk files so the
production design can be iterated on without round-tripping the full
Snakemake DAG (~30 s end-to-end vs. ~5 minutes via snakemake). Uses the same
library helpers (`add_tiered_maf_bin`, `bin_feature`) as the smk rules so the
output should be byte-identical to `results/dataset_unsplit/{name}.parquet`.

Single-threaded (kernel thrashes BLAS at 16 threads on 12M-row scale).
"""

import os

# Cap every thread source BEFORE importing numpy / sklearn / scipy / polars.
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
import resource
import time

import boto3
import polars as pl

from marin_dna.pipelines.evals.matching import (
    CAT_BASE,
    MAF_TIERED_LOG8_DISTAL_ONLY,
    MAF_TIERED_V1,
    add_subset_distance_bins,
    add_tiered_maf_bin,
    match_features,
)


_t0 = time.time()


def stamp(tag: str) -> None:
    elapsed = time.time() - _t0
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"  [+{elapsed:6.1f}s  peak={peak_mb:7.0f} MB] {tag}", flush=True)


s3 = boto3.client("s3", region_name="us-east-2")

CONT_BASE = [
    "distance_tss_pc",
    "distance_tss_nc",
    "distance_exon_pc",
    "distance_exon_nc",
]

DATASET_CONFIG = {
    "mendelian_traits": {"with_maf": False, "maf_scheme": None},
    "complex_traits": {"with_maf": True, "maf_scheme": MAF_TIERED_V1},
    "eqtl": {"with_maf": True, "maf_scheme": MAF_TIERED_LOG8_DISTAL_ONLY},
}
DATASETS = ("mendelian_traits", "complex_traits", "eqtl")


for name in DATASETS:
    cfg = DATASET_CONFIG[name]
    print(f"\n=== {name} (with_maf={cfg['with_maf']}) ===", flush=True)
    src = f"snakemake/evals/results/{name}/dataset_all.parquet"
    local_in = f"/tmp/{name}_da.parquet"
    if not os.path.exists(local_in):
        print(f"  downloading {src}...", flush=True)
        s3.download_file("oa-bolinas", src, local_in)

    V = pl.read_parquet(local_in)
    if cfg["with_maf"]:
        # Drop NaN/null MAF (~1.5%); matches smk rule behaviour.
        V = V.filter(pl.col("MAF").is_finite() & pl.col("MAF").is_not_null())
    print(
        f"  dataset_all: {V.height} rows ({V.filter(pl.col('label')).height} pos)",
        flush=True,
    )
    stamp("after read_parquet")

    V = add_subset_distance_bins(V)
    cat = CAT_BASE + [
        "distance_tss_pc_bin",
        "distance_tss_nc_bin",
        "distance_exon_pc_bin",
    ]
    cont = CONT_BASE.copy()
    if cfg["with_maf"]:
        V = add_tiered_maf_bin(V, cfg["maf_scheme"], log_local_group_cols=CAT_BASE)
        cat.append("MAF_bin")
        cont.append("MAF")

    pos_V = V.filter(pl.col("label"))
    neg_V = V.filter(~pl.col("label"))
    del V
    gc.collect()
    stamp(f"before match_features (pos={pos_V.height} neg={neg_V.height})")
    matched = match_features(pos_V, neg_V, cont, cat, k=1).with_columns(
        subset=pl.col("consequence_group")
    )
    del pos_V, neg_V
    gc.collect()
    stamp("after match_features")

    out = f"/tmp/{name}_unsplit_v2.parquet"
    matched.write_parquet(out)
    n_pairs = matched.filter(pl.col("label")).height
    print(f"  matched pairs: {n_pairs}  → {out}", flush=True)
    print("  per-subset:", flush=True)
    print(
        matched.filter(pl.col("label"))
        .group_by("subset")
        .len()
        .sort("len", descending=True)
    )
