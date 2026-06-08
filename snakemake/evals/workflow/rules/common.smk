import bioframe as bf
import pandas as pd
import polars as pl
import subprocess
from pathlib import Path

from marin_dna.data.genome import Genome
from cyvcf2 import VCF
from datasets import Dataset
from huggingface_hub import CommitOperationAdd, HfApi

from marin_dna.pipelines.evals.labeling import label_variants_by_pip
from marin_dna.pipelines.evals.materialize import materialize_sequences
from marin_dna.pipelines.evals import hf_readme
from marin_dna.pipelines.evals.matching import (
    CAT_BASE,
    add_subset_distance_bins_v2,
    match_features,
)
from marin_dna.pipelines.evals.matching_qc import compute_matching_qc
from marin_dna.pipelines.evals.trait_intervals import (
    add_exon,
    add_tss,
    build_dataset,
    get_exon,
    get_tss,
)
from marin_dna.pipelines.evals.variants import (
    COORDINATES,
    NUCLEOTIDES,
    attach_per_chrom_consequences,
    check_ref_alt,
    filter_chroms,
    filter_snp,
    lift_hg19_to_hg38,
)

CHROMS = [str(i) for i in range(1, 23)] + ["X", "Y"]
SPLIT_CHROMS = {
    "train": CHROMS[::2],  # odd chroms
    "test": CHROMS[1::2],  # even chroms
}
SPLITS = list(SPLIT_CHROMS.keys())
COORDS = ["chrom", "pos", "ref", "alt"]

# Column order for HF: coordinates, label, subset, match_group, then everything
# else. Datasets missing any of these columns just skip them.
PRIMARY_COLS = COORDS + ["label", "subset", "match_group"]

# Continuous features over which to compute per-subset AUPRC leak in the
# matching diagnostic. Mirrors the `continuous` list passed to
# `match_features` in each `{task}_dataset` rule.
QC_CONTINUOUS_FEATURES = {
    "mendelian_traits": [
        "distance_tss_pc",
        "distance_tss_nc",
        "distance_exon_pc",
        "distance_exon_nc",
    ],
    "complex_traits": [
        "distance_tss_pc",
        "distance_tss_nc",
        "distance_exon_pc",
        "distance_exon_nc",
        "MAF",
    ],
}

# Distance-bin schemes shared across the matched datasets. mendelian extends
# this with its own `distal` entry; complex_traits uses the base set verbatim.
# Edges: float("inf") closes the last bin as an open upper bound.
BASE_DISTANCE_BIN_SCHEME = {
    ("tss_proximal", "distance_tss_pc"): [0.0, 100.0, 1000.0, float("inf")],
    ("tss_proximal", "distance_exon_pc"): [0.0, 100.0, 1000.0, float("inf")],
    ("splicing", "distance_exon_pc"): [0.0, 5.0, 30.0, float("inf")],
}


def _reorder_columns(df):
    primary = [c for c in PRIMARY_COLS if c in df.columns]
    rest = [c for c in df.columns if c not in primary]
    return df[primary + rest]


# Commit SHA pinned at module-load so the HF README's pipeline permalink stays
# stable for an entire snakemake run. Falls back to "main" if git isn't reachable
# (shouldn't happen for sky workdir, but defensive).
try:
    GIT_SHA = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
except Exception:
    GIT_SHA = "main"



rule download_genome:
    output:
        "results/genome.fa.gz",
    params:
        url=config["genome_url"],
    shell:
        "wget {params.url} -O {output}"


rule split_dataset_by_chrom:
    input:
        "results/dataset_unsplit/{dataset}.parquet",
    output:
        expand("results/dataset/{{dataset}}/{split}.parquet", split=SPLITS),
    run:
        V = _reorder_columns(pd.read_parquet(input[0]))
        for split, path in zip(SPLITS, output):
            V[V.chrom.isin(SPLIT_CHROMS[split])].to_parquet(path, index=False)


rule materialize_eval_harness_dataset:
    input:
        parquet="results/dataset/{dataset}/{split}.parquet",
    output:
        "results/dataset/{dataset}_harness_{window_size}/{split}.parquet",
    params:
        # Canonical bgzipped + indexed GRCh38 (Ensembl release 115; sequence is
        # byte-identical to 113/114). pyfaidx reads it directly from S3 by
        # byte-range via fsspec/s3fs — no full download. Bypasses the
        # download_genome rule (Ensembl ships plain gzip; the new Genome class
        # post-#182 requires BGZF + .gzi index).
        genome_path=config["canonical_genome_path"],
    run:
        genome = Genome(params.genome_path)
        ds = Dataset.from_parquet(input.parquet)
        n_in = len(ds)
        ds = materialize_sequences(ds, genome, int(wildcards.window_size))
        # Sanity: two rows per input variant, exactly the two strand tags.
        assert len(ds) == 2 * n_in, (
            f"expected {2 * n_in} rows (2x input), got {len(ds)}"
        )
        strands = set(ds.unique("strand"))
        assert strands == {"+", "-"}, f"unexpected strand set: {strands}"
        ds.to_parquet(output[0])


def _hf_qc_input(wildcards):
    """QC parquet input — only the matched datasets have one. Harness
    derivatives and the unmatched DART-Eval datasets (caqtl, dsqtl) don't."""
    if wildcards.dataset not in QC_CONTINUOUS_FEATURES:
        return []
    return f"results/qc/{wildcards.dataset}.parquet"


def _hf_extra_files(dataset):
    """Dataset-specific companion files (path_in_repo -> local path) uploaded in the
    same commit as the splits + README. SGE ships its study-level MaveDB score
    calibrations as a tidy long table (wrong grain to fold into the per-variant
    splits)."""
    if dataset == "sge":
        return {"calibrations.parquet": "results/sge/calibrations.parquet"}
    return {}


rule hf_upload:
    input:
        train="results/dataset/{dataset}/train.parquet",
        test="results/dataset/{dataset}/test.parquet",
        qc=_hf_qc_input,
        extra=lambda wc: list(_hf_extra_files(wc.dataset).values()),
    output:
        touch("results/upload.done/{dataset}"),
    params:
        repo_name=lambda wildcards: f"{config['output_hf_prefix']}_{wildcards.dataset}",
    run:
        api = HfApi()
        api.create_repo(params.repo_name, repo_type="dataset", exist_ok=True)
        # Map each companion's repo filename -> its LOCALIZED input path. The S3
        # storage provider deletes the bare `results/...` local copy once the
        # producing rule finishes, so we must read the path snakemake localized for
        # *this* rule (`input.extra`), not the logical `_hf_extra_files` value.
        # keys() and the `extra` input (its values()) share dict order, so they zip.
        extra_local = dict(
            zip(_hf_extra_files(wildcards.dataset), input.extra)
        )
        # README: per-dataset card with splits, columns, retention, AUPRC-leak
        # diagnostic, provenance (commit-pinned permalink to the pipeline).
        readme = hf_readme.render(
            wildcards.dataset,
            sha=GIT_SHA,
            train_path=input.train,
            test_path=input.test,
            qc_path=input.qc if input.qc else None,
            calibration_path=extra_local.get("calibrations.parquet"),
        )
        # Single atomic commit: README + both splits (+ any companion files) land
        # together, so the repo is never in a half-updated state (train new / test
        # stale).
        ops = [
            CommitOperationAdd(
                path_in_repo="README.md", path_or_fileobj=readme.encode()
            ),
            CommitOperationAdd(
                path_in_repo="train.parquet", path_or_fileobj=str(input.train)
            ),
            CommitOperationAdd(
                path_in_repo="test.parquet", path_or_fileobj=str(input.test)
            ),
        ]
        for path_in_repo, local in extra_local.items():
            ops.append(
                CommitOperationAdd(
                    path_in_repo=path_in_repo, path_or_fileobj=str(local)
                )
            )
        api.create_commit(
            repo_id=params.repo_name,
            repo_type="dataset",
            operations=ops,
            commit_message=f"Upload {wildcards.dataset} dataset ({len(ops)} files)",
        )


ruleorder: materialize_eval_harness_dataset > split_dataset_by_chrom


rule dataset_matching_qc:
    """Per-subset matching diagnostics: subsampling drops + per-feature AUPRC leak."""
    input:
        pre="results/{dataset}/dataset_all.parquet",
        post="results/dataset_unsplit/{dataset}.parquet",
    output:
        "results/qc/{dataset}.parquet",
    wildcard_constraints:
        dataset="|".join(QC_CONTINUOUS_FEATURES.keys()),
    run:
        qc = compute_matching_qc(
            pl.read_parquet(input.pre),
            pl.read_parquet(input.post),
            QC_CONTINUOUS_FEATURES[wildcards.dataset],
        )
        qc.write_parquet(output[0])
