"""Additive review and Hugging Face publication rules for issue #473."""

import re

from marin_dna_vertebrate_projection.issue_473.publication import (
    parse_publication_datasets,
    validate_artifacts,
    write_dataset_card,
)
from marin_dna_vertebrate_projection.provenance import (
    hash_pipeline_config,
    resolve_pipeline_commit,
    validate_producer_manifest,
)
from marin_dna_vertebrate_projection.publication import upload_validated_dataset

ISSUE_473_PUBLICATION_DATASETS = parse_publication_datasets(
    config["publication_datasets"]
)
ISSUE_473_PUBLICATION_BY_KEY = {
    dataset.key: dataset for dataset in ISSUE_473_PUBLICATION_DATASETS
}
ISSUE_473_PUBLICATION_KEYS = list(ISSUE_473_PUBLICATION_BY_KEY)
ISSUE_473_PUBLICATION_KEY_RE = "|".join(
    re.escape(key) for key in ISSUE_473_PUBLICATION_KEYS
)
ISSUE_473_PRODUCER_COMMIT = str(config["producer_commit"])
ISSUE_473_PRODUCER_CONFIG_SHA256 = str(config["producer_config_sha256"])
ISSUE_473_PRODUCER_TIER = str(config["producer_tier"])
ISSUE_473_PIPELINE_VERSION = str(config["pipeline_version"])
ISSUE_473_PUBLICATION_COMMIT = resolve_pipeline_commit()
ISSUE_473_PUBLICATION_CONFIG_SHA256 = hash_pipeline_config(config)
ISSUE_473_PRODUCER_RESULTS = (
    f"results/{ISSUE_473_PIPELINE_VERSION}/{ISSUE_473_PRODUCER_COMMIT}/"
    f"{ISSUE_473_PRODUCER_CONFIG_SHA256}/{ISSUE_473_PRODUCER_TIER}"
)
ISSUE_473_SOURCE_DATASETS = (
    f"{ISSUE_473_PRODUCER_RESULTS}/experiments/473/fixed/full_scale/datasets"
)
ISSUE_473_PUBLICATION_ROOT = (
    f"results/{ISSUE_473_PIPELINE_VERSION}/{ISSUE_473_PUBLICATION_COMMIT}/"
    f"{ISSUE_473_PUBLICATION_CONFIG_SHA256}/issue_473_publication"
)
ISSUE_473_HF_RESULTS = f"{ISSUE_473_PUBLICATION_ROOT}/hf"
ISSUE_473_HF_MANIFEST = (
    f"{ISSUE_473_PUBLICATION_ROOT}/validation/hf_publication_manifest.json"
)
ISSUE_473_TRAIN_SHARD_COUNT = int(config["publication_train_shards"])
ISSUE_473_VALIDATION_SHARD_COUNT = int(config["publication_validation_shards"])
ISSUE_473_TRAIN_SHARDS = [
    f"shard_{index:04d}" for index in range(ISSUE_473_TRAIN_SHARD_COUNT)
]
ISSUE_473_VALIDATION_SHARDS = [
    f"shard_{index:04d}" for index in range(ISSUE_473_VALIDATION_SHARD_COUNT)
]


def issue_473_publication_dataset(wildcards):
    return ISSUE_473_PUBLICATION_BY_KEY[wildcards.dataset]


def issue_473_source_dataset_path(dataset_key, split):
    dataset = ISSUE_473_PUBLICATION_BY_KEY[dataset_key]
    return (
        f"{ISSUE_473_SOURCE_DATASETS}/{dataset.projection_policy}/"
        f"{dataset.region_label}/{split}.parquet"
    )


def issue_473_source_summary(wildcards):
    dataset = issue_473_publication_dataset(wildcards)
    return (
        f"{ISSUE_473_SOURCE_DATASETS}/{dataset.projection_policy}/"
        f"{dataset.region_label}/split_summary.json"
    )


rule issue_473_prepare_train_jsonl_shards:
    input:
        lambda wc: issue_473_source_dataset_path(wc.dataset, "train"),
    output:
        temp(
            local(
                expand(
                    f"{ISSUE_473_HF_RESULTS}/{{{{dataset}}}}/data/train/{{shard}}.jsonl",
                    shard=ISSUE_473_TRAIN_SHARDS,
                )
            )
        ),
    wildcard_constraints:
        dataset=ISSUE_473_PUBLICATION_KEY_RE,
    threads: workflow.cores
    resources:
        mem_mb=240000,
        final_large_scan=1,
    run:
        from marin_dna_vertebrate_projection.projection.dataset import (
            prepare_shards,
        )

        prepare_shards(
            parquet_path=str(input[0]),
            shard_paths=[str(path) for path in output],
            add_rc=False,
            shuffle_seed=int(config["publication_shuffle_seed"]),
        )


rule issue_473_prepare_validation_jsonl_shards:
    input:
        lambda wc: issue_473_source_dataset_path(wc.dataset, "validation"),
    output:
        temp(
            local(
                expand(
                    f"{ISSUE_473_HF_RESULTS}/{{{{dataset}}}}/data/validation/"
                    "{shard}.jsonl",
                    shard=ISSUE_473_VALIDATION_SHARDS,
                )
            )
        ),
    wildcard_constraints:
        dataset=ISSUE_473_PUBLICATION_KEY_RE,
    threads: workflow.cores
    resources:
        mem_mb=8000,
    run:
        from marin_dna_vertebrate_projection.projection.dataset import (
            prepare_shards,
        )

        prepare_shards(
            parquet_path=str(input[0]),
            shard_paths=[str(path) for path in output],
            add_rc=False,
            shuffle_seed=int(config["publication_shuffle_seed"]),
        )


rule issue_473_compress_publication_shard:
    input:
        local(f"{ISSUE_473_HF_RESULTS}/{{dataset}}/data/{{split}}/" "{shard}.jsonl"),
    output:
        local(f"{ISSUE_473_HF_RESULTS}/{{dataset}}/data/{{split}}/" "{shard}.jsonl.zst"),
    wildcard_constraints:
        dataset=ISSUE_473_PUBLICATION_KEY_RE,
        split="train|validation",
        shard=r"shard_\d{4}",
    conda:
        "../envs/bioinformatics.yaml"
    threads: 8
    shell:
        "zstd -T{threads} --force {input} -o {output}"


rule issue_473_dataset_card:
    input:
        train=lambda wc: issue_473_source_dataset_path(wc.dataset, "train"),
        validation=lambda wc: issue_473_source_dataset_path(wc.dataset, "validation"),
        summary=issue_473_source_summary,
        species=f"{ISSUE_473_PRODUCER_RESULTS}/metadata/species_active.tsv",
    output:
        f"{ISSUE_473_HF_RESULTS}/{{dataset}}/README.md",
    wildcard_constraints:
        dataset=ISSUE_473_PUBLICATION_KEY_RE,
    resources:
        mem_mb=2000,
    run:
        write_dataset_card(
            input.train,
            input.validation,
            input.summary,
            input.species,
            output[0],
            dataset=issue_473_publication_dataset(wildcards),
            producer_commit=ISSUE_473_PRODUCER_COMMIT,
            producer_config_sha256=ISSUE_473_PRODUCER_CONFIG_SHA256,
            publication_commit=ISSUE_473_PUBLICATION_COMMIT,
            validation_chrom=str(config["validation_chrom"]),
            target_length=int(config["target_length"]),
        )


rule issue_473_hf_artifact_manifest:
    input:
        producer=f"{ISSUE_473_PRODUCER_RESULTS}/metadata/producer.json",
        sources=[
            f"{ISSUE_473_SOURCE_DATASETS}/{dataset.projection_policy}/"
            f"{dataset.region_label}/{filename}"
            for dataset in ISSUE_473_PUBLICATION_DATASETS
            for filename in [
                "train.parquet",
                "validation.parquet",
                "split_summary.json",
            ]
        ],
        train=local(
            expand(
                f"{ISSUE_473_HF_RESULTS}/{{dataset}}/data/train/{{shard}}.jsonl.zst",
                dataset=ISSUE_473_PUBLICATION_KEYS,
                shard=ISSUE_473_TRAIN_SHARDS,
            )
        ),
        validation=local(
            expand(
                f"{ISSUE_473_HF_RESULTS}/{{dataset}}/data/validation/"
                "{shard}.jsonl.zst",
                dataset=ISSUE_473_PUBLICATION_KEYS,
                shard=ISSUE_473_VALIDATION_SHARDS,
            )
        ),
        cards=expand(
            f"{ISSUE_473_HF_RESULTS}/{{dataset}}/README.md",
            dataset=ISSUE_473_PUBLICATION_KEYS,
        ),
    output:
        ISSUE_473_HF_MANIFEST,
    threads: 8
    resources:
        mem_mb=8000,
        final_large_scan=1,
    run:
        validate_producer_manifest(
            input.producer,
            pipeline_commit=ISSUE_473_PRODUCER_COMMIT,
            config_sha256=ISSUE_473_PRODUCER_CONFIG_SHA256,
            pipeline_version=ISSUE_473_PIPELINE_VERSION,
            tier=ISSUE_473_PRODUCER_TIER,
        )
        validate_artifacts(
            ISSUE_473_HF_RESULTS,
            ISSUE_473_SOURCE_DATASETS,
            output[0],
            datasets=ISSUE_473_PUBLICATION_DATASETS,
            producer_commit=ISSUE_473_PRODUCER_COMMIT,
            producer_config_sha256=ISSUE_473_PRODUCER_CONFIG_SHA256,
            publication_commit=ISSUE_473_PUBLICATION_COMMIT,
            publication_config_sha256=ISSUE_473_PUBLICATION_CONFIG_SHA256,
            train_shards=ISSUE_473_TRAIN_SHARD_COUNT,
            validation_shards=ISSUE_473_VALIDATION_SHARD_COUNT,
            validation_chrom=str(config["validation_chrom"]),
            target_length=int(config["target_length"]),
            workers=threads,
        )


rule issue_473_all_hf_files:
    input:
        ISSUE_473_HF_MANIFEST,
        local(
            expand(
                f"{ISSUE_473_HF_RESULTS}/{{dataset}}/data/train/{{shard}}.jsonl.zst",
                dataset=ISSUE_473_PUBLICATION_KEYS,
                shard=ISSUE_473_TRAIN_SHARDS,
            )
        ),
        local(
            expand(
                f"{ISSUE_473_HF_RESULTS}/{{dataset}}/data/validation/"
                "{shard}.jsonl.zst",
                dataset=ISSUE_473_PUBLICATION_KEYS,
                shard=ISSUE_473_VALIDATION_SHARDS,
            )
        ),
        expand(
            f"{ISSUE_473_HF_RESULTS}/{{dataset}}/README.md",
            dataset=ISSUE_473_PUBLICATION_KEYS,
        ),


rule issue_473_hf_upload_dataset:
    input:
        manifest=ISSUE_473_HF_MANIFEST,
        train=lambda wc: [
            local(f"{ISSUE_473_HF_RESULTS}/{wc.dataset}/data/train/{shard}.jsonl.zst")
            for shard in ISSUE_473_TRAIN_SHARDS
        ],
        validation=lambda wc: [
            local(
                f"{ISSUE_473_HF_RESULTS}/{wc.dataset}/data/validation/"
                f"{shard}.jsonl.zst"
            )
            for shard in ISSUE_473_VALIDATION_SHARDS
        ],
        card=f"{ISSUE_473_HF_RESULTS}/{{dataset}}/README.md",
    output:
        local(f"{ISSUE_473_PUBLICATION_ROOT}/upload.done/{{dataset}}"),
    wildcard_constraints:
        dataset=ISSUE_473_PUBLICATION_KEY_RE,
    resources:
        hf_uploads=1,
    params:
        repo=lambda wc: ISSUE_473_PUBLICATION_BY_KEY[wc.dataset].hf_repo,
        workers=int(config["hf_upload_workers"]),
    run:
        upload_validated_dataset(
            ISSUE_473_HF_RESULTS,
            input.manifest,
            output[0],
            cohort=wildcards.dataset,
            repo_id=params.repo,
            workers=params.workers,
        )


rule issue_473_all_hf:
    input:
        local(
            expand(
                f"{ISSUE_473_PUBLICATION_ROOT}/upload.done/{{dataset}}",
                dataset=ISSUE_473_PUBLICATION_KEYS,
            )
        ),
