"""Isolated CDS full-window row-random split and public dataset rules."""

from marin_dna_vertebrate_projection.issue_473.immutable import (
    read_immutable_sources,
    stage_immutable_s3_object,
)
from marin_dna_vertebrate_projection.issue_473.random_validation import (
    upload_public_dataset,
    validate_publication,
    write_dataset_card,
    write_uniform_random_split,
)
from marin_dna_vertebrate_projection.provenance import (
    hash_pipeline_config,
    resolve_pipeline_commit,
)

ISSUE_473_RANDOM_COMMIT = resolve_pipeline_commit()
ISSUE_473_RANDOM_CONFIG_SHA256 = hash_pipeline_config(config)
ISSUE_473_RANDOM_VERSION = str(config["pipeline_version"])
ISSUE_473_RANDOM_ROOT = (
    f"results/{ISSUE_473_RANDOM_VERSION}/{ISSUE_473_RANDOM_COMMIT}/"
    f"{ISSUE_473_RANDOM_CONFIG_SHA256}/issue_473_random_validation"
)
ISSUE_473_RANDOM_SOURCE_MANIFEST = str(config["source_manifest"])
ISSUE_473_RANDOM_SOURCES = read_immutable_sources(ISSUE_473_RANDOM_SOURCE_MANIFEST)
ISSUE_473_RANDOM_SOURCE = ISSUE_473_RANDOM_SOURCES[str(config["source_name"])]
ISSUE_473_RANDOM_SOURCE_LOCAL = str(config["source_local_path"])
ISSUE_473_RANDOM_SOURCE_RECEIPT = f"{ISSUE_473_RANDOM_SOURCE_LOCAL}.receipt.json"
ISSUE_473_RANDOM_DATA_ROOT = f"{ISSUE_473_RANDOM_ROOT}/dataset"
ISSUE_473_RANDOM_TRAIN = f"{ISSUE_473_RANDOM_DATA_ROOT}/train_original.parquet"
ISSUE_473_RANDOM_VALIDATION = f"{ISSUE_473_RANDOM_DATA_ROOT}/validation.parquet"
ISSUE_473_RANDOM_SUMMARY = f"{ISSUE_473_RANDOM_DATA_ROOT}/split_summary.json"
ISSUE_473_RANDOM_COHORT = str(config["cohort"])
ISSUE_473_RANDOM_HF_REPO = str(config["hf_repo"])
ISSUE_473_RANDOM_HF_ROOT = f"{ISSUE_473_RANDOM_ROOT}/hf"
ISSUE_473_RANDOM_MANIFEST = f"{ISSUE_473_RANDOM_ROOT}/publication_manifest.json"
ISSUE_473_RANDOM_TRAIN_SHARD_COUNT = int(config["publication_train_shards"])
ISSUE_473_RANDOM_VALIDATION_SHARD_COUNT = int(config["publication_validation_shards"])
ISSUE_473_RANDOM_TRAIN_SHARDS = [
    f"shard_{index:04d}" for index in range(ISSUE_473_RANDOM_TRAIN_SHARD_COUNT)
]
ISSUE_473_RANDOM_VALIDATION_SHARDS = [
    f"shard_{index:04d}" for index in range(ISSUE_473_RANDOM_VALIDATION_SHARD_COUNT)
]


rule issue_473_random_validation_stage_source:
    input:
        local(ISSUE_473_RANDOM_SOURCE_MANIFEST),
    output:
        artifact=local(ISSUE_473_RANDOM_SOURCE_LOCAL),
        receipt=local(ISSUE_473_RANDOM_SOURCE_RECEIPT),
    resources:
        mem_mb=2000,
    run:
        stage_immutable_s3_object(
            ISSUE_473_RANDOM_SOURCE,
            output.artifact,
            output.receipt,
        )


rule issue_473_random_validation_split:
    input:
        local(ISSUE_473_RANDOM_SOURCE_LOCAL),
    output:
        train=ISSUE_473_RANDOM_TRAIN,
        validation=ISSUE_473_RANDOM_VALIDATION,
        summary=ISSUE_473_RANDOM_SUMMARY,
    resources:
        mem_mb=120000,
        final_large_scan=1,
    run:
        write_uniform_random_split(
            input[0],
            output.train,
            output.validation,
            output.summary,
            region_label=str(config["region_label"]),
            validation_rows=int(config["validation_rows"]),
            seed=int(config["split_seed"]),
            target_length=int(config["target_length"]),
        )


rule issue_473_random_validation_prepare_train:
    input:
        ISSUE_473_RANDOM_TRAIN,
    output:
        temp(
            local(
                expand(
                    f"{ISSUE_473_RANDOM_HF_ROOT}/{ISSUE_473_RANDOM_COHORT}/"
                    "data/train/{shard}.jsonl",
                    shard=ISSUE_473_RANDOM_TRAIN_SHARDS,
                )
            )
        ),
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
            add_rc=True,
            shuffle_seed=int(config["publication_shuffle_seed"]),
        )


rule issue_473_random_validation_prepare_validation:
    input:
        ISSUE_473_RANDOM_VALIDATION,
    output:
        temp(
            local(
                expand(
                    f"{ISSUE_473_RANDOM_HF_ROOT}/{ISSUE_473_RANDOM_COHORT}/"
                    "data/validation/{shard}.jsonl",
                    shard=ISSUE_473_RANDOM_VALIDATION_SHARDS,
                )
            )
        ),
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


rule issue_473_random_validation_compress:
    input:
        local(
            f"{ISSUE_473_RANDOM_HF_ROOT}/{ISSUE_473_RANDOM_COHORT}/"
            "data/{split}/{shard}.jsonl"
        ),
    output:
        local(
            f"{ISSUE_473_RANDOM_HF_ROOT}/{ISSUE_473_RANDOM_COHORT}/"
            "data/{split}/{shard}.jsonl.zst"
        ),
    wildcard_constraints:
        split="train|validation",
        shard=r"shard_\d{4}",
    conda:
        "../envs/bioinformatics.yaml"
    threads: 8
    shell:
        "zstd -T{threads} --force {input} -o {output}"


rule issue_473_random_validation_card:
    input:
        ISSUE_473_RANDOM_SUMMARY,
    output:
        f"{ISSUE_473_RANDOM_HF_ROOT}/{ISSUE_473_RANDOM_COHORT}/README.md",
    resources:
        mem_mb=1000,
    run:
        write_dataset_card(
            input[0],
            output[0],
            hf_repo=ISSUE_473_RANDOM_HF_REPO,
            pipeline_commit=ISSUE_473_RANDOM_COMMIT,
        )


rule issue_473_random_validation_manifest:
    input:
        train=ISSUE_473_RANDOM_TRAIN,
        validation=ISSUE_473_RANDOM_VALIDATION,
        summary=ISSUE_473_RANDOM_SUMMARY,
        train_shards=local(
            expand(
                f"{ISSUE_473_RANDOM_HF_ROOT}/{ISSUE_473_RANDOM_COHORT}/"
                "data/train/{shard}.jsonl.zst",
                shard=ISSUE_473_RANDOM_TRAIN_SHARDS,
            )
        ),
        validation_shards=local(
            expand(
                f"{ISSUE_473_RANDOM_HF_ROOT}/{ISSUE_473_RANDOM_COHORT}/"
                "data/validation/{shard}.jsonl.zst",
                shard=ISSUE_473_RANDOM_VALIDATION_SHARDS,
            )
        ),
        card=f"{ISSUE_473_RANDOM_HF_ROOT}/{ISSUE_473_RANDOM_COHORT}/README.md",
    output:
        ISSUE_473_RANDOM_MANIFEST,
    threads: 8
    resources:
        mem_mb=8000,
        final_large_scan=1,
    run:
        validate_publication(
            ISSUE_473_RANDOM_HF_ROOT,
            input.train,
            input.validation,
            input.summary,
            output[0],
            cohort=ISSUE_473_RANDOM_COHORT,
            hf_repo=ISSUE_473_RANDOM_HF_REPO,
            pipeline_commit=ISSUE_473_RANDOM_COMMIT,
            config_sha256=ISSUE_473_RANDOM_CONFIG_SHA256,
            train_shards=ISSUE_473_RANDOM_TRAIN_SHARD_COUNT,
            validation_shards=ISSUE_473_RANDOM_VALIDATION_SHARD_COUNT,
            target_length=int(config["target_length"]),
            workers=threads,
        )


rule issue_473_random_validation_all_hf_files:
    input:
        ISSUE_473_RANDOM_MANIFEST,
        local(
            expand(
                f"{ISSUE_473_RANDOM_HF_ROOT}/{ISSUE_473_RANDOM_COHORT}/"
                "data/train/{shard}.jsonl.zst",
                shard=ISSUE_473_RANDOM_TRAIN_SHARDS,
            )
        ),
        local(
            expand(
                f"{ISSUE_473_RANDOM_HF_ROOT}/{ISSUE_473_RANDOM_COHORT}/"
                "data/validation/{shard}.jsonl.zst",
                shard=ISSUE_473_RANDOM_VALIDATION_SHARDS,
            )
        ),
        f"{ISSUE_473_RANDOM_HF_ROOT}/{ISSUE_473_RANDOM_COHORT}/README.md",


rule issue_473_random_validation_upload:
    input:
        manifest=ISSUE_473_RANDOM_MANIFEST,
        train=local(
            expand(
                f"{ISSUE_473_RANDOM_HF_ROOT}/{ISSUE_473_RANDOM_COHORT}/"
                "data/train/{shard}.jsonl.zst",
                shard=ISSUE_473_RANDOM_TRAIN_SHARDS,
            )
        ),
        validation=local(
            expand(
                f"{ISSUE_473_RANDOM_HF_ROOT}/{ISSUE_473_RANDOM_COHORT}/"
                "data/validation/{shard}.jsonl.zst",
                shard=ISSUE_473_RANDOM_VALIDATION_SHARDS,
            )
        ),
        card=f"{ISSUE_473_RANDOM_HF_ROOT}/{ISSUE_473_RANDOM_COHORT}/README.md",
    output:
        temp(local(f"{ISSUE_473_RANDOM_ROOT}/upload.done")),
    resources:
        hf_uploads=1,
    run:
        upload_public_dataset(
            ISSUE_473_RANDOM_HF_ROOT,
            input.manifest,
            output[0],
            cohort=ISSUE_473_RANDOM_COHORT,
            repo_id=ISSUE_473_RANDOM_HF_REPO,
            workers=int(config["hf_upload_workers"]),
        )


rule issue_473_random_validation_all_hf:
    input:
        local(f"{ISSUE_473_RANDOM_ROOT}/upload.done"),
