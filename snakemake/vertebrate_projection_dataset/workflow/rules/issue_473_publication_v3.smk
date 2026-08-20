"""Additive staged-source publication rules for issue #473.

Every immutable producer input is retrieved exactly once into a retained local
snapshot. All shuffling, card generation, and validation rules consume that
snapshot, so no rule depends on a storage plugin re-retrieving a released
remote input.
"""

from pathlib import Path
import shutil

ISSUE_473_V3_PUBLICATION_ROOT = (
    f"results/{ISSUE_473_PIPELINE_VERSION}/{ISSUE_473_PUBLICATION_COMMIT}/"
    f"{ISSUE_473_PUBLICATION_CONFIG_SHA256}/issue_473_publication_v3"
)
ISSUE_473_V3_SOURCE_ROOT = f"{ISSUE_473_V3_PUBLICATION_ROOT}/source_snapshot"
ISSUE_473_V3_SOURCE_DATASETS = f"{ISSUE_473_V3_SOURCE_ROOT}/datasets"
ISSUE_473_V3_SOURCE_METADATA = f"{ISSUE_473_V3_SOURCE_ROOT}/metadata"
ISSUE_473_V3_HF_RESULTS = f"{ISSUE_473_V3_PUBLICATION_ROOT}/hf"
ISSUE_473_V3_HF_MANIFEST = (
    f"{ISSUE_473_V3_PUBLICATION_ROOT}/validation/hf_publication_manifest.json"
)
ISSUE_473_V3_HF_MANIFEST_ARCHIVE = (
    f"{ISSUE_473_V3_PUBLICATION_ROOT}/validation/"
    "hf_publication_manifest.archived.json"
)


def issue_473_v3_original_dataset_source(wildcards):
    return (
        f"{ISSUE_473_SOURCE_DATASETS}/{wildcards.projection_policy}/"
        f"{wildcards.region_label}/{wildcards.filename}"
    )


def issue_473_v3_staged_dataset_path(dataset_key, filename):
    dataset = ISSUE_473_PUBLICATION_BY_KEY[dataset_key]
    return (
        f"{ISSUE_473_V3_SOURCE_DATASETS}/{dataset.projection_policy}/"
        f"{dataset.region_label}/{filename}"
    )


def issue_473_v3_staged_summary(wildcards):
    return local(
        issue_473_v3_staged_dataset_path(wildcards.dataset, "split_summary.json")
    )


rule issue_473_v3_stage_dataset_source:
    input:
        issue_473_v3_original_dataset_source,
    output:
        local(
            f"{ISSUE_473_V3_SOURCE_DATASETS}/{{projection_policy}}/"
            "{region_label}/{filename}"
        ),
    wildcard_constraints:
        projection_policy="center_1|full_window",
        region_label="cds|ccre_enhancer_centered",
        filename=r"train[.]parquet|validation[.]parquet|split_summary[.]json",
    run:
        destination = Path(output[0])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(input[0], destination)


rule issue_473_v3_stage_metadata_source:
    input:
        lambda wc: f"{ISSUE_473_PRODUCER_RESULTS}/metadata/{wc.filename}",
    output:
        local(f"{ISSUE_473_V3_SOURCE_METADATA}/{{filename}}"),
    wildcard_constraints:
        filename=r"producer[.]json|species_active[.]tsv",
    run:
        destination = Path(output[0])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(input[0], destination)


rule issue_473_v3_prepare_train_jsonl_shards:
    input:
        lambda wc: local(issue_473_v3_staged_dataset_path(wc.dataset, "train.parquet")),
    output:
        temp(
            local(
                expand(
                    f"{ISSUE_473_V3_HF_RESULTS}/{{{{dataset}}}}/data/train/"
                    "{shard}.jsonl",
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


rule issue_473_v3_prepare_validation_jsonl_shards:
    input:
        lambda wc: local(
            issue_473_v3_staged_dataset_path(wc.dataset, "validation.parquet")
        ),
    output:
        temp(
            local(
                expand(
                    f"{ISSUE_473_V3_HF_RESULTS}/{{{{dataset}}}}/data/validation/"
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


rule issue_473_v3_compress_publication_shard:
    input:
        local(f"{ISSUE_473_V3_HF_RESULTS}/{{dataset}}/data/{{split}}/" "{shard}.jsonl"),
    output:
        local(
            f"{ISSUE_473_V3_HF_RESULTS}/{{dataset}}/data/{{split}}/"
            "{shard}.jsonl.zst"
        ),
    wildcard_constraints:
        dataset=ISSUE_473_PUBLICATION_KEY_RE,
        split="train|validation",
        shard=r"shard_\d{4}",
    conda:
        "../envs/bioinformatics.yaml"
    threads: 8
    shell:
        "zstd -T{threads} --force {input} -o {output}"


rule issue_473_v3_dataset_card:
    input:
        train=lambda wc: local(
            issue_473_v3_staged_dataset_path(wc.dataset, "train.parquet")
        ),
        validation=lambda wc: local(
            issue_473_v3_staged_dataset_path(wc.dataset, "validation.parquet")
        ),
        summary=issue_473_v3_staged_summary,
        species=local(f"{ISSUE_473_V3_SOURCE_METADATA}/species_active.tsv"),
    output:
        local(f"{ISSUE_473_V3_HF_RESULTS}/{{dataset}}/README.md"),
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


rule issue_473_v3_hf_artifact_manifest:
    input:
        producer=local(f"{ISSUE_473_V3_SOURCE_METADATA}/producer.json"),
        sources=local(
            [
                issue_473_v3_staged_dataset_path(dataset.key, filename)
                for dataset in ISSUE_473_PUBLICATION_DATASETS
                for filename in [
                    "train.parquet",
                    "validation.parquet",
                    "split_summary.json",
                ]
            ]
        ),
        train=local(
            expand(
                f"{ISSUE_473_V3_HF_RESULTS}/{{dataset}}/data/train/"
                "{shard}.jsonl.zst",
                dataset=ISSUE_473_PUBLICATION_KEYS,
                shard=ISSUE_473_TRAIN_SHARDS,
            )
        ),
        validation=local(
            expand(
                f"{ISSUE_473_V3_HF_RESULTS}/{{dataset}}/data/validation/"
                "{shard}.jsonl.zst",
                dataset=ISSUE_473_PUBLICATION_KEYS,
                shard=ISSUE_473_VALIDATION_SHARDS,
            )
        ),
        cards=local(
            expand(
                f"{ISSUE_473_V3_HF_RESULTS}/{{dataset}}/README.md",
                dataset=ISSUE_473_PUBLICATION_KEYS,
            )
        ),
    output:
        local(ISSUE_473_V3_HF_MANIFEST),
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
            ISSUE_473_V3_HF_RESULTS,
            ISSUE_473_V3_SOURCE_DATASETS,
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


rule issue_473_v3_archive_hf_artifact_manifest:
    input:
        local(ISSUE_473_V3_HF_MANIFEST),
    output:
        ISSUE_473_V3_HF_MANIFEST_ARCHIVE,
    run:
        destination = Path(output[0])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(input[0]).read_bytes())


rule issue_473_v3_all_hf_files:
    input:
        local(ISSUE_473_V3_HF_MANIFEST),
        ISSUE_473_V3_HF_MANIFEST_ARCHIVE,
        local(
            expand(
                f"{ISSUE_473_V3_HF_RESULTS}/{{dataset}}/data/train/"
                "{shard}.jsonl.zst",
                dataset=ISSUE_473_PUBLICATION_KEYS,
                shard=ISSUE_473_TRAIN_SHARDS,
            )
        ),
        local(
            expand(
                f"{ISSUE_473_V3_HF_RESULTS}/{{dataset}}/data/validation/"
                "{shard}.jsonl.zst",
                dataset=ISSUE_473_PUBLICATION_KEYS,
                shard=ISSUE_473_VALIDATION_SHARDS,
            )
        ),
        local(
            expand(
                f"{ISSUE_473_V3_HF_RESULTS}/{{dataset}}/README.md",
                dataset=ISSUE_473_PUBLICATION_KEYS,
            )
        ),


rule issue_473_v3_hf_upload_dataset:
    input:
        manifest=local(ISSUE_473_V3_HF_MANIFEST),
        train=lambda wc: [
            local(
                f"{ISSUE_473_V3_HF_RESULTS}/{wc.dataset}/data/train/"
                f"{shard}.jsonl.zst"
            )
            for shard in ISSUE_473_TRAIN_SHARDS
        ],
        validation=lambda wc: [
            local(
                f"{ISSUE_473_V3_HF_RESULTS}/{wc.dataset}/data/validation/"
                f"{shard}.jsonl.zst"
            )
            for shard in ISSUE_473_VALIDATION_SHARDS
        ],
        card=local(f"{ISSUE_473_V3_HF_RESULTS}/{{dataset}}/README.md"),
    output:
        local(f"{ISSUE_473_V3_PUBLICATION_ROOT}/upload.done/{{dataset}}"),
    wildcard_constraints:
        dataset=ISSUE_473_PUBLICATION_KEY_RE,
    resources:
        hf_uploads=1,
    params:
        repo=lambda wc: ISSUE_473_PUBLICATION_BY_KEY[wc.dataset].hf_repo,
        workers=int(config["hf_upload_workers"]),
    run:
        upload_private_validated_dataset(
            ISSUE_473_V3_HF_RESULTS,
            input.manifest,
            output[0],
            cohort=wildcards.dataset,
            repo_id=params.repo,
            workers=params.workers,
        )


rule issue_473_v3_all_hf:
    input:
        local(
            expand(
                f"{ISSUE_473_V3_PUBLICATION_ROOT}/upload.done/{{dataset}}",
                dataset=ISSUE_473_PUBLICATION_KEYS,
            )
        ),
