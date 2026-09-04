"""Audit, split, validate, and publish the order-deduplicated enhancer arm."""

import json
from pathlib import Path

from marin_dna_vertebrate_projection.order_publication import (
    write_order_dataset_card,
    write_order_dataset_split_files,
    write_order_source_audit,
)
from marin_dna_vertebrate_projection.provenance import (
    hash_pipeline_config,
    resolve_pipeline_commit,
    validate_producer_manifest,
    write_producer_manifest,
)
from marin_dna_vertebrate_projection.publication import (
    upload_validated_dataset,
    validate_artifacts,
)

PIPELINE_VERSION = str(config["pipeline_version"])
TIER = str(config["tier"])
assert TIER == "full"
PIPELINE_COMMIT = resolve_pipeline_commit()
PIPELINE_CONFIG_SHA256 = hash_pipeline_config(config)
RESULTS = (
    f"results/{PIPELINE_VERSION}/{PIPELINE_COMMIT}/"
    f"{PIPELINE_CONFIG_SHA256}/{TIER}"
)
PRODUCER_MANIFEST = f"{RESULTS}/metadata/producer.json"

SOURCE_PIPELINE_VERSION = str(config["source_pipeline_version"])
SOURCE_PIPELINE_COMMIT = str(config["source_pipeline_commit"])
SOURCE_CONFIG_SHA256 = str(config["source_config_sha256"])
SOURCE_TIER = str(config["source_tier"])
assert len(SOURCE_PIPELINE_COMMIT) == 40
assert len(SOURCE_CONFIG_SHA256) == 64
assert SOURCE_TIER == "full"
SOURCE_RESULTS = (
    f"results/{SOURCE_PIPELINE_VERSION}/{SOURCE_PIPELINE_COMMIT}/"
    f"{SOURCE_CONFIG_SHA256}/{SOURCE_TIER}"
)
SOURCE_PRODUCER_MANIFEST = f"{SOURCE_RESULTS}/metadata/producer.json"
COMBINED_SEQUENCES = f"{SOURCE_RESULTS}/sequences/all_sources.parquet"

ORDER_MANIFEST = local("config/species_vertebrate_order.tsv")
SOURCE_SPECIES_MANIFEST = local("config/species_selected.tsv")
SOURCE_REGION_ROWS = int(config["source_region_rows"])
EXPECTED_SOURCE_ROWS = int(config["source_rows"])
ORDER_TARGETS = int(config["order_manifest_targets"])
ORDER_MAMMALS = int(config["order_manifest_mammals"])
ORDER_NONMAMMALS = int(config["order_manifest_nonmammals"])
assert SOURCE_REGION_ROWS == 39_879_096
assert (ORDER_TARGETS, ORDER_MAMMALS, ORDER_NONMAMMALS) == (39, 18, 21)

COHORTS = list(config["region_cohorts"])
assert COHORTS == ["enhancer"]
COHORT_RE = "enhancer"
VALIDATION_ROWS = int(config["validation_rows"])
assert VALIDATION_ROWS == 16_384

HF_OWNER = str(config["hf_owner"])
HF_REPO_NAMES = {
    str(cohort): str(repo_name) for cohort, repo_name in config["hf_repo_names"].items()
}
assert set(HF_REPO_NAMES) == {"enhancer"}


def order_hf_repo(cohort):
    assert cohort == "enhancer"
    return f"{HF_OWNER}/{HF_REPO_NAMES[cohort]}"


PUBLICATION_TRAIN_SHARD_COUNT = int(config["publication_train_shards"])
PUBLICATION_VALIDATION_SHARD_COUNT = int(config["publication_validation_shards"])
PUBLICATION_SHUFFLE_SEED = int(config["publication_shuffle_seed"])
PUBLICATION_MAX_IN_MEMORY_ROWS = int(config["publication_max_in_memory_rows"])
assert 0 < PUBLICATION_MAX_IN_MEMORY_ROWS < 2 * (
    EXPECTED_SOURCE_ROWS - VALIDATION_ROWS
)
PUBLICATION_TRAIN_SHARDS = [
    f"shard_{index:04d}" for index in range(PUBLICATION_TRAIN_SHARD_COUNT)
]
PUBLICATION_VALIDATION_SHARDS = [
    f"shard_{index:04d}" for index in range(PUBLICATION_VALIDATION_SHARD_COUNT)
]
HF_RESULTS = f"{RESULTS}/hf"
HF_MANIFEST = f"{RESULTS}/hf_validation/hf_publication_manifest.json"
SOURCE_AUDIT = f"{RESULTS}/metadata/enhancer_order_source_audit.json"


rule phylop_order_publication_producer_manifest:
    output:
        PRODUCER_MANIFEST,
    run:
        write_producer_manifest(
            output[0],
            pipeline_commit=PIPELINE_COMMIT,
            config_sha256=PIPELINE_CONFIG_SHA256,
            pipeline_version=PIPELINE_VERSION,
            tier=TIER,
        )


rule phylop_order_source_audit:
    input:
        combined=COMBINED_SEQUENCES,
        order_manifest=ORDER_MANIFEST,
        source_manifest=SOURCE_SPECIES_MANIFEST,
    output:
        SOURCE_AUDIT,
    resources:
        mem_mb=48000,
        final_large_scan=1,
    run:
        write_order_source_audit(
            input.combined,
            input.order_manifest,
            input.source_manifest,
            output[0],
        )
        audit = json.loads(Path(output[0]).read_text())
        assert int(audit["candidate_region_rows"]) == SOURCE_REGION_ROWS
        assert int(audit["order_manifest_targets"]) == ORDER_TARGETS
        assert int(audit["selected_alignment_count_including_human"]) == (
            ORDER_TARGETS + 1
        )


rule phylop_order_dataset_splits:
    input:
        combined=COMBINED_SEQUENCES,
        audit=SOURCE_AUDIT,
        order_manifest=ORDER_MANIFEST,
        source_manifest=SOURCE_SPECIES_MANIFEST,
    output:
        train=f"{RESULTS}/datasets/{{region}}/train.parquet",
        validation=f"{RESULTS}/datasets/{{region}}/validation.parquet",
        selection=f"{RESULTS}/datasets/{{region}}/validation_selection.tsv",
        composition=f"{RESULTS}/datasets/{{region}}/validation_composition.tsv",
        summary=f"{RESULTS}/datasets/{{region}}/split_summary.json",
    wildcard_constraints:
        region=COHORT_RE,
    resources:
        mem_mb=60000,
        final_large_scan=1,
    run:
        assert EXPECTED_SOURCE_ROWS > VALIDATION_ROWS, (
            "pin source_rows from phylop_order_source_audit before building"
        )
        audit = json.loads(Path(input.audit).read_text())
        assert int(audit["source_rows"]) == EXPECTED_SOURCE_ROWS
        write_order_dataset_split_files(
            input.combined,
            input.order_manifest,
            input.source_manifest,
            output.train,
            output.validation,
            output.selection,
            output.composition,
            output.summary,
            add_rc=bool(config["add_rc"]),
            validation_rows=VALIDATION_ROWS,
            seed=int(config["validation_seed"]),
        )
        summary = json.loads(Path(output.summary).read_text())
        assert int(summary["source_rows"]) == EXPECTED_SOURCE_ROWS


rule phylop_order_prepare_train_jsonl_shards:
    input:
        f"{RESULTS}/datasets/{{region}}/train.parquet",
    output:
        temp(
            local(
                expand(
                    f"{HF_RESULTS}/{{{{region}}}}/data/train/{{shard}}.jsonl",
                    shard=PUBLICATION_TRAIN_SHARDS,
                )
            )
        ),
    wildcard_constraints:
        region=COHORT_RE,
    threads: workflow.cores
    resources:
        mem_mb=60000,
        final_large_scan=1,
    run:
        from marin_dna_vertebrate_projection.projection.dataset import prepare_shards

        prepare_shards(
            parquet_path=str(input[0]),
            shard_paths=[str(path) for path in output],
            add_rc=False,
            shuffle_seed=PUBLICATION_SHUFFLE_SEED,
            max_in_memory_rows=PUBLICATION_MAX_IN_MEMORY_ROWS,
        )


rule phylop_order_prepare_validation_jsonl_shards:
    input:
        f"{RESULTS}/datasets/{{region}}/validation.parquet",
    output:
        temp(
            local(
                expand(
                    f"{HF_RESULTS}/{{{{region}}}}/data/validation/{{shard}}.jsonl",
                    shard=PUBLICATION_VALIDATION_SHARDS,
                )
            )
        ),
    wildcard_constraints:
        region=COHORT_RE,
    threads: workflow.cores
    resources:
        mem_mb=8000,
    run:
        from marin_dna_vertebrate_projection.projection.dataset import prepare_shards

        prepare_shards(
            parquet_path=str(input[0]),
            shard_paths=[str(path) for path in output],
            add_rc=False,
            shuffle_seed=PUBLICATION_SHUFFLE_SEED,
            max_in_memory_rows=PUBLICATION_MAX_IN_MEMORY_ROWS,
        )


rule phylop_order_compress_publication_shard:
    input:
        local(f"{HF_RESULTS}/{{region}}/data/{{split}}/{{shard}}.jsonl"),
    output:
        local(f"{HF_RESULTS}/{{region}}/data/{{split}}/{{shard}}.jsonl.zst"),
    wildcard_constraints:
        region=COHORT_RE,
        split="train|validation",
        shard=r"shard_\d{4}",
    conda:
        "../envs/bioinformatics.yaml"
    threads: 8
    shell:
        "zstd -T{threads} --force {input} -o {output}"


rule phylop_order_dataset_card:
    input:
        train=f"{RESULTS}/datasets/{{region}}/train.parquet",
        validation=f"{RESULTS}/datasets/{{region}}/validation.parquet",
        manifest=ORDER_MANIFEST,
        source_manifest=SOURCE_SPECIES_MANIFEST,
    output:
        local(f"{HF_RESULTS}/{{region}}/README.md"),
    wildcard_constraints:
        region=COHORT_RE,
    resources:
        mem_mb=2000,
    params:
        repo=lambda wc: order_hf_repo(wc.region),
    run:
        write_order_dataset_card(
            input.train,
            input.validation,
            input.manifest,
            input.source_manifest,
            output[0],
            pipeline_commit=PIPELINE_COMMIT,
            hf_repo=params.repo,
            validation_seed=int(config["validation_seed"]),
            source_pipeline_commit=SOURCE_PIPELINE_COMMIT,
            source_config_sha256=SOURCE_CONFIG_SHA256,
        )


rule phylop_order_hf_artifact_manifest:
    input:
        producer=PRODUCER_MANIFEST,
        source_producer=SOURCE_PRODUCER_MANIFEST,
        audit=SOURCE_AUDIT,
        train_source=f"{RESULTS}/datasets/enhancer/train.parquet",
        validation_source=f"{RESULTS}/datasets/enhancer/validation.parquet",
        split_selection=f"{RESULTS}/datasets/enhancer/validation_selection.tsv",
        split_composition=f"{RESULTS}/datasets/enhancer/validation_composition.tsv",
        split_summary=f"{RESULTS}/datasets/enhancer/split_summary.json",
        train=local(
            expand(
                f"{HF_RESULTS}/enhancer/data/train/{{shard}}.jsonl.zst",
                shard=PUBLICATION_TRAIN_SHARDS,
            )
        ),
        validation=local(
            expand(
                f"{HF_RESULTS}/enhancer/data/validation/{{shard}}.jsonl.zst",
                shard=PUBLICATION_VALIDATION_SHARDS,
            )
        ),
        card=local(f"{HF_RESULTS}/enhancer/README.md"),
    output:
        HF_MANIFEST,
    threads: 8
    resources:
        mem_mb=8000,
        final_large_scan=1,
    run:
        assert EXPECTED_SOURCE_ROWS > VALIDATION_ROWS
        validate_producer_manifest(
            input.producer,
            pipeline_commit=PIPELINE_COMMIT,
            config_sha256=PIPELINE_CONFIG_SHA256,
            pipeline_version=PIPELINE_VERSION,
            tier=TIER,
        )
        validate_producer_manifest(
            input.source_producer,
            pipeline_commit=SOURCE_PIPELINE_COMMIT,
            config_sha256=SOURCE_CONFIG_SHA256,
            pipeline_version=SOURCE_PIPELINE_VERSION,
            tier=SOURCE_TIER,
        )
        publication = validate_artifacts(
            HF_RESULTS,
            f"{RESULTS}/datasets",
            output[0],
            config_path="config/phylop_uniform_enhancer_order_publication.yaml",
            pipeline_commit=PIPELINE_COMMIT,
            config_sha256=PIPELINE_CONFIG_SHA256,
            source_artifacts={
                "enhancer": {
                    "train.parquet": input.train_source,
                    "validation.parquet": input.validation_source,
                    "validation_selection.tsv": input.split_selection,
                    "validation_composition.tsv": input.split_composition,
                    "split_summary.json": input.split_summary,
                }
            },
            tier=TIER,
            workers=threads,
        )
        observed = int(
            publication["cohorts"]["enhancer"]["split_summary"]["source_rows"]
        )
        assert observed == EXPECTED_SOURCE_ROWS
        audit = json.loads(Path(input.audit).read_text())
        assert int(audit["source_rows"]) == observed


rule all_phylop_order_hf_files:
    """Build and validate the public artifact without external writes."""
    input:
        HF_MANIFEST,
        local(
            expand(
                f"{HF_RESULTS}/enhancer/data/train/{{shard}}.jsonl.zst",
                shard=PUBLICATION_TRAIN_SHARDS,
            )
        ),
        local(
            expand(
                f"{HF_RESULTS}/enhancer/data/validation/{{shard}}.jsonl.zst",
                shard=PUBLICATION_VALIDATION_SHARDS,
            )
        ),
        local(f"{HF_RESULTS}/enhancer/README.md"),


rule phylop_order_hf_upload_dataset:
    """Upload the validated public dataset and verify its immutable revision."""
    input:
        manifest=HF_MANIFEST,
        train=lambda wc: [
            local(f"{HF_RESULTS}/enhancer/data/train/{shard}.jsonl.zst")
            for shard in PUBLICATION_TRAIN_SHARDS
        ],
        validation=lambda wc: [
            local(f"{HF_RESULTS}/enhancer/data/validation/{shard}.jsonl.zst")
            for shard in PUBLICATION_VALIDATION_SHARDS
        ],
        card=local(f"{HF_RESULTS}/enhancer/README.md"),
    output:
        temp(local(f"{RESULTS}/upload.done/enhancer")),
    resources:
        hf_uploads=1,
    params:
        repo=order_hf_repo("enhancer"),
        workers=int(config["hf_upload_workers"]),
    run:
        upload_validated_dataset(
            HF_RESULTS,
            input.manifest,
            output[0],
            cohort="enhancer",
            repo_id=params.repo,
            workers=params.workers,
        )


rule all_phylop_order_hf:
    """Upload and anonymously verify the one-per-order enhancer dataset."""
    input:
        local(f"{RESULTS}/upload.done/enhancer"),
