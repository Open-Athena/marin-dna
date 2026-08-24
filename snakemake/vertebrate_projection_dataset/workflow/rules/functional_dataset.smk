"""Issue #517 projection QC, training splits, cards, and opt-in publication."""

from marin_dna_vertebrate_projection.functional_pipeline import (
    write_training_sequences,
)
from marin_dna_vertebrate_projection.functional_projection_review import (
    write_functional_projection_review,
)
from marin_dna_vertebrate_projection.pipeline_io import (
    write_dataset_card,
    write_dataset_split_files,
    write_qc_files,
)
from marin_dna_vertebrate_projection.provenance import validate_producer_manifest
from marin_dna_vertebrate_projection.publication import (
    upload_validated_dataset,
    validate_artifacts,
)

HAL_REJECTIONS = expand(
    f"{RESULTS}/hal/rejected/{{species}}.parquet", species=MAMMALS
) + expand(f"{RESULTS}/hal/sequence_rejected/{{species}}.parquet", species=MAMMALS)
MULTIZ_REJECTIONS = expand(
    f"{RESULTS}/multiz/rejected/{{species}}.parquet", species=NON_MAMMALS
) + expand(
    f"{RESULTS}/multiz/sequence_rejected/{{species}}.parquet",
    species=NON_MAMMALS,
)
ALL_REJECTIONS = HAL_REJECTIONS + MULTIZ_REJECTIONS


rule functional_training_sequences:
    input:
        combined=COMBINED_SEQUENCES,
        training=TRAINING_CATALOG,
    output:
        TRAINING_SEQUENCES,
    resources:
        mem_mb=16000,
        final_large_scan=1,
    run:
        write_training_sequences(input.combined, input.training, output[0])


rule functional_projection_qc:
    input:
        anchors=ANCHOR_CATALOG_INPUT,
        accepted=COMBINED_SEQUENCES,
        manifest=ACTIVE_MANIFEST,
        rejected=ALL_REJECTIONS,
    output:
        per_anchor=f"{RESULTS}/qc/per_anchor.parquet",
        per_scope=f"{RESULTS}/qc/per_anchor_scope.parquet",
        rejections=f"{RESULTS}/qc/rejection_counts.parquet",
        aggregates=f"{RESULTS}/qc/aggregates.parquet",
    resources:
        mem_mb=30000,
        final_large_scan=1,
    run:
        write_qc_files(
            input.anchors,
            input.accepted,
            list(input.rejected),
            input.manifest,
            output.per_anchor,
            output.per_scope,
            output.rejections,
            output.aggregates,
        )


rule functional_projection_inspection:
    input:
        sequences=COMBINED_SEQUENCES,
        rejected=ALL_REJECTIONS,
    output:
        sample=f"{RESULTS}/qc/manual_inspection_sample.tsv",
        rejected=f"{RESULTS}/qc/manual_inspection_rejections.tsv",
        report=f"{RESULTS}/qc/manual_inspection.md",
    resources:
        mem_mb=30000,
        final_large_scan=1,
    run:
        write_functional_projection_review(
            input.sequences,
            list(input.rejected),
            output.sample,
            output.rejected,
            output.report,
            seed=int(config["inspection_seed"]),
            rows_per_arm=int(config["inspection_rows_per_region"]),
            fragmented_rows=int(config["inspection_fragmented_rows"]),
            rejected_rows_per_reason=int(
                config["inspection_rejected_rows_per_reason"]
            ),
        )


rule functional_dataset_splits:
    input:
        TRAINING_SEQUENCES,
    output:
        train=f"{RESULTS}/datasets/{{region}}/train.parquet",
        validation=f"{RESULTS}/datasets/{{region}}/validation.parquet",
        selection=f"{RESULTS}/datasets/{{region}}/validation_selection.tsv",
        composition=f"{RESULTS}/datasets/{{region}}/validation_composition.tsv",
        summary=f"{RESULTS}/datasets/{{region}}/split_summary.json",
    wildcard_constraints:
        region=COHORT_RE,
    resources:
        mem_mb=24000,
        final_large_scan=1,
    run:
        write_dataset_split_files(
            input[0],
            output.train,
            output.validation,
            output.selection,
            output.composition,
            output.summary,
            region_label=wildcards.region,
            species_scope="all",
            add_rc=bool(config["add_rc"]),
            validation_rows=VALIDATION_ROWS,
            seed=int(config["validation_seed"]),
        )


rule functional_prepare_train_jsonl_shards:
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
            shuffle_seed=PUBLICATION_SHUFFLE_SEED,
        )


rule functional_prepare_validation_jsonl_shards:
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
        from marin_dna_vertebrate_projection.projection.dataset import (
            prepare_shards,
        )

        prepare_shards(
            parquet_path=str(input[0]),
            shard_paths=[str(path) for path in output],
            add_rc=False,
            shuffle_seed=PUBLICATION_SHUFFLE_SEED,
        )


rule functional_compress_publication_shard:
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


rule functional_dataset_card:
    input:
        train=f"{RESULTS}/datasets/{{region}}/train.parquet",
        validation=f"{RESULTS}/datasets/{{region}}/validation.parquet",
        manifest=ACTIVE_MANIFEST,
    output:
        f"{HF_RESULTS}/{{region}}/README.md",
    wildcard_constraints:
        region=COHORT_RE,
    resources:
        mem_mb=2000,
    params:
        repo=lambda wc: f"{config['hf_owner']}/{HF_REPO_PREFIX}-{wc.region}",
    run:
        write_dataset_card(
            input.train,
            input.validation,
            input.manifest,
            output[0],
            pipeline_commit=resolve_pipeline_commit(),
            hf_repo=params.repo,
            region_label=wildcards.region,
            species_scope="all",
            validation_seed=int(config["validation_seed"]),
        )


rule functional_hf_artifact_manifest:
    input:
        producer=PRODUCER_MANIFEST,
        train_source=expand(
            f"{RESULTS}/datasets/{{region}}/train.parquet", region=COHORTS
        ),
        validation_source=expand(
            f"{RESULTS}/datasets/{{region}}/validation.parquet", region=COHORTS
        ),
        split_selection=expand(
            f"{RESULTS}/datasets/{{region}}/validation_selection.tsv", region=COHORTS
        ),
        split_composition=expand(
            f"{RESULTS}/datasets/{{region}}/validation_composition.tsv", region=COHORTS
        ),
        split_summary=expand(
            f"{RESULTS}/datasets/{{region}}/split_summary.json", region=COHORTS
        ),
        train=local(
            expand(
                f"{HF_RESULTS}/{{region}}/data/train/{{shard}}.jsonl.zst",
                region=COHORTS,
                shard=PUBLICATION_TRAIN_SHARDS,
            )
        ),
        validation=local(
            expand(
                f"{HF_RESULTS}/{{region}}/data/validation/{{shard}}.jsonl.zst",
                region=COHORTS,
                shard=PUBLICATION_VALIDATION_SHARDS,
            )
        ),
        cards=expand(f"{HF_RESULTS}/{{region}}/README.md", region=COHORTS),
    output:
        HF_MANIFEST,
    threads: 8
    resources:
        mem_mb=8000,
        final_large_scan=1,
    run:
        validate_producer_manifest(
            input.producer,
            pipeline_commit=PIPELINE_COMMIT,
            config_sha256=PIPELINE_CONFIG_SHA256,
            pipeline_version=PIPELINE_VERSION,
            tier=TIER,
        )
        validate_artifacts(
            HF_RESULTS,
            f"{RESULTS}/datasets",
            output[0],
            config_path="config/functional_anchors.yaml",
            pipeline_commit=PIPELINE_COMMIT,
            config_sha256=PIPELINE_CONFIG_SHA256,
            tier=TIER,
            workers=threads,
        )


rule all_functional_hf_files:
    """Build and validate review artifacts without external writes."""
    input:
        HF_MANIFEST,
        local(
            expand(
                f"{HF_RESULTS}/{{region}}/data/train/{{shard}}.jsonl.zst",
                region=COHORTS,
                shard=PUBLICATION_TRAIN_SHARDS,
            )
        ),
        local(
            expand(
                f"{HF_RESULTS}/{{region}}/data/validation/{{shard}}.jsonl.zst",
                region=COHORTS,
                shard=PUBLICATION_VALIDATION_SHARDS,
            )
        ),
        expand(f"{HF_RESULTS}/{{region}}/README.md", region=COHORTS),


rule functional_hf_upload_dataset:
    """Opt-in external write; requires explicit human publication approval."""
    input:
        manifest=HF_MANIFEST,
        train=lambda wc: [
            local(f"{HF_RESULTS}/{wc.region}/data/train/{shard}.jsonl.zst")
            for shard in PUBLICATION_TRAIN_SHARDS
        ],
        validation=lambda wc: [
            local(f"{HF_RESULTS}/{wc.region}/data/validation/{shard}.jsonl.zst")
            for shard in PUBLICATION_VALIDATION_SHARDS
        ],
        card=f"{HF_RESULTS}/{{region}}/README.md",
    output:
        temp(local(f"{RESULTS}/upload.done/{{region}}")),
    wildcard_constraints:
        region=COHORT_RE,
    resources:
        hf_uploads=1,
    params:
        repo=lambda wc: f"{config['hf_owner']}/{HF_REPO_PREFIX}-{wc.region}",
        workers=int(config["hf_upload_workers"]),
    run:
        upload_validated_dataset(
            HF_RESULTS,
            input.manifest,
            output[0],
            cohort=wildcards.region,
            repo_id=params.repo,
            workers=params.workers,
        )


rule all_functional_hf:
    """External-write target; never invoke without explicit human approval."""
    input:
        local(expand(f"{RESULTS}/upload.done/{{region}}", region=COHORTS)),
