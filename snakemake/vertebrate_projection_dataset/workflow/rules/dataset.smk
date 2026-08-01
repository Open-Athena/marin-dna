"""QC tables, chromosome-18 splits, cards, and opt-in HF upload targets."""

from marin_dna.pipelines.vertebrate_projection_dataset.pipeline_io import (
    write_dataset_card,
    write_dataset_split_files,
    write_inspection_files,
    write_qc_files,
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


def dataset_region_label(cohort):
    return "cds" if cohort == "cds_mammals_only" else cohort


def dataset_species_scope(cohort):
    return "mammals_only" if cohort == "cds_mammals_only" else "all"


rule projection_qc:
    input:
        anchors=ANCHOR_CATALOG,
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
            validation_chrom=str(config["validation_chrom"]),
        )


rule projection_inspection_report:
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
        write_inspection_files(
            input.sequences,
            list(input.rejected),
            output.sample,
            output.rejected,
            output.report,
            seed=int(config["inspection_seed"]),
            rows_per_region=int(config["inspection_rows_per_region"]),
            fragmented_rows=int(config["inspection_fragmented_rows"]),
            rejected_rows_per_reason=int(
                config["inspection_rejected_rows_per_reason"]
            ),
            require_zrs=TIER == "smoke",
        )


rule dataset_splits:
    input:
        COMBINED_SEQUENCES,
    output:
        train=f"{RESULTS}/datasets/{{region}}/train.parquet",
        validation=f"{RESULTS}/datasets/{{region}}/validation.parquet",
        selection=f"{RESULTS}/datasets/{{region}}/validation_selection.tsv",
        counts=f"{RESULTS}/datasets/{{region}}/validation_species_counts.tsv",
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
            output.counts,
            output.summary,
            region_label=dataset_region_label(wildcards.region),
            species_scope=dataset_species_scope(wildcards.region),
            add_rc=bool(config["add_rc"]),
            validation_chrom=str(config["validation_chrom"]),
            max_validation_rows=int(config["validation_max_rows"]),
            seed=int(config["validation_seed"]),
        )


rule prepare_train_jsonl_shards:
    """Shuffle the augmented train Parquet into established JSONL shards."""
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
        from marin_dna.pipelines.projection.dataset import prepare_shards

        prepare_shards(
            parquet_path=str(input[0]),
            shard_paths=[str(path) for path in output],
            add_rc=False,
            shuffle_seed=PUBLICATION_SHUFFLE_SEED,
        )


rule prepare_validation_jsonl_shards:
    """Shuffle the held-out original-orientation rows into JSONL shards."""
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
        from marin_dna.pipelines.projection.dataset import prepare_shards

        prepare_shards(
            parquet_path=str(input[0]),
            shard_paths=[str(path) for path in output],
            add_rc=False,
            shuffle_seed=PUBLICATION_SHUFFLE_SEED,
        )


rule compress_publication_shard:
    """Compress one JSONL shard with the established zstd encoding."""
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


rule dataset_card:
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
        repo=lambda wc: (f"{config['hf_owner']}/{config['hf_repo_prefix']}-{wc.region}"),
        region=lambda wc: dataset_region_label(wc.region),
        scope=lambda wc: dataset_species_scope(wc.region),
    run:
        write_dataset_card(
            input.train,
            input.validation,
            input.manifest,
            output[0],
            pipeline_commit=resolve_pipeline_commit(),
            hf_repo=params.repo,
            region_label=params.region,
            species_scope=params.scope,
        )


rule all_hf_files:
    """Build reviewed HF artifacts without writing any external state."""
    input:
        expand(
            f"{HF_RESULTS}/{{region}}/data/train/{{shard}}.jsonl.zst",
            region=COHORTS,
            shard=PUBLICATION_TRAIN_SHARDS,
        ),
        expand(
            f"{HF_RESULTS}/{{region}}/data/validation/{{shard}}.jsonl.zst",
            region=COHORTS,
            shard=PUBLICATION_VALIDATION_SHARDS,
        ),
        expand(f"{HF_RESULTS}/{{region}}/README.md", region=COHORTS),


rule hf_upload_dataset:
    """Opt-in only: run after a human approves the generated dataset card."""
    input:
        train=lambda wc: [
            f"{HF_RESULTS}/{wc.region}/data/train/{shard}.jsonl.zst"
            for shard in PUBLICATION_TRAIN_SHARDS
        ],
        validation=lambda wc: [
            f"{HF_RESULTS}/{wc.region}/data/validation/{shard}.jsonl.zst"
            for shard in PUBLICATION_VALIDATION_SHARDS
        ],
        card=f"{HF_RESULTS}/{{region}}/README.md",
    output:
        f"{RESULTS}/upload.done/{{region}}",
    wildcard_constraints:
        region=COHORT_RE,
    resources:
        hf_uploads=1,
    params:
        repo=lambda wc: (f"{config['hf_owner']}/{config['hf_repo_prefix']}-{wc.region}"),
        data_dir=lambda wc: f"{HF_RESULTS}/{wc.region}",
        workers=int(config["hf_upload_workers"]),
    shell:
        """
        HF_XET_HIGH_PERFORMANCE=1 hf upload-large-folder {params.repo} --repo-type dataset --num-workers {params.workers} {params.data_dir}
        HF_XET_HIGH_PERFORMANCE=1 hf upload {params.repo} {input.card} README.md --repo-type dataset
        mkdir -p $(dirname {output})
        touch {output}
        """


rule all_hf:
    input:
        expand(f"{RESULTS}/upload.done/{{region}}", region=COHORTS),
