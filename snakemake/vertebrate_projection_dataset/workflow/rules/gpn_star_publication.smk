"""Split, validate, and publish the issue #517 GPN-Star-P datasets."""

from marin_dna_vertebrate_projection.gpn_star_publication import (
    rewrite_gpn_star_dataset_card,
)
from marin_dna_vertebrate_projection.pipeline_io import (
    write_dataset_card,
    write_dataset_split_files,
)
from marin_dna_vertebrate_projection.provenance import validate_producer_manifest
from marin_dna_vertebrate_projection.publication import (
    upload_validated_dataset,
    validate_artifacts,
)


rule gpn_dataset_splits:
    input:
        COMBINED_SEQUENCES,
    output:
        train=f"{RESULTS}/datasets/{{region}}/train.parquet",
        validation=f"{RESULTS}/datasets/{{region}}/validation.parquet",
        selection=f"{RESULTS}/datasets/{{region}}/validation_selection.tsv",
        composition=f"{RESULTS}/datasets/{{region}}/validation_composition.tsv",
        summary=f"{RESULTS}/datasets/{{region}}/split_summary.json",
    wildcard_constraints:
        region=COHORT_RE,
    resources:
        mem_mb=48000,
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


rule gpn_prepare_train_jsonl_shards:
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
        from marin_dna_vertebrate_projection.projection.dataset import prepare_shards

        prepare_shards(
            parquet_path=str(input[0]),
            shard_paths=[str(path) for path in output],
            add_rc=False,
            shuffle_seed=PUBLICATION_SHUFFLE_SEED,
        )


rule gpn_prepare_validation_jsonl_shards:
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
        )


rule gpn_compress_publication_shard:
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


rule gpn_dataset_card:
    input:
        train=f"{RESULTS}/datasets/{{region}}/train.parquet",
        validation=f"{RESULTS}/datasets/{{region}}/validation.parquet",
        manifest=ACTIVE_MANIFEST,
    output:
        local(f"{HF_RESULTS}/{{region}}/README.md"),
    wildcard_constraints:
        region=COHORT_RE,
    resources:
        mem_mb=2000,
    params:
        repo=lambda wc: gpn_hf_repo(wc.region),
    run:
        write_dataset_card(
            input.train,
            input.validation,
            input.manifest,
            output[0],
            pipeline_commit=PIPELINE_COMMIT,
            hf_repo=params.repo,
            region_label=wildcards.region,
            species_scope="all",
            validation_seed=int(config["validation_seed"]),
            anchor_provenance=(
                "The authoritative projection table was produced by source commit "
                f"`{SOURCE_PIPELINE_COMMIT}` with source config SHA-256 "
                f"`{SOURCE_CONFIG_SHA256}`. Human anchors use the uniform GRCh38 "
                "255 bp grid at 128 bp stride and require at least 51 positions "
                "with `entropy_calibrated < 0.081001` from the pinned primate "
                "GPN-Star-P score set. The six labels are the issue #232 v4 CDS, "
                "3-prime UTR, protein-coding TSS/5-prime UTR, and ncRNA-exon "
                "assignments; issue #326 Arm A enhancer; and the exhaustive "
                "GPN-constrained remainder background."
            ),
        )
        rewrite_gpn_star_dataset_card(output[0])


rule gpn_hf_artifact_manifest:
    input:
        producer=PRODUCER_MANIFEST,
        source_producer=SOURCE_PRODUCER_MANIFEST,
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
        cards=local(expand(f"{HF_RESULTS}/{{region}}/README.md", region=COHORTS)),
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
            config_path="config/gpn_star_p_publication.yaml",
            pipeline_commit=PIPELINE_COMMIT,
            config_sha256=PIPELINE_CONFIG_SHA256,
            source_artifacts={
                cohort: {
                    "train.parquet": input.train_source[index],
                    "validation.parquet": input.validation_source[index],
                    "validation_selection.tsv": input.split_selection[index],
                    "validation_composition.tsv": input.split_composition[index],
                    "split_summary.json": input.split_summary[index],
                }
                for index, cohort in enumerate(COHORTS)
            },
            tier=TIER,
            workers=threads,
        )
        observed_source_rows = {
            cohort: int(
                publication["cohorts"][cohort]["split_summary"]["source_rows"]
            )
            for cohort in COHORTS
        }
        assert observed_source_rows == SOURCE_ARM_ROWS
        assert sum(observed_source_rows.values()) == SOURCE_ROWS


rule all_gpn_hf_files:
    """Build and validate all six public artifacts without external writes."""
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
        local(expand(f"{HF_RESULTS}/{{region}}/README.md", region=COHORTS)),


rule gpn_hf_upload_dataset:
    """Upload one validated public dataset and verify its immutable revision."""
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
        card=local(f"{HF_RESULTS}/{{region}}/README.md"),
    output:
        temp(local(f"{RESULTS}/upload.done/{{region}}")),
    wildcard_constraints:
        region=COHORT_RE,
    resources:
        hf_uploads=1,
    params:
        repo=lambda wc: gpn_hf_repo(wc.region),
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


rule all_gpn_hf:
    """Upload and verify all six datasets after explicit training approval."""
    input:
        local(expand(f"{RESULTS}/upload.done/{{region}}", region=COHORTS)),
