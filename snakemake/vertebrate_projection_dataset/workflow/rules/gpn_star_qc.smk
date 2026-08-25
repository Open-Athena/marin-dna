"""Projection QC for the GPN-Star-P six-arm catalog."""

from marin_dna_vertebrate_projection.pipeline_io import (
    write_inspection_files,
    write_qc_files,
)

GPN_HAL_REJECTIONS = expand(
    f"{RESULTS}/hal/rejected/{{species}}.parquet", species=MAMMALS
) + expand(f"{RESULTS}/hal/sequence_rejected/{{species}}.parquet", species=MAMMALS)
GPN_MULTIZ_REJECTIONS = expand(
    f"{RESULTS}/multiz/rejected/{{species}}.parquet", species=NON_MAMMALS
) + expand(
    f"{RESULTS}/multiz/sequence_rejected/{{species}}.parquet",
    species=NON_MAMMALS,
)
GPN_ALL_REJECTIONS = GPN_HAL_REJECTIONS + GPN_MULTIZ_REJECTIONS


rule gpn_projection_qc:
    input:
        anchors=ANCHOR_CATALOG_INPUT,
        accepted=COMBINED_SEQUENCES,
        manifest=ACTIVE_MANIFEST,
        rejected=GPN_ALL_REJECTIONS,
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


rule gpn_projection_inspection_report:
    input:
        sequences=COMBINED_SEQUENCES,
        rejected=GPN_ALL_REJECTIONS,
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
            require_zrs=False,
            required_region_labels=tuple(config["assignment_arms"]),
        )
