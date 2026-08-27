"""Issue #517 strict phyloP-selector control with current center-1 projection."""


configfile: "config/phylop_uniform.yaml"


include: "rules/common.smk"
include: "rules/anchors.smk"


BASE_ANCHOR_CATALOG = f"{RESULTS}/anchors/catalog.parquet"
PHYLOP_ASSIGNMENTS = f"{RESULTS}/anchors/assignments.parquet"
PHYLOP_ASSIGNMENT_SUMMARY = f"{RESULTS}/anchors/audit/assignment_summary.json"
ANCHOR_CATALOG = f"{RESULTS}/anchors/phylop_uniform_catalog.parquet"
ANCHOR_CATALOG_INPUT = ANCHOR_CATALOG


include: "rules/phylop_uniform_anchors.smk"
include: "rules/staging.smk"
include: "rules/projection.smk"
include: "rules/phylop_uniform_qc.smk"


rule all:
    input:
        PRODUCER_MANIFEST,
        ANCHOR_CATALOG_INPUT,
        PHYLOP_ASSIGNMENTS,
        PHYLOP_ASSIGNMENT_SUMMARY,
        COMBINED_SEQUENCES,
        f"{RESULTS}/qc/per_anchor.parquet",
        f"{RESULTS}/qc/per_anchor_scope.parquet",
        f"{RESULTS}/qc/rejection_counts.parquet",
        f"{RESULTS}/qc/aggregates.parquet",
        f"{RESULTS}/qc/manual_inspection_sample.tsv",
        f"{RESULTS}/qc/manual_inspection_rejections.tsv",
        f"{RESULTS}/qc/manual_inspection.md",
