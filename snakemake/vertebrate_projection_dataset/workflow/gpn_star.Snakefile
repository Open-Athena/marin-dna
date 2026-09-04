"""Issue #517 GPN-Star-P uniform-grid catalog and projection."""


configfile: "config/gpn_star_p.yaml"


include: "rules/gpn_star_common.smk"
include: "rules/gpn_star_anchors.smk"
include: "rules/staging.smk"
include: "rules/projection.smk"
include: "rules/gpn_star_qc.smk"


rule all:
    """Build and audit the human catalog; performs no cross-species projection."""
    input:
        PRODUCER_MANIFEST,
        ANCHOR_CATALOG_INPUT,
        ASSIGNMENTS,
        f"{RESULTS}/anchors/audit/gpn_threshold_summary.json",
        f"{RESULTS}/anchors/audit/assignment_summary.json",


rule all_projection:
    """Paid/cloud target authorized in issue #517 after catalog and smoke gates."""
    input:
        PRODUCER_MANIFEST,
        ASSIGNMENTS,
        COMBINED_SEQUENCES,
        f"{RESULTS}/qc/per_anchor.parquet",
        f"{RESULTS}/qc/per_anchor_scope.parquet",
        f"{RESULTS}/qc/rejection_counts.parquet",
        f"{RESULTS}/qc/aggregates.parquet",
        f"{RESULTS}/qc/manual_inspection_sample.tsv",
        f"{RESULTS}/qc/manual_inspection_rejections.tsv",
        f"{RESULTS}/qc/manual_inspection.md",
