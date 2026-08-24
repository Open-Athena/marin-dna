"""Issue #517 functional-specialist datasets on the production projection contract."""


configfile: "config/functional_anchors.yaml"


include: "rules/functional_common.smk"
include: "rules/functional_anchors.smk"
include: "rules/staging.smk"
include: "rules/projection.smk"
include: "rules/functional_dataset.smk"


rule all:
    input:
        PRODUCER_MANIFEST,
        ANCHOR_CATALOG_INPUT,
        TRAINING_CATALOG,
        DEFERRED_CATALOG,
        f"{RESULTS}/anchors/audit/feature_summary.tsv",
        f"{RESULTS}/anchors/audit/raw_overlap.tsv",
        f"{RESULTS}/anchors/audit/construction_summary.json",
        f"{RESULTS}/anchors/audit/conservation_summary.json",
        f"{RESULTS}/anchors/audit/preprojection_sample.tsv",
        f"{RESULTS}/anchors/audit/preprojection_review.md",


rule all_projection:
    """Paid/cloud execution target; run only after explicit approval."""
    input:
        COMBINED_SEQUENCES,
        TRAINING_SEQUENCES,
        f"{RESULTS}/anchors/audit/human_sequence.parquet",
        f"{RESULTS}/qc/per_anchor.parquet",
        f"{RESULTS}/qc/aggregates.parquet",
        f"{RESULTS}/qc/manual_inspection.md",
        expand(f"{RESULTS}/datasets/{{region}}/train.parquet", region=COHORTS),
        expand(f"{RESULTS}/datasets/{{region}}/validation.parquet", region=COHORTS),
        expand(f"{HF_RESULTS}/{{region}}/README.md", region=COHORTS),
