"""Fixed-anchor pilot and full-scale execution graph for issue #473.

This module is additive. It does not modify or replace any established rule.
The unchanged full-window baseline is restored from #417; only center-seeded
runs and the missing exp351-centered full-window arm execute new projections.
"""

import re

from marin_dna_vertebrate_projection.issue_473.assembly import (
    write_baseline_compatibility_receipt,
    write_full_window_qc_union,
    write_full_window_sequence_union,
    write_intersection_validation_views,
)
from marin_dna_vertebrate_projection.issue_473.catalog import (
    ENHANCER_REGION,
    FIXED_REGIONS,
    write_fixed_scored_anchor_catalog,
)
from marin_dna_vertebrate_projection.issue_473.diagnostics import (
    write_manual_pair_sample,
    write_paired_diagnostics,
)
from marin_dna_vertebrate_projection.issue_473.immutable import (
    read_immutable_sources,
    stage_immutable_s3_object,
    stage_inventory_object,
)

ISSUE_473_FIXED_ROOT = f"{ISSUE_473_ROOT}/fixed"
ISSUE_473_FIXED_LOCAL_ROOT = (
    "/mnt/nvme/vertebrate_projection/issue_473_immutable_sources"
)
ISSUE_473_FIXED_SOURCE_MANIFEST = "config/issue_473_immutable_sources.tsv"
ISSUE_473_FIXED_SOURCE_MANIFEST_INPUT = local(ISSUE_473_FIXED_SOURCE_MANIFEST)
ISSUE_473_FIXED_SOURCES = read_immutable_sources(ISSUE_473_FIXED_SOURCE_MANIFEST)
ISSUE_473_FIXED_SOURCE_FILE_BY_NAME = {
    "exp351_noexon_bed": "exp351_noexon.bed.gz",
    "exp351_scored_anchors": "exp351_scored_anchors.parquet",
    "issue417_artifact_inventory": "issue417_artifact_inventory.tsv",
    "issue417_anchor_labels": "issue417_anchor_labels.parquet",
    "issue417_active_species": "issue417_active_species.tsv",
    "issue417_all_sequences": "issue417_all_sequences.parquet",
    "issue417_per_anchor_qc": "issue417_per_anchor_qc.parquet",
    "issue417_per_scope_qc": "issue417_per_scope_qc.parquet",
    "issue417_rejection_counts": "issue417_rejection_counts.parquet",
    "issue417_aggregates": "issue417_aggregates.parquet",
}
assert set(ISSUE_473_FIXED_SOURCE_FILE_BY_NAME) == set(ISSUE_473_FIXED_SOURCES)
ISSUE_473_FIXED_SOURCE_NAME_BY_FILE = {
    filename: name for name, filename in ISSUE_473_FIXED_SOURCE_FILE_BY_NAME.items()
}
ISSUE_473_FIXED_SOURCE_FILE_RE = "|".join(
    re.escape(filename) for filename in ISSUE_473_FIXED_SOURCE_NAME_BY_FILE
)
ISSUE_473_417_PREFIX = (
    "s3://oa-bolinas/staging/vertebrate_projection_dataset/v1/"
    "06549d8f7f3ba76151b9c54a5e52d3e3f4402a2d/full"
)
ISSUE_473_FIXED_CATALOG = f"{ISSUE_473_FIXED_ROOT}/anchors/fixed_scored.parquet"
ISSUE_473_FIXED_PILOT = f"{ISSUE_473_FIXED_ROOT}/pilot/anchors.parquet"
ISSUE_473_FIXED_RUN_ROOT = f"{ISSUE_473_FIXED_ROOT}/runs"
ISSUE_473_FIXED_FULL_WINDOW = (
    f"{ISSUE_473_FIXED_ROOT}/full_scale/full_window/sequences/all_sources.parquet"
)
ISSUE_473_FIXED_CENTER_1 = (
    f"{ISSUE_473_FIXED_RUN_ROOT}/full_center_1/sequences/all_sources.parquet"
)
ISSUE_473_FIXED_QC_TABLES = {
    "per_anchor": "issue417_per_anchor_qc",
    "per_anchor_scope": "issue417_per_scope_qc",
    "rejection_counts": "issue417_rejection_counts",
    "aggregates": "issue417_aggregates",
}
ISSUE_473_FIXED_QC_RE = "|".join(ISSUE_473_FIXED_QC_TABLES)


def issue_473_fixed_source_path(name):
    filename = ISSUE_473_FIXED_SOURCE_FILE_BY_NAME[name]
    return f"{ISSUE_473_FIXED_LOCAL_ROOT}/direct/{filename}"


rule issue_473_fixed_stage_source:
    input:
        ISSUE_473_FIXED_SOURCE_MANIFEST_INPUT,
    output:
        artifact=local(f"{ISSUE_473_FIXED_LOCAL_ROOT}/direct/{{source_file}}"),
        receipt=local(
            f"{ISSUE_473_FIXED_LOCAL_ROOT}/receipts/direct/" "{source_file}.json"
        ),
    wildcard_constraints:
        source_file=ISSUE_473_FIXED_SOURCE_FILE_RE,
    resources:
        mem_mb=1000,
    params:
        source=lambda wc: ISSUE_473_FIXED_SOURCES[
            ISSUE_473_FIXED_SOURCE_NAME_BY_FILE[wc.source_file]
        ],
    run:
        stage_immutable_s3_object(
            params.source,
            output.artifact,
            output.receipt,
        )


rule issue_473_fixed_stage_417_scored:
    input:
        inventory=local(issue_473_fixed_source_path("issue417_artifact_inventory")),
    output:
        artifact=local(
            f"{ISSUE_473_FIXED_LOCAL_ROOT}/issue417/scored/{{chrom}}.parquet"
        ),
        receipt=local(
            f"{ISSUE_473_FIXED_LOCAL_ROOT}/receipts/issue417/scored/" "{chrom}.json"
        ),
    wildcard_constraints:
        chrom=FULL_CHROM_RE,
    resources:
        mem_mb=1000,
    run:
        stage_inventory_object(
            source_prefix=ISSUE_473_417_PREFIX,
            inventory_path=input.inventory,
            relative_path=f"anchors/scored/{wildcards.chrom}.parquet",
            destination=output.artifact,
            receipt_path=output.receipt,
        )


rule issue_473_fixed_stage_417_rejection:
    input:
        inventory=local(issue_473_fixed_source_path("issue417_artifact_inventory")),
    output:
        artifact=local(
            f"{ISSUE_473_FIXED_LOCAL_ROOT}/issue417/rejections/"
            "{backend}/{kind}/{species}.parquet"
        ),
        receipt=local(
            f"{ISSUE_473_FIXED_LOCAL_ROOT}/receipts/issue417/rejections/"
            "{backend}/{kind}/{species}.json"
        ),
    wildcard_constraints:
        backend="hal|multiz",
        kind="rejected|sequence_rejected",
        species=f"{MAMMAL_RE}|{NON_MAMMAL_RE}",
    resources:
        mem_mb=1000,
    run:
        stage_inventory_object(
            source_prefix=ISSUE_473_417_PREFIX,
            inventory_path=input.inventory,
            relative_path=(
                f"{wildcards.backend}/{wildcards.kind}/"
                f"{wildcards.species}.parquet"
            ),
            destination=output.artifact,
            receipt_path=output.receipt,
        )


rule issue_473_fixed_anchor_catalog:
    input:
        labels=local(issue_473_fixed_source_path("issue417_anchor_labels")),
        scored=local(
            expand(
                f"{ISSUE_473_FIXED_LOCAL_ROOT}/issue417/scored/{{chrom}}.parquet",
                chrom=FULL_CHROMS,
            )
        ),
        enhancer_bed=local(issue_473_fixed_source_path("exp351_noexon_bed")),
        enhancer_scored=local(issue_473_fixed_source_path("exp351_scored_anchors")),
    output:
        catalog=ISSUE_473_FIXED_CATALOG,
        summary=f"{ISSUE_473_FIXED_ROOT}/anchors/summary.json",
    resources:
        mem_mb=24000,
        final_large_scan=1,
    run:
        write_fixed_scored_anchor_catalog(
            input.labels,
            list(input.scored),
            input.enhancer_bed,
            input.enhancer_scored,
            output.catalog,
            output.summary,
            target_length=TARGET_LENGTH,
            min_proportion_conserved=MIN_PROPORTION,
            expected_enhancer_anchors=116_162,
        )


rule issue_473_fixed_pilot_sample:
    input:
        ISSUE_473_FIXED_CATALOG,
    output:
        anchors=ISSUE_473_FIXED_PILOT,
        selection=f"{ISSUE_473_FIXED_ROOT}/pilot/selection.tsv",
        strata=f"{ISSUE_473_FIXED_ROOT}/pilot/stratum_counts.tsv",
    resources:
        mem_mb=8000,
    run:
        sample = sample_projection_pilot_anchors(
            pl.read_parquet(input[0]),
            regions=FIXED_REGIONS,
            max_per_region=ISSUE_473_MAX_ANCHORS_PER_REGION,
            conservation_quantiles=ISSUE_473_CONSERVATION_QUANTILES,
            seed=ISSUE_473_SAMPLE_SEED,
        )
        Path(output.anchors).parent.mkdir(parents=True, exist_ok=True)
        sample.anchors.write_parquet(output.anchors)
        sample.selection_manifest.write_csv(output.selection, separator="\t")
        sample.stratum_counts.write_csv(output.strata, separator="\t")


ISSUE_473_FIXED_PILOT_RUNS = [f"pilot_{policy}" for policy in ISSUE_473_POLICIES]
ISSUE_473_FIXED_RUN_POLICIES = {
    **{f"pilot_{policy}": policy for policy in ISSUE_473_POLICIES},
    "full_center_1": "center_1",
    "full_enhancer_full_window": "full_window",
}
ISSUE_473_FIXED_RUNS = list(ISSUE_473_FIXED_RUN_POLICIES)
ISSUE_473_FIXED_RUN_RE = "|".join(ISSUE_473_FIXED_RUNS)


def issue_473_fixed_policy(run_name):
    return policy_by_name(ISSUE_473_FIXED_RUN_POLICIES[run_name])


def issue_473_fixed_anchor_input(wildcards):
    if wildcards.run.startswith("pilot_"):
        return ISSUE_473_FIXED_PILOT
    return ISSUE_473_FIXED_CATALOG


rule issue_473_fixed_projection_requests:
    input:
        issue_473_fixed_anchor_input,
    output:
        f"{ISSUE_473_FIXED_RUN_ROOT}/{{run}}/requests.parquet",
    wildcard_constraints:
        run=ISSUE_473_FIXED_RUN_RE,
    run:
        anchors = pl.read_parquet(input[0])
        if wildcards.run == "full_enhancer_full_window":
            anchors = anchors.filter(pl.col("region_label") == ENHANCER_REGION)
        requests = build_projection_requests(
            anchors,
            issue_473_fixed_policy(wildcards.run),
        )
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        requests.write_parquet(output[0])


rule issue_473_fixed_hal_request_bed:
    input:
        f"{ISSUE_473_FIXED_RUN_ROOT}/{{run}}/requests.parquet",
    output:
        local(f"{ISSUE_473_FIXED_RUN_ROOT}/{{run}}/hal/input.bed"),
    wildcard_constraints:
        run=ISSUE_473_FIXED_RUN_RE,
    run:
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        write_hal_request_bed6(input[0], output[0])


rule issue_473_fixed_human_sequences:
    input:
        anchors=f"{ISSUE_473_FIXED_RUN_ROOT}/{{run}}/requests.parquet",
        twobit=f"{RESULTS}/reference/hg38.2bit",
        sizes=f"{RESULTS}/reference/hg38.chrom.sizes",
    output:
        f"{ISSUE_473_FIXED_RUN_ROOT}/{{run}}/sequences/human_reference.parquet",
    wildcard_constraints:
        run=ISSUE_473_FIXED_RUN_RE,
    conda:
        "../envs/bioinformatics.yaml"
    resources:
        mem_mb=4000,
    shell:
        "uv run python -m "
        "marin_dna_vertebrate_projection.sequence_cli "
        "human {input.anchors} {input.twobit} {input.sizes} {output}"


rule issue_473_fixed_hal_liftover:
    input:
        hal=local(HAL_PATH),
        bed=local(f"{ISSUE_473_FIXED_RUN_ROOT}/{{run}}/hal/input.bed"),
        validation=local(HAL_VALIDATION),
    output:
        f"{ISSUE_473_FIXED_RUN_ROOT}/{{run}}/hal/raw/{{species}}.bed",
    wildcard_constraints:
        run=ISSUE_473_FIXED_RUN_RE,
        species=MAMMAL_RE,
    threads: 1
    resources:
        mem_mb=2000,
    run:
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        run_halliftover(
            input.hal,
            "Homo_sapiens",
            input.bed,
            wildcards.species,
            output[0],
            no_dupes=True,
        )


rule issue_473_fixed_hal_fragments:
    input:
        raw=f"{ISSUE_473_FIXED_RUN_ROOT}/{{run}}/hal/raw/{{species}}.bed",
        sizes=f"{RESULTS}/hal/chrom_sizes/{{species}}.tsv",
        requests=f"{ISSUE_473_FIXED_RUN_ROOT}/{{run}}/requests.parquet",
        manifest=ACTIVE_MANIFEST,
    output:
        f"{ISSUE_473_FIXED_RUN_ROOT}/{{run}}/hal/fragments/{{species}}.parquet",
    wildcard_constraints:
        run=ISSUE_473_FIXED_RUN_RE,
        species=MAMMAL_RE,
    resources:
        mem_mb=8000,
    run:
        records = attach_src_size(
            parse_halliftover_bed(input.raw, wildcards.species),
            input.sizes,
        )
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        write_hal_fragments(records, input.requests, input.manifest, output[0])


rule issue_473_fixed_hal_contract:
    input:
        f"{ISSUE_473_FIXED_RUN_ROOT}/{{run}}/hal/fragments/{{species}}.parquet",
    output:
        accepted=(
            f"{ISSUE_473_FIXED_RUN_ROOT}/{{run}}/" "hal/accepted/{species}.parquet"
        ),
        rejected=(
            f"{ISSUE_473_FIXED_RUN_ROOT}/{{run}}/" "hal/rejected/{species}.parquet"
        ),
    wildcard_constraints:
        run=ISSUE_473_FIXED_RUN_RE,
        species=MAMMAL_RE,
    resources:
        mem_mb=10000,
    run:
        policy = issue_473_fixed_policy(wildcards.run)
        write_contract_outputs(
            input[0],
            output.accepted,
            output.rejected,
            target_length=TARGET_LENGTH,
            pre_resize_min_length=policy.pre_resize_min_length,
            pre_resize_max_length=policy.pre_resize_max_length,
        )


rule issue_473_fixed_hal_sequences:
    input:
        accepted=(
            f"{ISSUE_473_FIXED_RUN_ROOT}/{{run}}/" "hal/accepted/{species}.parquet"
        ),
        twobit=f"{RESULTS}/hal/genomes/{{species}}.2bit",
    output:
        sequences=(
            f"{ISSUE_473_FIXED_RUN_ROOT}/{{run}}/" "sequences/hal/{species}.parquet"
        ),
        rejected=(
            f"{ISSUE_473_FIXED_RUN_ROOT}/{{run}}/"
            "hal/sequence_rejected/{species}.parquet"
        ),
    wildcard_constraints:
        run=ISSUE_473_FIXED_RUN_RE,
        species=MAMMAL_RE,
    conda:
        "../envs/bioinformatics.yaml"
    resources:
        mem_mb=4000,
    shell:
        "uv run python -m "
        "marin_dna_vertebrate_projection.sequence_cli "
        "projected {input.accepted} {input.twobit} "
        "{output.sequences} {output.rejected}"


rule issue_473_fixed_multiz_candidates:
    input:
        maf=local(f"{MULTIZ_STAGE_DIR}/maf/{{chrom}}.maf.gz"),
        requests=f"{ISSUE_473_FIXED_RUN_ROOT}/{{run}}/requests.parquet",
        manifest=ACTIVE_MANIFEST,
    output:
        (f"{ISSUE_473_FIXED_RUN_ROOT}/{{run}}/" "multiz/fragments/{chrom}.parquet"),
    wildcard_constraints:
        run=ISSUE_473_FIXED_RUN_RE,
        chrom=CHROM_RE,
    threads: 4
    resources:
        mem_mb=16000,
    run:
        write_maf_request_candidates(
            input.maf,
            input.requests,
            input.manifest,
            output[0],
        )


rule issue_473_fixed_multiz_contract:
    input:
        (f"{ISSUE_473_FIXED_RUN_ROOT}/{{run}}/" "multiz/fragments/{chrom}.parquet"),
    output:
        accepted=(
            f"{ISSUE_473_FIXED_RUN_ROOT}/{{run}}/"
            "multiz/accepted/by_chrom/{chrom}/{species}.parquet"
        ),
        rejected=(
            f"{ISSUE_473_FIXED_RUN_ROOT}/{{run}}/"
            "multiz/rejected/by_chrom/{chrom}/{species}.parquet"
        ),
    wildcard_constraints:
        run=ISSUE_473_FIXED_RUN_RE,
        chrom=CHROM_RE,
        species=NON_MAMMAL_RE,
    resources:
        mem_mb=8000,
    run:
        policy = issue_473_fixed_policy(wildcards.run)
        write_contract_outputs_for_alignment(
            input[0],
            wildcards.species,
            output.accepted,
            output.rejected,
            target_length=TARGET_LENGTH,
            pre_resize_min_length=policy.pre_resize_min_length,
            pre_resize_max_length=policy.pre_resize_max_length,
        )


rule issue_473_fixed_merge_multiz_accepted:
    input:
        lambda wc: expand(
            (
                f"{ISSUE_473_FIXED_RUN_ROOT}/{{run}}/"
                "multiz/accepted/by_chrom/{chrom}/{species}.parquet"
            ),
            run=[wc.run],
            chrom=CHROMS,
            species=[wc.species],
        ),
    output:
        (f"{ISSUE_473_FIXED_RUN_ROOT}/{{run}}/" "multiz/accepted/{species}.parquet"),
    wildcard_constraints:
        run=ISSUE_473_FIXED_RUN_RE,
        species=NON_MAMMAL_RE,
    resources:
        mem_mb=4000,
    run:
        merge_parquets_streaming(list(input), output[0])


rule issue_473_fixed_merge_multiz_rejected:
    input:
        lambda wc: expand(
            (
                f"{ISSUE_473_FIXED_RUN_ROOT}/{{run}}/"
                "multiz/rejected/by_chrom/{chrom}/{species}.parquet"
            ),
            run=[wc.run],
            chrom=CHROMS,
            species=[wc.species],
        ),
    output:
        (f"{ISSUE_473_FIXED_RUN_ROOT}/{{run}}/" "multiz/rejected/{species}.parquet"),
    wildcard_constraints:
        run=ISSUE_473_FIXED_RUN_RE,
        species=NON_MAMMAL_RE,
    resources:
        mem_mb=4000,
    run:
        merge_parquets_streaming(list(input), output[0])


rule issue_473_fixed_validate_multiz_twobit:
    input:
        accepted=(
            f"{ISSUE_473_FIXED_RUN_ROOT}/{{run}}/" "multiz/accepted/{species}.parquet"
        ),
        sizes=f"{RESULTS}/multiz/genomes/{{species}}.chrom.sizes",
    output:
        (
            f"{ISSUE_473_FIXED_RUN_ROOT}/{{run}}/"
            "multiz/genomes/{species}.compatibility.json"
        ),
    wildcard_constraints:
        run=ISSUE_473_FIXED_RUN_RE,
        species=NON_MAMMAL_RE,
    run:
        validate_projected_twobit_sizes(input.accepted, input.sizes, output[0])


rule issue_473_fixed_multiz_sequences:
    input:
        accepted=(
            f"{ISSUE_473_FIXED_RUN_ROOT}/{{run}}/" "multiz/accepted/{species}.parquet"
        ),
        twobit=f"{RESULTS}/multiz/genomes/{{species}}.2bit",
        compatibility=(
            f"{ISSUE_473_FIXED_RUN_ROOT}/{{run}}/"
            "multiz/genomes/{species}.compatibility.json"
        ),
    output:
        sequences=(
            f"{ISSUE_473_FIXED_RUN_ROOT}/{{run}}/" "sequences/multiz/{species}.parquet"
        ),
        rejected=(
            f"{ISSUE_473_FIXED_RUN_ROOT}/{{run}}/"
            "multiz/sequence_rejected/{species}.parquet"
        ),
    wildcard_constraints:
        run=ISSUE_473_FIXED_RUN_RE,
        species=NON_MAMMAL_RE,
    conda:
        "../envs/bioinformatics.yaml"
    resources:
        mem_mb=4000,
    shell:
        "uv run python -m "
        "marin_dna_vertebrate_projection.sequence_cli "
        "projected {input.accepted} {input.twobit} "
        "{output.sequences} {output.rejected}"


rule issue_473_fixed_combine_sequences:
    input:
        lambda wc: [
            f"{ISSUE_473_FIXED_RUN_ROOT}/{wc.run}/" "sequences/human_reference.parquet"
        ]
        + expand(
            (f"{ISSUE_473_FIXED_RUN_ROOT}/{{run}}/" "sequences/hal/{species}.parquet"),
            run=[wc.run],
            species=MAMMALS,
        )
        + expand(
            (
                f"{ISSUE_473_FIXED_RUN_ROOT}/{{run}}/"
                "sequences/multiz/{species}.parquet"
            ),
            run=[wc.run],
            species=NON_MAMMALS,
        ),
    output:
        f"{ISSUE_473_FIXED_RUN_ROOT}/{{run}}/sequences/all_sources.parquet",
    wildcard_constraints:
        run=ISSUE_473_FIXED_RUN_RE,
    resources:
        mem_mb=12000,
    run:
        combine_sequence_parquets(list(input), output[0])


def issue_473_fixed_rejections_for_run(run_name):
    run_root = f"{ISSUE_473_FIXED_RUN_ROOT}/{run_name}"
    return (
        expand(
            f"{run_root}/hal/rejected/{{species}}.parquet",
            species=MAMMALS,
        )
        + expand(
            f"{run_root}/hal/sequence_rejected/{{species}}.parquet",
            species=MAMMALS,
        )
        + expand(
            f"{run_root}/multiz/rejected/{{species}}.parquet",
            species=NON_MAMMALS,
        )
        + expand(
            f"{run_root}/multiz/sequence_rejected/{{species}}.parquet",
            species=NON_MAMMALS,
        )
    )


def issue_473_fixed_qc_rejections(wildcards):
    return issue_473_fixed_rejections_for_run(wildcards.run)


rule issue_473_fixed_projection_qc:
    input:
        requests=f"{ISSUE_473_FIXED_RUN_ROOT}/{{run}}/requests.parquet",
        accepted=(
            f"{ISSUE_473_FIXED_RUN_ROOT}/{{run}}/" "sequences/all_sources.parquet"
        ),
        manifest=ACTIVE_MANIFEST,
        rejected=issue_473_fixed_qc_rejections,
    output:
        per_anchor=(f"{ISSUE_473_FIXED_RUN_ROOT}/{{run}}/qc/per_anchor.parquet"),
        per_scope=(f"{ISSUE_473_FIXED_RUN_ROOT}/{{run}}/qc/per_anchor_scope.parquet"),
        rejections=(f"{ISSUE_473_FIXED_RUN_ROOT}/{{run}}/qc/rejection_counts.parquet"),
        aggregates=(f"{ISSUE_473_FIXED_RUN_ROOT}/{{run}}/qc/aggregates.parquet"),
    wildcard_constraints:
        run=ISSUE_473_FIXED_RUN_RE,
    resources:
        mem_mb=12000,
        final_large_scan=1,
    run:
        write_qc_files(
            input.requests,
            input.accepted,
            list(input.rejected),
            input.manifest,
            output.per_anchor,
            output.per_scope,
            output.rejections,
            output.aggregates,
            validation_chrom=str(config["validation_chrom"]),
        )


rule issue_473_fixed_pilot_comparison:
    input:
        sequences=[
            f"{ISSUE_473_FIXED_RUN_ROOT}/{run}/sequences/all_sources.parquet"
            for run in ISSUE_473_FIXED_PILOT_RUNS
        ],
        per_anchor=[
            f"{ISSUE_473_FIXED_RUN_ROOT}/{run}/qc/per_anchor.parquet"
            for run in ISSUE_473_FIXED_PILOT_RUNS
        ],
    output:
        summary=f"{ISSUE_473_FIXED_ROOT}/pilot/qc/policy_summary.parquet",
        pairwise=f"{ISSUE_473_FIXED_ROOT}/pilot/qc/full_window_pairwise.parquet",
    resources:
        mem_mb=12000,
        final_large_scan=1,
    run:
        write_policy_comparison(
            dict(zip(ISSUE_473_POLICIES, input.sequences, strict=True)),
            dict(zip(ISSUE_473_POLICIES, input.per_anchor, strict=True)),
            output.summary,
            output.pairwise,
            baseline_policy="full_window",
            target_length=TARGET_LENGTH,
        )


rule issue_473_fixed_pilot_diagnostics:
    input:
        full=(
            f"{ISSUE_473_FIXED_RUN_ROOT}/pilot_full_window/"
            "sequences/all_sources.parquet"
        ),
        center=(
            f"{ISSUE_473_FIXED_RUN_ROOT}/pilot_center_1/"
            "sequences/all_sources.parquet"
        ),
        full_rejected=lambda wc: issue_473_fixed_rejections_for_run("pilot_full_window"),
        center_rejected=lambda wc: issue_473_fixed_rejections_for_run("pilot_center_1"),
        anchors=ISSUE_473_FIXED_PILOT,
    output:
        paired=f"{ISSUE_473_FIXED_ROOT}/pilot/qc/paired_union.parquet",
        scopes=f"{ISSUE_473_FIXED_ROOT}/pilot/qc/paired_scopes.parquet",
        anchors=f"{ISSUE_473_FIXED_ROOT}/pilot/qc/paired_anchors.parquet",
        uncertainty=(f"{ISSUE_473_FIXED_ROOT}/pilot/qc/" "anchor_uncertainty.parquet"),
    resources:
        mem_mb=16000,
        final_large_scan=1,
    run:
        write_paired_diagnostics(
            input.full,
            input.center,
            list(input.full_rejected),
            list(input.center_rejected),
            input.anchors,
            output.paired,
            output.scopes,
            output.anchors,
            output.uncertainty,
        )


rule issue_473_fixed_pilot_inspection:
    input:
        f"{ISSUE_473_FIXED_ROOT}/pilot/qc/paired_union.parquet",
    output:
        sample=f"{ISSUE_473_FIXED_ROOT}/pilot/qc/manual_sample.tsv",
        report=f"{ISSUE_473_FIXED_ROOT}/pilot/qc/manual_inspection.md",
    resources:
        mem_mb=4000,
    run:
        write_manual_pair_sample(
            input[0],
            output.sample,
            output.report,
            seed=ISSUE_473_SAMPLE_SEED,
        )


def issue_473_417_rejection_inputs():
    root = f"{ISSUE_473_FIXED_LOCAL_ROOT}/issue417/rejections"
    return (
        local(expand(f"{root}/hal/rejected/{{species}}.parquet", species=MAMMALS))
        + local(
            expand(
                f"{root}/hal/sequence_rejected/{{species}}.parquet",
                species=MAMMALS,
            )
        )
        + local(
            expand(f"{root}/multiz/rejected/{{species}}.parquet", species=NON_MAMMALS)
        )
        + local(
            expand(
                f"{root}/multiz/sequence_rejected/{{species}}.parquet",
                species=NON_MAMMALS,
            )
        )
    )


rule issue_473_fixed_baseline_compatibility:
    input:
        labels=local(issue_473_fixed_source_path("issue417_anchor_labels")),
        old_species=local(issue_473_fixed_source_path("issue417_active_species")),
        current_species=ACTIVE_MANIFEST,
        catalog=ISSUE_473_FIXED_CATALOG,
    output:
        f"{ISSUE_473_FIXED_ROOT}/full_scale/baseline_compatibility.json",
    resources:
        mem_mb=4000,
    run:
        write_baseline_compatibility_receipt(
            input.labels,
            input.old_species,
            input.current_species,
            input.catalog,
            output[0],
        )


rule issue_473_fixed_full_window_sequences:
    input:
        baseline=local(issue_473_fixed_source_path("issue417_all_sequences")),
        enhancer=(
            f"{ISSUE_473_FIXED_RUN_ROOT}/full_enhancer_full_window/"
            "sequences/all_sources.parquet"
        ),
        compatibility=(
            f"{ISSUE_473_FIXED_ROOT}/full_scale/" "baseline_compatibility.json"
        ),
    output:
        ISSUE_473_FIXED_FULL_WINDOW,
    resources:
        mem_mb=16000,
        final_large_scan=1,
    run:
        write_full_window_sequence_union(
            input.baseline,
            input.enhancer,
            output[0],
        )


rule issue_473_fixed_full_window_qc:
    input:
        baseline=lambda wc: local(
            issue_473_fixed_source_path(ISSUE_473_FIXED_QC_TABLES[wc.qc_table])
        ),
        enhancer=(
            f"{ISSUE_473_FIXED_RUN_ROOT}/full_enhancer_full_window/"
            "qc/{qc_table}.parquet"
        ),
    output:
        (f"{ISSUE_473_FIXED_ROOT}/full_scale/full_window/" "qc/{qc_table}.parquet"),
    wildcard_constraints:
        qc_table=ISSUE_473_FIXED_QC_RE,
    resources:
        mem_mb=8000,
        final_large_scan=1,
    run:
        write_full_window_qc_union(
            input.baseline,
            input.enhancer,
            output[0],
        )


rule issue_473_fixed_full_diagnostics:
    input:
        full=ISSUE_473_FIXED_FULL_WINDOW,
        center=ISSUE_473_FIXED_CENTER_1,
        full_rejected=lambda wc: (
            issue_473_417_rejection_inputs()
            + issue_473_fixed_rejections_for_run("full_enhancer_full_window")
        ),
        center_rejected=lambda wc: issue_473_fixed_rejections_for_run("full_center_1"),
        anchors=ISSUE_473_FIXED_CATALOG,
    output:
        paired=(f"{ISSUE_473_FIXED_ROOT}/full_scale/qc/paired_union.parquet"),
        scopes=(f"{ISSUE_473_FIXED_ROOT}/full_scale/qc/paired_scopes.parquet"),
        anchors=(f"{ISSUE_473_FIXED_ROOT}/full_scale/qc/paired_anchors.parquet"),
        uncertainty=(
            f"{ISSUE_473_FIXED_ROOT}/full_scale/qc/" "anchor_uncertainty.parquet"
        ),
    resources:
        mem_mb=24000,
        final_large_scan=1,
    run:
        write_paired_diagnostics(
            input.full,
            input.center,
            list(input.full_rejected),
            list(input.center_rejected),
            input.anchors,
            output.paired,
            output.scopes,
            output.anchors,
            output.uncertainty,
        )


rule issue_473_fixed_full_inspection:
    input:
        f"{ISSUE_473_FIXED_ROOT}/full_scale/qc/paired_union.parquet",
    output:
        sample=f"{ISSUE_473_FIXED_ROOT}/full_scale/qc/manual_sample.tsv",
        report=(f"{ISSUE_473_FIXED_ROOT}/full_scale/qc/manual_inspection.md"),
    resources:
        mem_mb=4000,
    run:
        write_manual_pair_sample(
            input[0],
            output.sample,
            output.report,
            seed=ISSUE_473_SAMPLE_SEED,
        )


def issue_473_fixed_dataset_source(wildcards):
    if wildcards.projection_policy == "full_window":
        return ISSUE_473_FIXED_FULL_WINDOW
    assert wildcards.projection_policy == "center_1"
    return ISSUE_473_FIXED_CENTER_1


rule issue_473_fixed_dataset:
    input:
        issue_473_fixed_dataset_source,
    output:
        train=(
            f"{ISSUE_473_FIXED_ROOT}/full_scale/datasets/"
            "{projection_policy}/{region}/train.parquet"
        ),
        validation=(
            f"{ISSUE_473_FIXED_ROOT}/full_scale/datasets/"
            "{projection_policy}/{region}/validation.parquet"
        ),
        selection=(
            f"{ISSUE_473_FIXED_ROOT}/full_scale/datasets/"
            "{projection_policy}/{region}/validation_selection.tsv"
        ),
        counts=(
            f"{ISSUE_473_FIXED_ROOT}/full_scale/datasets/"
            "{projection_policy}/{region}/validation_species_counts.tsv"
        ),
        summary=(
            f"{ISSUE_473_FIXED_ROOT}/full_scale/datasets/"
            "{projection_policy}/{region}/split_summary.json"
        ),
    wildcard_constraints:
        projection_policy="full_window|center_1",
        region=f"cds|{ENHANCER_REGION}",
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
            region_label=wildcards.region,
            add_rc=bool(config["add_reverse_complement"]),
            validation_chrom=str(config["validation_chrom"]),
            max_validation_rows=int(config["max_validation_rows"]),
            seed=int(config["split_seed"]),
        )


rule issue_473_fixed_intersection_validation:
    input:
        full=ISSUE_473_FIXED_FULL_WINDOW,
        center=ISSUE_473_FIXED_CENTER_1,
    output:
        full=(
            f"{ISSUE_473_FIXED_ROOT}/full_scale/intersection/"
            "{region}/full_window_validation.parquet"
        ),
        center=(
            f"{ISSUE_473_FIXED_ROOT}/full_scale/intersection/"
            "{region}/center_1_validation.parquet"
        ),
        selection=(
            f"{ISSUE_473_FIXED_ROOT}/full_scale/intersection/" "{region}/selection.tsv"
        ),
        summary=(
            f"{ISSUE_473_FIXED_ROOT}/full_scale/intersection/" "{region}/summary.json"
        ),
    wildcard_constraints:
        region=f"cds|{ENHANCER_REGION}",
    resources:
        mem_mb=24000,
        final_large_scan=1,
    run:
        write_intersection_validation_views(
            input.full,
            input.center,
            output.full,
            output.center,
            output.selection,
            output.summary,
            region_label=wildcards.region,
            validation_chrom=str(config["validation_chrom"]),
            max_validation_rows=int(config["max_validation_rows"]),
            seed=ISSUE_473_SAMPLE_SEED,
        )


def issue_473_fixed_pilot_outputs(_wildcards):
    assert TIER == "full", (
        "issue_473_fixed_landmark_pilot requires tier=full so the fixed "
        "sample is evaluated against the complete species and chromosome cohort"
    )
    return (
        [
            f"{ISSUE_473_FIXED_ROOT}/anchors/summary.json",
            f"{ISSUE_473_FIXED_ROOT}/pilot/selection.tsv",
            f"{ISSUE_473_FIXED_ROOT}/pilot/stratum_counts.tsv",
            f"{ISSUE_473_FIXED_ROOT}/pilot/qc/policy_summary.parquet",
            f"{ISSUE_473_FIXED_ROOT}/pilot/qc/full_window_pairwise.parquet",
            f"{ISSUE_473_FIXED_ROOT}/pilot/qc/paired_union.parquet",
            f"{ISSUE_473_FIXED_ROOT}/pilot/qc/paired_scopes.parquet",
            f"{ISSUE_473_FIXED_ROOT}/pilot/qc/anchor_uncertainty.parquet",
            f"{ISSUE_473_FIXED_ROOT}/pilot/qc/manual_inspection.md",
        ]
        + [
            f"{ISSUE_473_FIXED_RUN_ROOT}/{run}/" "sequences/all_sources.parquet"
            for run in ISSUE_473_FIXED_PILOT_RUNS
        ]
        + [
            f"{ISSUE_473_FIXED_RUN_ROOT}/{run}/qc/aggregates.parquet"
            for run in ISSUE_473_FIXED_PILOT_RUNS
        ]
    )


rule issue_473_fixed_landmark_pilot:
    """Run the six-policy fixed five-region pilot at full cohort scale."""
    input:
        issue_473_fixed_pilot_outputs,


def issue_473_fixed_experiment_outputs(_wildcards):
    assert TIER == "full", "issue_473_fixed_projection_experiment requires tier=full"
    dataset_pairs = [
        ("center_1", "cds"),
        ("center_1", ENHANCER_REGION),
        ("full_window", ENHANCER_REGION),
    ]
    return (
        issue_473_fixed_pilot_outputs(_wildcards)
        + [
            PRODUCER_MANIFEST,
            f"{ISSUE_473_FIXED_ROOT}/full_scale/" "baseline_compatibility.json",
            ISSUE_473_FIXED_FULL_WINDOW,
            ISSUE_473_FIXED_CENTER_1,
            f"{ISSUE_473_FIXED_ROOT}/full_scale/" "full_window/qc/per_anchor.parquet",
            f"{ISSUE_473_FIXED_ROOT}/full_scale/"
            "full_window/qc/per_anchor_scope.parquet",
            f"{ISSUE_473_FIXED_ROOT}/full_scale/"
            "full_window/qc/rejection_counts.parquet",
            f"{ISSUE_473_FIXED_ROOT}/full_scale/" "full_window/qc/aggregates.parquet",
            f"{ISSUE_473_FIXED_RUN_ROOT}/full_center_1/" "qc/aggregates.parquet",
            f"{ISSUE_473_FIXED_ROOT}/full_scale/qc/paired_union.parquet",
            f"{ISSUE_473_FIXED_ROOT}/full_scale/qc/paired_scopes.parquet",
            f"{ISSUE_473_FIXED_ROOT}/full_scale/qc/" "anchor_uncertainty.parquet",
            f"{ISSUE_473_FIXED_ROOT}/full_scale/qc/" "manual_inspection.md",
        ]
        + [
            f"{ISSUE_473_FIXED_ROOT}/full_scale/datasets/"
            f"{policy}/{region}/{filename}"
            for policy, region in dataset_pairs
            for filename in [
                "train.parquet",
                "validation.parquet",
                "split_summary.json",
            ]
        ]
        + [
            f"{ISSUE_473_FIXED_ROOT}/full_scale/intersection/" f"{region}/{filename}"
            for region in ["cds", ENHANCER_REGION]
            for filename in [
                "full_window_validation.parquet",
                "center_1_validation.parquet",
                "selection.tsv",
                "summary.json",
            ]
        ]
    )


rule issue_473_fixed_projection_experiment:
    """Execute every data and QC goal preceding #473 model training."""
    input:
        issue_473_fixed_experiment_outputs,
