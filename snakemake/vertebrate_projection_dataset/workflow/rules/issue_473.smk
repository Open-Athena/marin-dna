"""Additive center-seeded projection experiment for issue #473."""

from marin_dna_zoonomia_projection.projection.hal import (
    attach_src_size,
    parse_halliftover_bed,
    run_halliftover,
)
from marin_dna_zoonomia_projection.region_labels import REGION_LABELS
from marin_dna_vertebrate_projection.issue_473.comparison import (
    write_policy_comparison,
)
from marin_dna_vertebrate_projection.issue_473.pilot import (
    build_scored_anchor_catalog,
    sample_projection_pilot_anchors,
)
from marin_dna_vertebrate_projection.issue_473.policy import (
    FULL_WINDOW_POLICY,
    build_projection_requests,
    centered_landmark_policy,
    policy_by_name,
)
from marin_dna_vertebrate_projection.issue_473.projection import (
    write_contract_outputs,
    write_contract_outputs_for_alignment,
    write_hal_request_bed6,
    write_maf_request_candidates,
)
from marin_dna_vertebrate_projection.pipeline_io import (
    combine_sequence_parquets,
    merge_parquets_streaming,
    write_hal_fragments,
    write_qc_files,
)
from marin_dna_vertebrate_projection.sequence_compatibility import (
    validate_projected_twobit_sizes,
)

ISSUE_473_ROOT = f"{RESULTS}/experiments/473"
ISSUE_473_POLICIES = [
    FULL_WINDOW_POLICY.name,
    *[centered_landmark_policy(width).name for width in [1, 17, 33, 65, 129]],
]
ISSUE_473_POLICY_RE = "|".join(ISSUE_473_POLICIES)
ISSUE_473_MAX_ANCHORS_PER_REGION = 10_000
ISSUE_473_CONSERVATION_QUANTILES = 5
ISSUE_473_SAMPLE_SEED = 473
ISSUE_473_SCORED_ANCHORS = f"{ISSUE_473_ROOT}/pilot/scored_anchors.parquet"
ISSUE_473_SAMPLE_ANCHORS = f"{ISSUE_473_ROOT}/pilot/anchors.parquet"
ISSUE_473_PROJECTION_ROOT = f"{ISSUE_473_ROOT}/projection"
ISSUE_473_HUMAN_SEQUENCES = (
    f"{ISSUE_473_PROJECTION_ROOT}/sequences/human_reference.parquet"
)


def issue_473_request_anchor_input(_wildcards):
    if TIER == "smoke":
        return ANCHOR_CATALOG_INPUT
    return ISSUE_473_SAMPLE_ANCHORS


rule issue_473_scored_anchor_catalog:
    """Rejoin scores in a new artifact without changing the established catalog."""
    input:
        labels=f"{RESULTS}/anchors/labels.parquet",
        scored=expand(f"{RESULTS}/anchors/scored/{{chrom}}.parquet", chrom=FULL_CHROMS),
    output:
        ISSUE_473_SCORED_ANCHORS,
    resources:
        mem_mb=24000,
        final_large_scan=1,
    run:
        catalog = build_scored_anchor_catalog(
            pl.read_parquet(input.labels),
            pl.concat([pl.read_parquet(path) for path in input.scored]),
            min_proportion_conserved=MIN_PROPORTION,
            target_length=WINDOW_SIZE,
        )
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        catalog.write_parquet(output[0])


rule issue_473_pilot_anchor_sample:
    """Select up to 10,000 anchors per functional region for the wider pilot."""
    input:
        ISSUE_473_SCORED_ANCHORS,
    output:
        anchors=ISSUE_473_SAMPLE_ANCHORS,
        selection=f"{ISSUE_473_ROOT}/pilot/selection.tsv",
        strata=f"{ISSUE_473_ROOT}/pilot/stratum_counts.tsv",
    resources:
        mem_mb=24000,
        final_large_scan=1,
    run:
        sample = sample_projection_pilot_anchors(
            pl.read_parquet(input[0]),
            regions=tuple(REGION_LABELS),
            max_per_region=ISSUE_473_MAX_ANCHORS_PER_REGION,
            conservation_quantiles=ISSUE_473_CONSERVATION_QUANTILES,
            seed=ISSUE_473_SAMPLE_SEED,
        )
        Path(output.anchors).parent.mkdir(parents=True, exist_ok=True)
        sample.anchors.write_parquet(output.anchors)
        sample.selection_manifest.write_csv(output.selection, separator="\t")
        sample.stratum_counts.write_csv(output.strata, separator="\t")


rule issue_473_policy_manifest:
    output:
        f"{ISSUE_473_ROOT}/policies.tsv",
    run:
        policies = [policy_by_name(name) for name in ISSUE_473_POLICIES]
        frame = pl.DataFrame(
            [
                {
                    "projection_policy": policy.name,
                    "landmark_width": policy.landmark_width,
                    "pre_resize_min_length": policy.pre_resize_min_length,
                    "pre_resize_max_length": policy.pre_resize_max_length,
                }
                for policy in policies
            ]
        )
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        frame.write_csv(output[0], separator="\t")


rule issue_473_projection_requests:
    input:
        issue_473_request_anchor_input,
    output:
        f"{ISSUE_473_ROOT}/requests/{{projection_policy}}.parquet",
    wildcard_constraints:
        projection_policy=ISSUE_473_POLICY_RE,
    run:
        anchor_path = Path(input[0])
        anchors = (
            pl.read_parquet(anchor_path)
            if anchor_path.suffix == ".parquet"
            else pl.read_csv(anchor_path, separator="\t")
        )
        requests = build_projection_requests(
            anchors, policy_by_name(wildcards.projection_policy)
        )
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        requests.write_parquet(output[0])


rule issue_473_hal_request_bed:
    input:
        f"{ISSUE_473_ROOT}/requests/{{projection_policy}}.parquet",
    output:
        local(f"{ISSUE_473_ROOT}/hal/input/{{projection_policy}}.bed"),
    wildcard_constraints:
        projection_policy=ISSUE_473_POLICY_RE,
    run:
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        write_hal_request_bed6(input[0], output[0])


rule issue_473_request_artifacts:
    """Build reviewable request artifacts without launching a projection."""
    input:
        f"{ISSUE_473_ROOT}/policies.tsv",
        expand(
            f"{ISSUE_473_ROOT}/requests/{{projection_policy}}.parquet",
            projection_policy=ISSUE_473_POLICIES,
        ),
        local(
            expand(
                f"{ISSUE_473_ROOT}/hal/input/{{projection_policy}}.bed",
                projection_policy=ISSUE_473_POLICIES,
            )
        ),


def issue_473_qc_rejections(wildcards):
    policy_root = f"{ISSUE_473_PROJECTION_ROOT}/{wildcards.projection_policy}"
    return (
        expand(
            f"{policy_root}/hal/rejected/{{species}}.parquet",
            species=MAMMALS,
        )
        + expand(
            f"{policy_root}/hal/sequence_rejected/{{species}}.parquet",
            species=MAMMALS,
        )
        + expand(
            f"{policy_root}/multiz/rejected/{{species}}.parquet",
            species=NON_MAMMALS,
        )
        + expand(
            f"{policy_root}/multiz/sequence_rejected/{{species}}.parquet",
            species=NON_MAMMALS,
        )
    )


def issue_473_smoke_projection_outputs(_wildcards):
    assert TIER == "smoke", (
        "issue_473_projection_smoke requires the smoke tier; "
        "do not launch the full projection through this target"
    )
    return (
        [
            PRODUCER_MANIFEST,
            f"{ISSUE_473_ROOT}/policies.tsv",
            f"{ISSUE_473_PROJECTION_ROOT}/qc/policy_summary.parquet",
            f"{ISSUE_473_PROJECTION_ROOT}/qc/full_window_pairwise.parquet",
        ]
        + [
            f"{ISSUE_473_PROJECTION_ROOT}/{policy}/sequences/all_sources.parquet"
            for policy in ISSUE_473_POLICIES
        ]
        + [
            f"{ISSUE_473_PROJECTION_ROOT}/{policy}/qc/aggregates.parquet"
            for policy in ISSUE_473_POLICIES
        ]
    )


rule issue_473_human_reference_sequences:
    input:
        anchors=issue_473_request_anchor_input,
        twobit=f"{RESULTS}/reference/hg38.2bit",
        sizes=f"{RESULTS}/reference/hg38.chrom.sizes",
    output:
        ISSUE_473_HUMAN_SEQUENCES,
    conda:
        "../envs/bioinformatics.yaml"
    resources:
        mem_mb=4000,
    shell:
        "uv run python -m "
        "marin_dna_vertebrate_projection.sequence_cli "
        "human {input.anchors} {input.twobit} {input.sizes} {output}"


rule issue_473_hal_liftover:
    input:
        hal=local(HAL_PATH),
        bed=local(f"{ISSUE_473_ROOT}/hal/input/{{projection_policy}}.bed"),
        validation=local(HAL_VALIDATION),
    output:
        f"{ISSUE_473_PROJECTION_ROOT}/{{projection_policy}}/hal/raw/{{species}}.bed",
    wildcard_constraints:
        projection_policy=ISSUE_473_POLICY_RE,
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


rule issue_473_hal_fragments:
    input:
        raw=(
            f"{ISSUE_473_PROJECTION_ROOT}/{{projection_policy}}/"
            "hal/raw/{species}.bed"
        ),
        sizes=f"{RESULTS}/hal/chrom_sizes/{{species}}.tsv",
        requests=(f"{ISSUE_473_ROOT}/requests/{{projection_policy}}.parquet"),
        manifest=ACTIVE_MANIFEST,
    output:
        (
            f"{ISSUE_473_PROJECTION_ROOT}/{{projection_policy}}/"
            "hal/fragments/{species}.parquet"
        ),
    wildcard_constraints:
        projection_policy=ISSUE_473_POLICY_RE,
        species=MAMMAL_RE,
    resources:
        mem_mb=8000,
    run:
        records = attach_src_size(
            parse_halliftover_bed(input.raw, wildcards.species), input.sizes
        )
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        write_hal_fragments(records, input.requests, input.manifest, output[0])


rule issue_473_hal_contract:
    input:
        (
            f"{ISSUE_473_PROJECTION_ROOT}/{{projection_policy}}/"
            "hal/fragments/{species}.parquet"
        ),
    output:
        accepted=(
            f"{ISSUE_473_PROJECTION_ROOT}/{{projection_policy}}/"
            "hal/accepted/{species}.parquet"
        ),
        rejected=(
            f"{ISSUE_473_PROJECTION_ROOT}/{{projection_policy}}/"
            "hal/rejected/{species}.parquet"
        ),
    wildcard_constraints:
        projection_policy=ISSUE_473_POLICY_RE,
        species=MAMMAL_RE,
    resources:
        mem_mb=10000,
    run:
        policy = policy_by_name(wildcards.projection_policy)
        write_contract_outputs(
            input[0],
            output.accepted,
            output.rejected,
            target_length=TARGET_LENGTH,
            pre_resize_min_length=policy.pre_resize_min_length,
            pre_resize_max_length=policy.pre_resize_max_length,
        )


rule issue_473_hal_sequences:
    input:
        accepted=(
            f"{ISSUE_473_PROJECTION_ROOT}/{{projection_policy}}/"
            "hal/accepted/{species}.parquet"
        ),
        twobit=f"{RESULTS}/hal/genomes/{{species}}.2bit",
    output:
        sequences=(
            f"{ISSUE_473_PROJECTION_ROOT}/{{projection_policy}}/"
            "sequences/hal/{species}.parquet"
        ),
        rejected=(
            f"{ISSUE_473_PROJECTION_ROOT}/{{projection_policy}}/"
            "hal/sequence_rejected/{species}.parquet"
        ),
    wildcard_constraints:
        projection_policy=ISSUE_473_POLICY_RE,
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


rule issue_473_multiz_candidates:
    input:
        maf=local(f"{MULTIZ_STAGE_DIR}/maf/{{chrom}}.maf.gz"),
        requests=(f"{ISSUE_473_ROOT}/requests/{{projection_policy}}.parquet"),
        manifest=ACTIVE_MANIFEST,
    output:
        (
            f"{ISSUE_473_PROJECTION_ROOT}/{{projection_policy}}/"
            "multiz/fragments/{chrom}.parquet"
        ),
    wildcard_constraints:
        projection_policy=ISSUE_473_POLICY_RE,
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


rule issue_473_multiz_contract:
    input:
        (
            f"{ISSUE_473_PROJECTION_ROOT}/{{projection_policy}}/"
            "multiz/fragments/{chrom}.parquet"
        ),
    output:
        accepted=(
            f"{ISSUE_473_PROJECTION_ROOT}/{{projection_policy}}/"
            "multiz/accepted/by_chrom/{chrom}/{species}.parquet"
        ),
        rejected=(
            f"{ISSUE_473_PROJECTION_ROOT}/{{projection_policy}}/"
            "multiz/rejected/by_chrom/{chrom}/{species}.parquet"
        ),
    wildcard_constraints:
        projection_policy=ISSUE_473_POLICY_RE,
        chrom=CHROM_RE,
        species=NON_MAMMAL_RE,
    resources:
        mem_mb=8000,
    run:
        policy = policy_by_name(wildcards.projection_policy)
        write_contract_outputs_for_alignment(
            input[0],
            wildcards.species,
            output.accepted,
            output.rejected,
            target_length=TARGET_LENGTH,
            pre_resize_min_length=policy.pre_resize_min_length,
            pre_resize_max_length=policy.pre_resize_max_length,
        )


rule issue_473_merge_multiz_accepted:
    input:
        lambda wc: expand(
            (
                f"{ISSUE_473_PROJECTION_ROOT}/{{projection_policy}}/"
                "multiz/accepted/by_chrom/{chrom}/{species}.parquet"
            ),
            projection_policy=[wc.projection_policy],
            chrom=CHROMS,
            species=[wc.species],
        ),
    output:
        (
            f"{ISSUE_473_PROJECTION_ROOT}/{{projection_policy}}/"
            "multiz/accepted/{species}.parquet"
        ),
    wildcard_constraints:
        projection_policy=ISSUE_473_POLICY_RE,
        species=NON_MAMMAL_RE,
    resources:
        mem_mb=4000,
    run:
        merge_parquets_streaming(list(input), output[0])


rule issue_473_merge_multiz_rejected:
    input:
        lambda wc: expand(
            (
                f"{ISSUE_473_PROJECTION_ROOT}/{{projection_policy}}/"
                "multiz/rejected/by_chrom/{chrom}/{species}.parquet"
            ),
            projection_policy=[wc.projection_policy],
            chrom=CHROMS,
            species=[wc.species],
        ),
    output:
        (
            f"{ISSUE_473_PROJECTION_ROOT}/{{projection_policy}}/"
            "multiz/rejected/{species}.parquet"
        ),
    wildcard_constraints:
        projection_policy=ISSUE_473_POLICY_RE,
        species=NON_MAMMAL_RE,
    resources:
        mem_mb=4000,
    run:
        merge_parquets_streaming(list(input), output[0])


rule issue_473_validate_multiz_twobit:
    input:
        accepted=(
            f"{ISSUE_473_PROJECTION_ROOT}/{{projection_policy}}/"
            "multiz/accepted/{species}.parquet"
        ),
        sizes=f"{RESULTS}/multiz/genomes/{{species}}.chrom.sizes",
    output:
        (
            f"{ISSUE_473_PROJECTION_ROOT}/{{projection_policy}}/"
            "multiz/genomes/{species}.compatibility.json"
        ),
    wildcard_constraints:
        projection_policy=ISSUE_473_POLICY_RE,
        species=NON_MAMMAL_RE,
    run:
        validate_projected_twobit_sizes(input.accepted, input.sizes, output[0])


rule issue_473_multiz_sequences:
    input:
        accepted=(
            f"{ISSUE_473_PROJECTION_ROOT}/{{projection_policy}}/"
            "multiz/accepted/{species}.parquet"
        ),
        twobit=f"{RESULTS}/multiz/genomes/{{species}}.2bit",
        compatibility=(
            f"{ISSUE_473_PROJECTION_ROOT}/{{projection_policy}}/"
            "multiz/genomes/{species}.compatibility.json"
        ),
    output:
        sequences=(
            f"{ISSUE_473_PROJECTION_ROOT}/{{projection_policy}}/"
            "sequences/multiz/{species}.parquet"
        ),
        rejected=(
            f"{ISSUE_473_PROJECTION_ROOT}/{{projection_policy}}/"
            "multiz/sequence_rejected/{species}.parquet"
        ),
    wildcard_constraints:
        projection_policy=ISSUE_473_POLICY_RE,
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


rule issue_473_combine_sequences:
    input:
        lambda wc: [ISSUE_473_HUMAN_SEQUENCES]
        + expand(
            (
                f"{ISSUE_473_PROJECTION_ROOT}/{{projection_policy}}/"
                "sequences/hal/{species}.parquet"
            ),
            projection_policy=[wc.projection_policy],
            species=MAMMALS,
        )
        + expand(
            (
                f"{ISSUE_473_PROJECTION_ROOT}/{{projection_policy}}/"
                "sequences/multiz/{species}.parquet"
            ),
            projection_policy=[wc.projection_policy],
            species=NON_MAMMALS,
        ),
    output:
        (
            f"{ISSUE_473_PROJECTION_ROOT}/{{projection_policy}}/"
            "sequences/all_sources.parquet"
        ),
    wildcard_constraints:
        projection_policy=ISSUE_473_POLICY_RE,
    resources:
        mem_mb=12000,
    run:
        combine_sequence_parquets(list(input), output[0])


rule issue_473_projection_qc:
    input:
        requests=(f"{ISSUE_473_ROOT}/requests/{{projection_policy}}.parquet"),
        accepted=(
            f"{ISSUE_473_PROJECTION_ROOT}/{{projection_policy}}/"
            "sequences/all_sources.parquet"
        ),
        manifest=ACTIVE_MANIFEST,
        rejected=issue_473_qc_rejections,
    output:
        per_anchor=(
            f"{ISSUE_473_PROJECTION_ROOT}/{{projection_policy}}/"
            "qc/per_anchor.parquet"
        ),
        per_scope=(
            f"{ISSUE_473_PROJECTION_ROOT}/{{projection_policy}}/"
            "qc/per_anchor_scope.parquet"
        ),
        rejections=(
            f"{ISSUE_473_PROJECTION_ROOT}/{{projection_policy}}/"
            "qc/rejection_counts.parquet"
        ),
        aggregates=(
            f"{ISSUE_473_PROJECTION_ROOT}/{{projection_policy}}/"
            "qc/aggregates.parquet"
        ),
    wildcard_constraints:
        projection_policy=ISSUE_473_POLICY_RE,
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


rule issue_473_compare_policies:
    input:
        sequences=expand(
            (
                f"{ISSUE_473_PROJECTION_ROOT}/{{projection_policy}}/"
                "sequences/all_sources.parquet"
            ),
            projection_policy=ISSUE_473_POLICIES,
        ),
        per_anchor=expand(
            (
                f"{ISSUE_473_PROJECTION_ROOT}/{{projection_policy}}/"
                "qc/per_anchor.parquet"
            ),
            projection_policy=ISSUE_473_POLICIES,
        ),
    output:
        summary=f"{ISSUE_473_PROJECTION_ROOT}/qc/policy_summary.parquet",
        pairwise=(f"{ISSUE_473_PROJECTION_ROOT}/qc/" "full_window_pairwise.parquet"),
    resources:
        mem_mb=12000,
        final_large_scan=1,
    run:
        write_policy_comparison(
            dict(zip(ISSUE_473_POLICIES, input.sequences, strict=True)),
            dict(zip(ISSUE_473_POLICIES, input.per_anchor, strict=True)),
            output.summary,
            output.pairwise,
            baseline_policy=FULL_WINDOW_POLICY.name,
            target_length=TARGET_LENGTH,
        )


rule issue_473_projection_smoke:
    """Run every issue #473 policy only on the configured smoke cohort."""
    input:
        issue_473_smoke_projection_outputs,
