"""Standalone additive one-pass prefill for the two required #473 policies."""


configfile: "config/config.yaml"


include: "rules/common.smk"
include: "rules/staging.smk"


from marin_dna_vertebrate_projection.issue_473.batched_core import (
    CORE_RUNS,
    run_batched_hal_liftover,
    write_batched_hal_request_bed6,
    write_batched_maf_request_candidates,
    write_batched_prefill_manifest,
)

ISSUE_473_BATCHED_FIXED_ROOT = f"{RESULTS}/experiments/473/fixed"
ISSUE_473_BATCHED_RUN_ROOT = f"{ISSUE_473_BATCHED_FIXED_ROOT}/runs"
ISSUE_473_BATCHED_ROOT = f"{ISSUE_473_BATCHED_FIXED_ROOT}/batched_core"
ISSUE_473_BATCHED_REQUESTS = {
    run: f"{ISSUE_473_BATCHED_RUN_ROOT}/{run}/requests.parquet" for run in CORE_RUNS
}


rule issue_473_batched_core_request_bed:
    input:
        center=ISSUE_473_BATCHED_REQUESTS["full_center_1"],
        enhancer=ISSUE_473_BATCHED_REQUESTS["full_enhancer_full_window"],
    output:
        bed=local(f"{ISSUE_473_BATCHED_ROOT}/hal/input.bed"),
        manifest=f"{ISSUE_473_BATCHED_ROOT}/requests.json",
    resources:
        mem_mb=8000,
    run:
        write_batched_hal_request_bed6(
            {
                "full_center_1": input.center,
                "full_enhancer_full_window": input.enhancer,
            },
            output.bed,
            output.manifest,
        )


rule issue_473_batched_core_hal_liftover:
    input:
        hal=local(HAL_PATH),
        validation=local(HAL_VALIDATION),
        bed=local(f"{ISSUE_473_BATCHED_ROOT}/hal/input.bed"),
        requests=f"{ISSUE_473_BATCHED_ROOT}/requests.json",
    output:
        center=(f"{ISSUE_473_BATCHED_RUN_ROOT}/full_center_1/hal/raw/" "{species}.bed"),
        enhancer=(
            f"{ISSUE_473_BATCHED_RUN_ROOT}/full_enhancer_full_window/hal/raw/"
            "{species}.bed"
        ),
        receipt=f"{ISSUE_473_BATCHED_ROOT}/receipts/hal/{{species}}.json",
    wildcard_constraints:
        species=MAMMAL_RE,
    threads: 1
    resources:
        mem_mb=2000,
    run:
        run_batched_hal_liftover(
            input.hal,
            "Homo_sapiens",
            input.bed,
            wildcards.species,
            {
                "full_center_1": output.center,
                "full_enhancer_full_window": output.enhancer,
            },
            output.receipt,
        )


rule issue_473_batched_core_multiz_candidates:
    input:
        maf=local(f"{MULTIZ_STAGE_DIR}/maf/{{chrom}}.maf.gz"),
        center=ISSUE_473_BATCHED_REQUESTS["full_center_1"],
        enhancer=ISSUE_473_BATCHED_REQUESTS["full_enhancer_full_window"],
        manifest=ACTIVE_MANIFEST,
    output:
        center=(
            f"{ISSUE_473_BATCHED_RUN_ROOT}/full_center_1/multiz/fragments/"
            "{chrom}.parquet"
        ),
        enhancer=(
            f"{ISSUE_473_BATCHED_RUN_ROOT}/full_enhancer_full_window/"
            "multiz/fragments/{chrom}.parquet"
        ),
        receipt=f"{ISSUE_473_BATCHED_ROOT}/receipts/multiz/{{chrom}}.json",
    wildcard_constraints:
        chrom=CHROM_RE,
    threads: 1
    resources:
        mem_mb=8000,
    run:
        write_batched_maf_request_candidates(
            input.maf,
            {
                "full_center_1": input.center,
                "full_enhancer_full_window": input.enhancer,
            },
            input.manifest,
            {
                "full_center_1": output.center,
                "full_enhancer_full_window": output.enhancer,
            },
            output.receipt,
        )


def issue_473_batched_core_receipts(_wildcards):
    return (
        [f"{ISSUE_473_BATCHED_ROOT}/requests.json"]
        + expand(
            f"{ISSUE_473_BATCHED_ROOT}/receipts/hal/{{species}}.json",
            species=MAMMALS,
        )
        + expand(
            f"{ISSUE_473_BATCHED_ROOT}/receipts/multiz/{{chrom}}.json",
            chrom=CHROMS,
        )
    )


rule issue_473_batched_core_prefill:
    input:
        issue_473_batched_core_receipts,
    output:
        f"{ISSUE_473_BATCHED_ROOT}/manifest.json",
    resources:
        mem_mb=1000,
    run:
        write_batched_prefill_manifest(
            list(input)[1:],
            output[0],
            expected_species=MAMMALS,
            expected_chroms=CHROMS,
        )
