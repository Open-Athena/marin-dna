"""Additive sampled HAL alignment trace for issue #473."""

import os

from marin_dna_vertebrate_projection.issue_473.hal_trace import (
    TRACE_POLICIES,
    run_named_psl_trace,
    write_hal_trace_bed,
    write_hal_trace_metrics,
    write_hal_trace_sample,
    write_hal_trace_summary,
)

ISSUE_473_TRACE_ROOT = f"{ISSUE_473_FIXED_ROOT}/hal_trace"
ISSUE_473_TRACE_POLICY_RE = "|".join(TRACE_POLICIES)
ISSUE_473_TRACE_SOURCE_ROOT = os.environ.get("ISSUE_473_TRACE_SOURCE_ROOT")
if ISSUE_473_TRACE_SOURCE_ROOT:
    ISSUE_473_TRACE_FULL_SOURCE = (
        f"{ISSUE_473_TRACE_SOURCE_ROOT}/full_scale/full_window/"
        "sequences/all_sources.parquet"
    )
    ISSUE_473_TRACE_CENTER_SOURCE = (
        f"{ISSUE_473_TRACE_SOURCE_ROOT}/runs/full_center_1/"
        "sequences/all_sources.parquet"
    )
else:
    ISSUE_473_TRACE_FULL_SOURCE = ISSUE_473_FIXED_FULL_WINDOW
    ISSUE_473_TRACE_CENTER_SOURCE = ISSUE_473_FIXED_CENTER_1


rule issue_473_hal_trace_sample:
    input:
        full=local(ISSUE_473_TRACE_FULL_SOURCE),
        center=local(ISSUE_473_TRACE_CENTER_SOURCE),
    output:
        sample=f"{ISSUE_473_TRACE_ROOT}/sample.tsv",
        summary=f"{ISSUE_473_TRACE_ROOT}/sample_summary.json",
    resources:
        mem_mb=24000,
        final_large_scan=1,
    run:
        write_hal_trace_sample(
            input.full,
            input.center,
            output.sample,
            output.summary,
            seed=ISSUE_473_SAMPLE_SEED,
        )


rule issue_473_hal_trace_bed:
    input:
        f"{ISSUE_473_TRACE_ROOT}/sample.tsv",
    output:
        local(f"{ISSUE_473_TRACE_ROOT}/input/{{policy}}/{{species}}.bed"),
    wildcard_constraints:
        policy=ISSUE_473_TRACE_POLICY_RE,
        species=MAMMAL_RE,
    run:
        write_hal_trace_bed(
            input[0],
            output[0],
            policy=wildcards.policy,
            alignment_name=wildcards.species,
        )


rule issue_473_hal_trace_liftover:
    input:
        hal=local(HAL_PATH),
        bed=local(f"{ISSUE_473_TRACE_ROOT}/input/{{policy}}/{{species}}.bed"),
        validation=local(HAL_VALIDATION),
    output:
        f"{ISSUE_473_TRACE_ROOT}/raw/{{policy}}/{{species}}.psl",
    wildcard_constraints:
        policy=ISSUE_473_TRACE_POLICY_RE,
        species=MAMMAL_RE,
    threads: 1
    resources:
        mem_mb=2000,
    run:
        run_named_psl_trace(
            input.hal,
            wildcards.species,
            input.bed,
            output[0],
        )


rule issue_473_hal_trace_metrics:
    input:
        sample=f"{ISSUE_473_TRACE_ROOT}/sample.tsv",
        psl=f"{ISSUE_473_TRACE_ROOT}/raw/{{policy}}/{{species}}.psl",
    output:
        f"{ISSUE_473_TRACE_ROOT}/metrics/{{policy}}/{{species}}.parquet",
    wildcard_constraints:
        policy=ISSUE_473_TRACE_POLICY_RE,
        species=MAMMAL_RE,
    resources:
        mem_mb=1000,
    run:
        write_hal_trace_metrics(
            input.sample,
            input.psl,
            output[0],
            policy=wildcards.policy,
            alignment_name=wildcards.species,
        )


def issue_473_hal_trace_metric_inputs(_wildcards):
    return expand(
        f"{ISSUE_473_TRACE_ROOT}/metrics/{{policy}}/{{species}}.parquet",
        policy=TRACE_POLICIES,
        species=MAMMALS,
    )


rule issue_473_hal_trace_summary:
    input:
        issue_473_hal_trace_metric_inputs,
    output:
        metrics=f"{ISSUE_473_TRACE_ROOT}/all_metrics.parquet",
        summary=f"{ISSUE_473_TRACE_ROOT}/summary.parquet",
        uncertainty=f"{ISSUE_473_TRACE_ROOT}/anchor_uncertainty.parquet",
        report=f"{ISSUE_473_TRACE_ROOT}/report.md",
    resources:
        mem_mb=4000,
    run:
        write_hal_trace_summary(
            list(input),
            output.metrics,
            output.summary,
            output.uncertainty,
            output.report,
        )


rule issue_473_fixed_hal_alignment_trace:
    """Reproduce exact HAL coverage on a deterministic accepted-row sample."""
    input:
        PRODUCER_MANIFEST,
        f"{ISSUE_473_TRACE_ROOT}/sample.tsv",
        f"{ISSUE_473_TRACE_ROOT}/sample_summary.json",
        f"{ISSUE_473_TRACE_ROOT}/all_metrics.parquet",
        f"{ISSUE_473_TRACE_ROOT}/summary.parquet",
        f"{ISSUE_473_TRACE_ROOT}/anchor_uncertainty.parquet",
        f"{ISSUE_473_TRACE_ROOT}/report.md",
