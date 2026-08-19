"""Additive center-seeded projection experiment for issue #473."""

from marin_dna_zoonomia_projection.region_labels import REGION_LABELS
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
    write_hal_request_bed6,
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
