"""Exhaustive six-arm assignment over the phyloP-selected uniform grid."""

from marin_dna_vertebrate_projection.gpn_star_anchors import (
    GPN_ARMS,
    GPN_ASSIGNMENT_RECIPE,
)
from marin_dna_vertebrate_projection.phylop_uniform import (
    write_phylop_uniform_anchor_catalog,
)

assert str(config["assignment_recipe"]) == GPN_ASSIGNMENT_RECIPE
assert tuple(config["assignment_arms"]) == GPN_ARMS
assert str(config["phylop_track"]) == "phyloP_447m"


rule phylop_uniform_anchor_catalog:
    input:
        labels=f"{RESULTS}/anchors/labels.parquet",
        scored=expand(
            f"{RESULTS}/anchors/scored/{{chrom}}.parquet",
            chrom=FULL_CHROMS,
        ),
    output:
        catalog=ANCHOR_CATALOG,
        assignments=PHYLOP_ASSIGNMENTS,
        summary=PHYLOP_ASSIGNMENT_SUMMARY,
    resources:
        mem_mb=24000,
        final_large_scan=1,
    run:
        write_phylop_uniform_anchor_catalog(
            input.labels,
            list(input.scored),
            output.catalog,
            output.assignments,
            output.summary,
            phylop_track=str(config["phylop_track"]),
            phylop_threshold=float(config["phyloP_447m_threshold"]),
            min_proportion_conserved=float(config["min_proportion_conserved"]),
            expected_full_count=(
                int(config["expected_full_anchors"]) if TIER == "full" else None
            ),
            smoke_anchors_per_arm=(
                int(config["smoke_anchors_per_arm"]) if TIER == "smoke" else None
            ),
            allowed_chroms=CHROMS if TIER == "smoke" else None,
        )
