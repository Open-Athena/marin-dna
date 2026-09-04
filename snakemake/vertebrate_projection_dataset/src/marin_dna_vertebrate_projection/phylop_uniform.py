"""Strict phyloP-selector control for the issue #517 uniform-grid experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from marin_dna_vertebrate_projection.gpn_star_anchors import (
    GPN_ARMS,
    GPN_ASSIGNMENT_RECIPE,
    assign_gpn_six_arms,
)

PHYLOP_UNIFORM_ARMS = GPN_ARMS
PHYLOP_ASSIGNMENT_RECIPE = GPN_ASSIGNMENT_RECIPE
_NON_CCRE_FUNCTIONAL_FRACS = (
    "cds_frac",
    "utr3_frac",
    "tss_region_and_utr5_frac",
    "ncrna_exon_frac",
)


def _evenly_spaced_by_arm(frame: pl.DataFrame, per_arm: int) -> pl.DataFrame:
    assert per_arm > 0
    sampled: list[pl.DataFrame] = []
    for arm in sorted(frame["arm"].unique().to_list()):
        group = frame.filter(pl.col("arm") == arm).sort("chrom", "start", "name")
        assert group.height >= per_arm
        indices = np.linspace(0, group.height - 1, per_arm, dtype=int).tolist()
        sampled.append(
            group.with_row_index("_sample_row")
            .filter(pl.col("_sample_row").is_in(indices))
            .drop("_sample_row")
        )
    return pl.concat(sampled, how="vertical").sort("chrom", "start", "name")


def write_phylop_uniform_anchor_catalog(
    labels_path: str | Path,
    scored_paths: list[str | Path],
    catalog_path: str | Path,
    assignments_path: str | Path,
    summary_path: str | Path,
    *,
    phylop_track: str,
    phylop_threshold: float,
    min_proportion_conserved: float,
    expected_full_count: int | None = None,
    smoke_anchors_per_arm: int | None = None,
    allowed_chroms: list[str] | None = None,
) -> dict[str, object]:
    """Write the exhaustive six-arm catalog over phyloP-selected grid windows."""
    assert scored_paths
    assert phylop_track == "phyloP_447m"
    assert phylop_threshold == 2.2162
    assert min_proportion_conserved == 0.20

    labels = pl.read_parquet(labels_path).with_columns(
        pl.col("chrom").str.strip_prefix("chr")
    )
    selected = (
        pl.concat([pl.scan_parquet(path) for path in scored_paths], how="vertical")
        .filter(pl.col("proportion_conserved") >= min_proportion_conserved)
        .with_columns(pl.col("chrom").str.strip_prefix("chr"))
        .collect(engine="streaming")
    )
    score_columns = [
        "name",
        "chrom",
        "start",
        "end",
        "conserved_bases",
        "proportion_conserved",
        "mean_phylop",
        "n_valid_bases",
    ]
    joined = labels.join(
        selected.select(score_columns),
        on=["name", "chrom", "start", "end"],
        how="inner",
        validate="1:1",
    )
    assert joined.height == labels.height == selected.height
    if allowed_chroms is not None:
        normalized_chroms = [chrom.removeprefix("chr") for chrom in allowed_chroms]
        joined = joined.filter(pl.col("chrom").is_in(normalized_chroms))
        assert joined.height > 0
    joined = assign_gpn_six_arms(joined)
    pre_cap_count = joined.height
    if smoke_anchors_per_arm is not None:
        joined = _evenly_spaced_by_arm(joined, smoke_anchors_per_arm)
    elif expected_full_count is not None:
        assert joined.height == expected_full_count

    assert set(joined["arm"].unique().to_list()) == set(PHYLOP_UNIFORM_ARMS)
    catalog = (
        joined.rename(
            {
                "name": "query_name",
                "start": "source_start",
                "end": "source_end",
                "label": "v4_region_label",
                "arm": "region_label",
            }
        )
        .with_columns(
            (pl.lit("chr") + pl.col("chrom")).alias("source_chrom"),
            pl.lit(phylop_track).alias("phylop_track"),
            pl.lit(phylop_threshold).alias("phylop_threshold"),
            pl.lit(min_proportion_conserved).alias("minimum_proportion_conserved"),
        )
        .drop("chrom")
        .select(
            "query_name",
            "source_chrom",
            "source_start",
            "source_end",
            "region_label",
            pl.exclude(
                "query_name",
                "source_chrom",
                "source_start",
                "source_end",
                "region_label",
            ),
        )
        .sort("source_chrom", "source_start", "query_name")
    )
    assert catalog["query_name"].n_unique() == catalog.height
    assert (catalog["source_end"] - catalog["source_start"] == 255).all()
    assert (catalog["conserved_bases"] >= 51).all()
    assert (catalog["proportion_conserved"] >= min_proportion_conserved).all()

    catalog_output = Path(catalog_path)
    catalog_output.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_parquet(catalog_output)

    assignments = catalog.select(
        "query_name",
        "source_chrom",
        "source_start",
        "source_end",
        pl.col("region_label").alias("arm"),
        "assignment_recipe",
        "assignment_reason",
        "v4_region_label",
        "functional_frac",
        *_NON_CCRE_FUNCTIONAL_FRACS,
        "ccre_non_promoter_frac",
        "conserved_bases",
        "proportion_conserved",
        "mean_phylop",
        "n_valid_bases",
        "phylop_track",
        "phylop_threshold",
        "minimum_proportion_conserved",
    )
    assert assignments.height == catalog.height
    assert assignments.select("assignment_recipe", "query_name").is_unique().all()
    assignments_output = Path(assignments_path)
    assignments_output.parent.mkdir(parents=True, exist_ok=True)
    assignments.write_parquet(assignments_output)

    summary: dict[str, object] = {
        "assignment_arm_count_sum": catalog.height,
        "assignment_is_exhaustive": True,
        "assignment_recipe": PHYLOP_ASSIGNMENT_RECIPE,
        "assignment_universe": (
            "uniform GRCh38 255 bp / 128 bp-stride windows with at least "
            f"51 of 255 bases satisfying {phylop_track} >= {phylop_threshold}"
        ),
        "background_by_assignment_reason": (
            catalog.filter(pl.col("region_label") == "background")
            .group_by("assignment_reason")
            .len()
            .sort("assignment_reason")
            .to_dicts()
        ),
        "by_arm": (
            catalog.group_by("region_label").len().sort("region_label").to_dicts()
        ),
        "by_chrom": (
            catalog.group_by("source_chrom").len().sort("source_chrom").to_dicts()
        ),
        "catalog_rows": catalog.height,
        "allowed_chroms": allowed_chroms,
        "minimum_proportion_conserved": min_proportion_conserved,
        "phylop_threshold": phylop_threshold,
        "phylop_track": phylop_track,
        "pre_smoke_cap_rows": pre_cap_count,
        "smoke_anchors_per_arm": smoke_anchors_per_arm,
    }
    summary_output = Path(summary_path)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
