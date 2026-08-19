"""Exact fixed-anchor population for the issue #473 comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from marin_dna_vertebrate_projection.issue_473.pilot import (
    CONSERVATION_SCORE_COLUMN,
    build_scored_anchor_catalog,
)
from marin_dna_vertebrate_projection.issue_473.policy import ANCHOR_COLUMNS

STANDARD_REGIONS = (
    "cds",
    "utr3",
    "ncrna_exon",
    "tss_region_and_utr5",
)
ENHANCER_REGION = "ccre_enhancer_centered"
FIXED_REGIONS = (*STANDARD_REGIONS, ENHANCER_REGION)


def read_exp351_enhancer_anchors(
    noexon_bed_path: str | Path,
    scored_path: str | Path,
    *,
    target_length: int = 255,
    min_proportion_conserved: float = 0.20,
) -> pl.DataFrame:
    """Restore exp351's exact centered, conserved, exon-free anchor population."""
    anchors = pl.read_csv(
        noexon_bed_path,
        separator="\t",
        has_header=False,
        new_columns=["chrom", "start", "end", "query_name"],
        schema_overrides={
            "chrom": pl.String,
            "start": pl.Int64,
            "end": pl.Int64,
            "query_name": pl.String,
        },
    )
    scores = pl.read_parquet(scored_path).select(
        pl.col("name").alias("query_name"),
        pl.col(CONSERVATION_SCORE_COLUMN).cast(pl.Float64),
    )
    assert anchors.height > 0
    assert anchors["query_name"].n_unique() == anchors.height
    assert scores["query_name"].n_unique() == scores.height
    result = (
        anchors.join(scores, on="query_name", how="left", validate="1:1")
        .select(
            "query_name",
            (pl.lit("chr") + pl.col("chrom")).alias("source_chrom"),
            pl.col("start").alias("source_start"),
            pl.col("end").alias("source_end"),
            pl.lit(ENHANCER_REGION).alias("region_label"),
            CONSERVATION_SCORE_COLUMN,
        )
        .sort("source_chrom", "source_start", "query_name")
    )
    assert result[CONSERVATION_SCORE_COLUMN].null_count() == 0
    assert (result[CONSERVATION_SCORE_COLUMN] >= min_proportion_conserved).all()
    assert (result["source_start"] >= 0).all()
    assert (result["source_end"] - result["source_start"] == target_length).all()
    return result


def build_fixed_scored_anchor_catalog(
    issue417_labels_path: str | Path,
    issue417_scored_paths: list[str | Path],
    exp351_noexon_bed_path: str | Path,
    exp351_scored_path: str | Path,
    *,
    target_length: int = 255,
    min_proportion_conserved: float = 0.20,
    expected_enhancer_anchors: int | None = None,
) -> pl.DataFrame:
    """Combine four immutable #417 regions with the immutable exp351 sentinel."""
    assert issue417_scored_paths
    labels = pl.read_parquet(issue417_labels_path)
    scored = pl.concat(
        [pl.read_parquet(path) for path in issue417_scored_paths],
        how="vertical",
    )
    standard = (
        build_scored_anchor_catalog(
            labels,
            scored,
            min_proportion_conserved=min_proportion_conserved,
            target_length=target_length,
        )
        .filter(pl.col("region_label").is_in(STANDARD_REGIONS))
        .with_columns(pl.col(CONSERVATION_SCORE_COLUMN).cast(pl.Float64))
    )
    assert set(standard["region_label"].unique().to_list()) == set(STANDARD_REGIONS)

    enhancer = read_exp351_enhancer_anchors(
        exp351_noexon_bed_path,
        exp351_scored_path,
        target_length=target_length,
        min_proportion_conserved=min_proportion_conserved,
    )
    if expected_enhancer_anchors is not None:
        assert enhancer.height == expected_enhancer_anchors, (
            f"exp351 anchor count changed: {enhancer.height} "
            f"!= {expected_enhancer_anchors}"
        )
    assert set(standard["query_name"]).isdisjoint(enhancer["query_name"])

    fixed = (
        pl.concat([standard, enhancer], how="vertical")
        .select(*ANCHOR_COLUMNS, CONSERVATION_SCORE_COLUMN)
        .sort("region_label", "source_chrom", "source_start", "query_name")
    )
    assert fixed["query_name"].n_unique() == fixed.height
    assert set(fixed["region_label"].unique().to_list()) == set(FIXED_REGIONS)
    assert (fixed["source_end"] - fixed["source_start"] == target_length).all()
    return fixed


def write_fixed_scored_anchor_catalog(
    issue417_labels_path: str | Path,
    issue417_scored_paths: list[str | Path],
    exp351_noexon_bed_path: str | Path,
    exp351_scored_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path,
    *,
    target_length: int = 255,
    min_proportion_conserved: float = 0.20,
    expected_enhancer_anchors: int = 116_162,
) -> None:
    """Write the fixed catalog and a small region/count provenance receipt."""
    catalog = build_fixed_scored_anchor_catalog(
        issue417_labels_path,
        issue417_scored_paths,
        exp351_noexon_bed_path,
        exp351_scored_path,
        target_length=target_length,
        min_proportion_conserved=min_proportion_conserved,
        expected_enhancer_anchors=expected_enhancer_anchors,
    )
    output = Path(output_path)
    summary = Path(summary_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_parquet(output)
    counts = {
        str(region): int(count)
        for region, count in catalog.group_by("region_label").len().iter_rows()
    }
    summary.write_text(
        json.dumps(
            {
                "coordinate_system": "0-based half-open",
                "target_length": target_length,
                "min_proportion_conserved": min_proportion_conserved,
                "regions": counts,
                "total_anchors": catalog.height,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
