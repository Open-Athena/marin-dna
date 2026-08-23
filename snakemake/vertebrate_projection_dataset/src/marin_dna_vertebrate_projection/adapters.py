"""Backend adapters into the shared vertebrate projection fragment schema."""

from __future__ import annotations

import polars as pl

from marin_dna_vertebrate_projection.maf import FRAGMENT_SCHEMA
from marin_dna_vertebrate_projection.manifest import (
    validate_species_manifest,
)


def hal_records_to_fragments(
    hal_records: pl.DataFrame,
    anchors: pl.DataFrame,
    species_manifest: pl.DataFrame,
) -> pl.DataFrame:
    """Convert parsed halLiftover BED rows into common candidate fragments.

    ``halLiftover --noDupes`` does not report the precise source sub-interval
    corresponding to each split BED row.  Those two fragment fields therefore
    remain null for HAL only; every other acceptance check is shared with MAF.
    """
    required_hal = {
        "query_name",
        "species",
        "t_chrom",
        "t_start",
        "t_end",
        "t_strand",
        "t_src_size",
    }
    missing_hal = required_hal - set(hal_records.columns)
    assert not missing_hal, f"HAL records missing columns: {sorted(missing_hal)}"
    required_anchors = {
        "query_name",
        "source_chrom",
        "source_start",
        "source_end",
        "region_label",
    }
    missing_anchors = required_anchors - set(anchors.columns)
    assert not missing_anchors, f"anchors missing columns: {sorted(missing_anchors)}"
    assert anchors["query_name"].n_unique() == anchors.height

    validate_species_manifest(species_manifest)
    metadata = species_manifest.filter(
        (pl.col("backend") == "zoonomia_cactus") & pl.col("selected")
    ).select(
        "alignment_name",
        "scientific_name",
        "assembly",
        "taxonomy_id",
        "family",
        "clade",
        "phylogenetic_rank",
    )
    assert metadata.height > 0

    joined = (
        hal_records.rename({"species": "alignment_name"})
        .join(anchors.select(sorted(required_anchors)), on="query_name", how="inner")
        .join(metadata, on="alignment_name", how="inner")
        .with_row_index("fragment_number")
    )
    assert joined.height == hal_records.height, (
        "every HAL row must match exactly one anchor and selected mammal"
    )
    result = joined.select(
        "query_name",
        "source_chrom",
        "source_start",
        "source_end",
        "region_label",
        pl.lit(None, dtype=pl.Int64).alias("source_fragment_start"),
        pl.lit(None, dtype=pl.Int64).alias("source_fragment_end"),
        pl.col("scientific_name").alias("species"),
        "alignment_name",
        "assembly",
        "taxonomy_id",
        "family",
        "clade",
        "phylogenetic_rank",
        pl.lit("zoonomia_cactus").alias("alignment_source"),
        "t_chrom",
        "t_start",
        "t_end",
        "t_strand",
        "t_src_size",
        pl.concat_str(
            pl.lit("hal:"), pl.col("query_name"), pl.lit(":"), pl.col("alignment_name")
        ).alias("mapping_id"),
        pl.concat_str(
            pl.lit("hal:"),
            pl.col("query_name"),
            pl.lit(":"),
            pl.col("alignment_name"),
            pl.lit(":"),
            pl.col("fragment_number"),
        ).alias("fragment_id"),
        (pl.col("t_end") - pl.col("t_start")).alias("aligned_bases"),
    )
    return result.cast(FRAGMENT_SCHEMA).sort("query_name", "species", "fragment_id")
