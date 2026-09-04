"""Pinned one-per-order target selection for the issue #517 control."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from marin_dna_vertebrate_projection.manifest import (
    REQUIRED_MANIFEST_COLUMNS,
    validate_species_manifest,
)
from marin_dna_vertebrate_projection.projection.taxonomy import (
    ASSEMBLY_LEVEL_RANK,
    QUALITY_SOURCE_RANK,
)

FORCE_INCLUDE = frozenset({"Bos_taurus", "Mus_musculus"})
REFERENCE_ORDERS = frozenset({"Primates"})
ORDER_MANIFEST_COLUMNS = (
    "alignment_name",
    "scientific_name",
    "assembly",
    "taxonomy_id",
    "family",
    "order",
    "clade",
    "phylogenetic_rank",
    "backend",
    "selection_priority",
    "assembly_level",
    "contig_n50",
    "quality_source",
)


def _ranking_key(row: dict[str, object]) -> tuple[int, int, int, str]:
    """Return the historical issue #255 lower-is-better ranking key."""
    return (
        -QUALITY_SOURCE_RANK.get(str(row["quality_source"]), 0),
        -ASSEMBLY_LEVEL_RANK.get(str(row["assembly_level"]), 0),
        -int(row["contig_n50"]),
        str(row["alignment_name"]),
    )


def select_order_representatives(candidates: pl.DataFrame) -> pl.DataFrame:
    """Select one target per NCBI order from family-deduplicated candidates.

    This reproduces the historical issue #255 order policy: known HAL assembly
    provenance outranks assembly level, which outranks contig N50, followed by
    alignment name as a deterministic tiebreak. Bos taurus and Mus musculus are
    retained as the established representatives of Artiodactyla and Rodentia.
    Human is emitted outside the projection target manifest and occupies the
    dataset's sole Primates slot, so every non-human primate is excluded here.
    """
    required = set(ORDER_MANIFEST_COLUMNS)
    missing = required - set(candidates.columns)
    assert not missing, f"order candidates missing columns: {sorted(missing)}"
    assert candidates.height > 0
    assert candidates["alignment_name"].n_unique() == candidates.height
    assert (candidates["order"].str.len_chars() > 0).all()

    eligible = candidates.filter(~pl.col("order").is_in(REFERENCE_ORDERS))
    winners: list[dict[str, object]] = []
    for order, group in eligible.group_by("order", maintain_order=False):
        order_name = str(order[0])
        rows = group.to_dicts()
        forced = [row for row in rows if row["alignment_name"] in FORCE_INCLUDE]
        winner = min(forced or rows, key=_ranking_key)
        assert winner["order"] == order_name
        winners.append(winner)
    return pl.DataFrame(winners).select(*ORDER_MANIFEST_COLUMNS).sort(
        "phylogenetic_rank", "order", "alignment_name"
    )


def validate_order_manifest(
    manifest: pl.DataFrame,
    source_manifest: pl.DataFrame,
) -> None:
    """Validate a committed order manifest against the family source cohort."""
    validate_species_manifest(source_manifest)
    missing = set(ORDER_MANIFEST_COLUMNS) - set(manifest.columns)
    assert not missing, f"order manifest missing columns: {sorted(missing)}"
    assert manifest.columns == list(ORDER_MANIFEST_COLUMNS)
    assert manifest.height > 0
    assert sum(manifest.null_count().row(0)) == 0
    assert manifest["alignment_name"].n_unique() == manifest.height
    assert manifest["taxonomy_id"].n_unique() == manifest.height
    assert manifest["assembly"].n_unique() == manifest.height
    assert manifest["order"].n_unique() == manifest.height
    assert set(manifest["order"].to_list()).isdisjoint(REFERENCE_ORDERS)
    assert set(manifest["quality_source"].to_list()) <= set(QUALITY_SOURCE_RANK)
    assert set(manifest["backend"].to_list()) == {
        "zoonomia_cactus",
        "ucsc_multiz100way",
    }

    source_selected = source_manifest.filter(pl.col("selected"))
    source_columns = sorted(
        REQUIRED_MANIFEST_COLUMNS - {"selected", "selection_reason"}
    )
    expected_metadata = source_selected.select(*source_columns)
    observed_metadata = manifest.select(*source_columns)
    matches = observed_metadata.join(
        expected_metadata,
        on=source_columns,
        how="inner",
        validate="1:1",
    )
    assert matches.height == manifest.height, (
        "order representatives must be exact rows from species_selected.tsv"
    )


def read_order_manifest(
    path: str | Path,
    source_path: str | Path,
) -> pl.DataFrame:
    """Read and validate the pinned order representatives."""
    manifest = pl.read_csv(path, separator="\t")
    source = pl.read_csv(source_path, separator="\t")
    validate_order_manifest(manifest, source)
    return manifest
