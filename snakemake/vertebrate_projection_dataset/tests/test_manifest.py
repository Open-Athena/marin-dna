from __future__ import annotations

import polars as pl
import pytest
from marin_dna_vertebrate_projection.manifest import (
    select_family_representatives,
    validate_species_manifest,
)

from .helpers import species_manifest


def test_select_family_representatives_records_winner_and_exclusion() -> None:
    manifest = species_manifest()
    phasianidae = manifest.filter(pl.col("family") == "Phasianidae")
    assert phasianidae.filter(pl.col("selected"))["alignment_name"].to_list() == [
        "galGal4"
    ]
    assert phasianidae.filter(~pl.col("selected"))["selection_reason"].to_list() == [
        "excluded_lower_ranked_than:galGal4"
    ]
    validate_species_manifest(manifest)


def test_selection_is_independent_of_input_order() -> None:
    manifest = species_manifest()
    candidates = manifest.drop("selected", "selection_reason")
    shuffled = candidates.reverse()
    assert select_family_representatives(candidates).equals(
        select_family_representatives(shuffled)
    )


def test_manifest_rejects_mammal_owned_by_multiz() -> None:
    manifest = species_manifest().with_columns(
        pl.when(pl.col("alignment_name") == "galGal4")
        .then(pl.lit("mammals"))
        .otherwise(pl.col("clade"))
        .alias("clade")
    )
    with pytest.raises(AssertionError, match="exclude every mammal"):
        validate_species_manifest(manifest)


def test_manifest_rejects_duplicate_selected_taxonomy_id() -> None:
    manifest = species_manifest().with_columns(
        pl.when(pl.col("alignment_name") == "xenTro7")
        .then(pl.lit(9031))
        .otherwise(pl.col("taxonomy_id"))
        .alias("taxonomy_id")
    )
    with pytest.raises(AssertionError, match="duplicate taxonomy"):
        validate_species_manifest(manifest)
