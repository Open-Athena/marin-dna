from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from marin_dna_vertebrate_projection.order_manifest import (
    ORDER_MANIFEST_COLUMNS,
    read_order_manifest,
    select_order_representatives,
    validate_order_manifest,
)

from .helpers import species_manifest


def _order_candidates() -> pl.DataFrame:
    return species_manifest().filter(pl.col("selected")).with_columns(
        pl.when(pl.col("alignment_name").is_in(["Mus_musculus"]))
        .then(pl.lit("Rodentia"))
        .when(pl.col("alignment_name").is_in(["Felis_catus"]))
        .then(pl.lit("Carnivora"))
        .when(pl.col("alignment_name").is_in(["galGal4"]))
        .then(pl.lit("Galliformes"))
        .otherwise(pl.lit("Anura"))
        .alias("order"),
        pl.when(pl.col("backend") == "zoonomia_cactus")
        .then(pl.lit("zoonomia_supp_st2"))
        .otherwise(pl.lit("unknown"))
        .alias("quality_source"),
    ).drop("selected", "selection_reason")


def test_select_order_representatives_picks_one_per_order() -> None:
    candidates = _order_candidates()
    selected = select_order_representatives(candidates)
    assert selected.columns == list(ORDER_MANIFEST_COLUMNS)
    assert selected.height == 4
    assert selected["order"].n_unique() == selected.height


def test_order_selection_is_independent_of_input_order() -> None:
    candidates = _order_candidates()
    assert select_order_representatives(candidates).equals(
        select_order_representatives(candidates.reverse())
    )


def test_human_reference_occupies_the_only_primates_slot() -> None:
    candidates = _order_candidates()
    primate = candidates.row(0, named=True) | {
        "alignment_name": "Microcebus_murinus",
        "scientific_name": "Microcebus murinus",
        "assembly": "GCA_000165445.3",
        "taxonomy_id": 30608,
        "family": "Cheirogaleidae",
        "order": "Primates",
    }
    selected = select_order_representatives(
        pl.concat([candidates, pl.DataFrame([primate])], how="vertical")
    )
    assert "Primates" not in set(selected["order"].to_list())
    assert "Microcebus_murinus" not in set(selected["alignment_name"].to_list())


def test_validate_order_manifest_rejects_non_source_row() -> None:
    source = species_manifest()
    selected = select_order_representatives(_order_candidates()).with_columns(
        pl.when(pl.col("alignment_name") == "Mus_musculus")
        .then(pl.lit("wrong assembly"))
        .otherwise(pl.col("assembly"))
        .alias("assembly")
    )
    with pytest.raises(AssertionError, match="exact rows"):
        validate_order_manifest(selected, source)


def test_committed_order_manifest_is_valid() -> None:
    project = Path(__file__).resolve().parents[1]
    manifest = read_order_manifest(
        project / "config/species_vertebrate_order.tsv",
        project / "config/species_selected.tsv",
    )
    assert manifest.height == 39
    assert manifest.filter(pl.col("backend") == "zoonomia_cactus").height == 18
    assert manifest.filter(pl.col("backend") == "ucsc_multiz100way").height == 21
    assert set(manifest["alignment_name"].to_list()) >= {
        "Bos_taurus",
        "Mus_musculus",
        "galGal4",
        "xenTro7",
        "petMar2",
    }
    assert "Primates" not in set(manifest["order"].to_list())
