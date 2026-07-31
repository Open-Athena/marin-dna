from __future__ import annotations

import polars as pl

from marin_dna.pipelines.vertebrate_projection_dataset.split import (
    add_stable_row_ids,
    assign_train_validation_splits,
)


def _rows() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for species in ["Homo sapiens", "Mus musculus", "Gallus gallus"]:
        for index in range(3):
            rows.append(
                {
                    "query_name": f"chr18_{index}",
                    "source_chrom": "chr18",
                    "source_start": index * 255,
                    "source_end": (index + 1) * 255,
                    "species": species,
                    "alignment_source": "human_reference"
                    if species == "Homo sapiens"
                    else "zoonomia_cactus",
                    "augmentation": "+",
                    "sequence": "A" * 255,
                }
            )
        rows.append(
            {
                "query_name": "chr18_0",
                "source_chrom": "chr18",
                "source_start": 0,
                "source_end": 255,
                "species": species,
                "alignment_source": "human_reference"
                if species == "Homo sapiens"
                else "zoonomia_cactus",
                "augmentation": "-",
                "sequence": "T" * 255,
            }
        )
        rows.append(
            {
                "query_name": "chr1_0",
                "source_chrom": "chr1",
                "source_start": 0,
                "source_end": 255,
                "species": species,
                "alignment_source": "human_reference"
                if species == "Homo sapiens"
                else "zoonomia_cactus",
                "augmentation": "+",
                "sequence": "C" * 255,
            }
        )
    return pl.DataFrame(rows)


def test_chr18_split_is_species_stratified_and_drops_rc_candidates() -> None:
    result = assign_train_validation_splits(_rows(), max_validation_rows=5, seed=7)
    assert result.train.height == 3
    assert result.validation.height == 5
    assert set(result.validation["species"].to_list()) == {
        "Homo sapiens",
        "Mus musculus",
        "Gallus gallus",
    }
    assert set(result.validation["augmentation"].to_list()) == {"+"}
    assert result.species_counts["selected_rows"].sort().to_list() == [1, 2, 2]
    assert result.realized_token_count == 5 * 256
    assert result.selection_manifest.height == result.validation.height


def test_split_selection_is_reproducible_under_row_reordering() -> None:
    rows = add_stable_row_ids(_rows())
    forward = assign_train_validation_splits(rows, max_validation_rows=5, seed=11)
    reverse = assign_train_validation_splits(
        rows.reverse(), max_validation_rows=5, seed=11
    )
    assert forward.selection_manifest.equals(reverse.selection_manifest)


def test_all_chr18_original_rows_are_retained_when_below_cap() -> None:
    rows = _rows().filter(pl.col("augmentation") == "+")
    result = assign_train_validation_splits(rows, max_validation_rows=20, seed=42)
    assert result.validation.height == 9
    assert result.realized_token_count == 9 * 256
