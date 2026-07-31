from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from marin_dna.pipelines.vertebrate_projection_dataset.pipeline_io import (
    write_dataset_split_files,
)
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


def test_streaming_split_writes_schema_matched_outputs(tmp_path: Path) -> None:
    combined = (
        _rows()
        .filter(pl.col("augmentation") == "+")
        .drop("augmentation")
        .with_columns(pl.lit("cds").alias("region_label"))
    )
    combined_path = tmp_path / "combined.parquet"
    combined.write_parquet(combined_path)
    train_path = tmp_path / "train.parquet"
    validation_path = tmp_path / "validation.parquet"
    selection_path = tmp_path / "selection.tsv"
    counts_path = tmp_path / "counts.tsv"
    summary_path = tmp_path / "summary.json"

    write_dataset_split_files(
        combined_path,
        train_path,
        validation_path,
        selection_path,
        counts_path,
        summary_path,
        region_label="cds",
        add_rc=True,
        validation_chrom="chr18",
        max_validation_rows=5,
        seed=7,
    )

    train = pl.read_parquet(train_path)
    validation = pl.read_parquet(validation_path)
    assert train.schema == validation.schema
    assert train.height == 6
    assert set(train["augmentation"]) == {"+", "-"}
    assert validation.height == 5
    assert set(validation["augmentation"]) == {"+"}
    assert "row_id" not in validation.columns
    assert pl.read_csv(selection_path, separator="\t").height == 5
    assert pl.read_csv(counts_path, separator="\t").height == 3
    assert json.loads(summary_path.read_text())["train_rows"] == 6


def test_mammals_only_scope_filters_multiz_after_region_selection(
    tmp_path: Path,
) -> None:
    combined = (
        _rows()
        .filter(pl.col("augmentation") == "+")
        .drop("augmentation")
        .with_columns(
            pl.lit("cds").alias("region_label"),
            pl.when(pl.col("species") == "Gallus gallus")
            .then(pl.lit("ucsc_multiz100way"))
            .otherwise(pl.col("alignment_source"))
            .alias("alignment_source"),
        )
    )
    combined_path = tmp_path / "combined.parquet"
    combined.write_parquet(combined_path)
    train_path = tmp_path / "train.parquet"
    validation_path = tmp_path / "validation.parquet"
    summary_path = tmp_path / "summary.json"

    write_dataset_split_files(
        combined_path,
        train_path,
        validation_path,
        tmp_path / "selection.tsv",
        tmp_path / "counts.tsv",
        summary_path,
        region_label="cds",
        species_scope="mammals_only",
        add_rc=True,
        validation_chrom="chr18",
        max_validation_rows=4,
        seed=7,
    )

    train = pl.read_parquet(train_path)
    validation = pl.read_parquet(validation_path)
    assert set(train["species"]) == {"Homo sapiens", "Mus musculus"}
    assert set(validation["species"]) == {"Homo sapiens", "Mus musculus"}
    assert "ucsc_multiz100way" not in set(train["alignment_source"])
    summary = json.loads(summary_path.read_text())
    assert summary["region_label"] == "cds"
    assert summary["species_scope"] == "mammals_only"
