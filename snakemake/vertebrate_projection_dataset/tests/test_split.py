import json
from pathlib import Path

import polars as pl
import pytest
from marin_dna_vertebrate_projection.pipeline_io import (
    write_dataset_split_files,
)
from marin_dna_vertebrate_projection.split import (
    VALIDATION_IDENTITY_COLUMNS,
    select_uniform_validation_rows,
)


def _rows() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    species_sources = [
        ("Homo sapiens", "human_reference"),
        ("Mus musculus", "zoonomia_cactus"),
        ("Gallus gallus", "ucsc_multiz100way"),
    ]
    chromosomes = ["chr1", "chr18", "chr2"]
    for index in range(10):
        for species, alignment_source in species_sources:
            rows.append(
                {
                    "query_name": f"anchor_{index}",
                    "source_chrom": chromosomes[index % len(chromosomes)],
                    "source_start": index * 255,
                    "source_end": (index + 1) * 255,
                    "species": species,
                    "alignment_source": alignment_source,
                    "region_label": "cds",
                    "sequence": "A" * 255,
                }
            )
    return pl.DataFrame(rows)


def test_uniform_selection_is_reproducible_under_row_reordering() -> None:
    rows = _rows()
    forward = select_uniform_validation_rows(
        rows,
        validation_rows=8,
        seed=42,
        selection_salt="region=cds|species_scope=all",
    )
    reverse = select_uniform_validation_rows(
        rows.reverse(),
        validation_rows=8,
        seed=42,
        selection_salt="region=cds|species_scope=all",
    )
    assert forward.equals(reverse)
    assert forward.height == 8
    assert forward["row_id"].n_unique() == 8
    assert forward["selection_rank"].to_list() == list(range(1, 9))


def test_cohort_salt_decouples_nested_cohort_samples() -> None:
    rows = _rows()
    combined = select_uniform_validation_rows(
        rows,
        validation_rows=8,
        seed=42,
        selection_salt="region=cds|species_scope=all",
    )
    mammals_only = select_uniform_validation_rows(
        rows,
        validation_rows=8,
        seed=42,
        selection_salt="region=cds|species_scope=mammals_only",
    )
    assert set(combined["row_id"]) != set(mammals_only["row_id"])
    assert (
        combined["selection_hash"].to_list() != mammals_only["selection_hash"].to_list()
    )


def test_uniform_selection_precedes_augmentation() -> None:
    with pytest.raises(AssertionError, match="must precede"):
        select_uniform_validation_rows(
            _rows().with_columns(pl.lit("+").alias("augmentation")),
            validation_rows=8,
            seed=42,
            selection_salt="region=cds|species_scope=all",
        )


def test_streaming_split_writes_random_schema_matched_outputs(tmp_path: Path) -> None:
    combined_path = tmp_path / "combined.parquet"
    _rows().write_parquet(combined_path)
    train_path = tmp_path / "train.parquet"
    validation_path = tmp_path / "validation.parquet"
    selection_path = tmp_path / "selection.tsv"
    composition_path = tmp_path / "composition.tsv"
    summary_path = tmp_path / "summary.json"

    write_dataset_split_files(
        combined_path,
        train_path,
        validation_path,
        selection_path,
        composition_path,
        summary_path,
        region_label="cds",
        add_rc=True,
        validation_rows=8,
        seed=42,
    )

    train = pl.read_parquet(train_path)
    validation = pl.read_parquet(validation_path)
    selection = pl.read_csv(selection_path, separator="\t")
    assert train.schema == validation.schema
    assert train.height == 44
    assert set(train["augmentation"]) == {"+", "-"}
    assert validation.height == 8
    assert set(validation["augmentation"]) == {"+"}
    assert "row_id" not in validation.columns
    assert selection.height == 8
    assert set(selection["selection_salt"]) == {"region=cds|species_scope=all"}

    selected_keys = set(selection.select(*VALIDATION_IDENTITY_COLUMNS).iter_rows())
    train_keys = set(train.select(*VALIDATION_IDENTITY_COLUMNS).iter_rows())
    validation_keys = set(validation.select(*VALIDATION_IDENTITY_COLUMNS).iter_rows())
    assert validation_keys == selected_keys
    assert train_keys.isdisjoint(selected_keys)
    assert set(train["source_chrom"]) & set(validation["source_chrom"])
    assert set(train["query_name"]) & set(validation["query_name"])

    composition = pl.read_csv(composition_path, separator="\t")
    assert set(composition["dimension"]) == {
        "alignment_source",
        "source_chrom",
        "species",
    }
    for _dimension, group in composition.group_by("dimension"):
        assert group["eligible_rows"].sum() == 30
        assert group["selected_rows"].sum() == 8

    summary = json.loads(summary_path.read_text())
    assert summary == {
        "add_reverse_complements": True,
        "realized_token_count": 8 * 256,
        "region_label": "cds",
        "seed": 42,
        "selection_salt": "region=cds|species_scope=all",
        "source_rows": 30,
        "species_scope": "all",
        "split_strategy": "uniform_row_random_before_reverse_complement",
        "train_original_rows": 22,
        "train_rows": 44,
        "validation_rows": 8,
    }


def test_mammals_only_scope_filters_multiz_before_random_selection(
    tmp_path: Path,
) -> None:
    combined_path = tmp_path / "combined.parquet"
    _rows().write_parquet(combined_path)
    train_path = tmp_path / "train.parquet"
    validation_path = tmp_path / "validation.parquet"
    summary_path = tmp_path / "summary.json"

    write_dataset_split_files(
        combined_path,
        train_path,
        validation_path,
        tmp_path / "selection.tsv",
        tmp_path / "composition.tsv",
        summary_path,
        region_label="cds",
        species_scope="mammals_only",
        add_rc=True,
        validation_rows=4,
        seed=7,
    )

    train = pl.read_parquet(train_path)
    validation = pl.read_parquet(validation_path)
    assert set(train["species"]) == {"Homo sapiens", "Mus musculus"}
    assert set(validation["species"]) <= {"Homo sapiens", "Mus musculus"}
    assert "ucsc_multiz100way" not in set(train["alignment_source"])
    summary = json.loads(summary_path.read_text())
    assert summary["region_label"] == "cds"
    assert summary["species_scope"] == "mammals_only"
    assert summary["source_rows"] == 20
    assert summary["train_original_rows"] == 16


def test_split_requires_training_rows_after_validation(tmp_path: Path) -> None:
    combined_path = tmp_path / "combined.parquet"
    _rows().head(3).write_parquet(combined_path)
    with pytest.raises(AssertionError, match="nonempty training"):
        write_dataset_split_files(
            combined_path,
            tmp_path / "train.parquet",
            tmp_path / "validation.parquet",
            tmp_path / "selection.tsv",
            tmp_path / "composition.tsv",
            tmp_path / "summary.json",
            region_label="cds",
            add_rc=True,
            validation_rows=3,
            seed=42,
        )
