from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from marin_dna.pipelines.vertebrate_projection_dataset.pipeline_io import (
    combine_sequence_parquets,
)


def test_combine_sequence_parquets_accepts_one_species_per_input(
    tmp_path: Path,
) -> None:
    paths: list[str] = []
    for index, species in enumerate(["Homo sapiens", "Mus musculus"]):
        path = tmp_path / f"species-{index}.parquet"
        pl.DataFrame(
            {
                "query_name": ["anchor-1", "anchor-2"],
                "species": [species, species],
                "sequence": ["A" * 255, "C" * 255],
            }
        ).write_parquet(path)
        paths.append(str(path))

    output = tmp_path / "combined.parquet"
    combine_sequence_parquets(paths, output)

    combined = pl.read_parquet(output)
    assert combined.height == 4
    assert set(combined["species"]) == {"Homo sapiens", "Mus musculus"}


def test_combine_sequence_parquets_accepts_schema_valid_empty_input(
    tmp_path: Path,
) -> None:
    populated = tmp_path / "populated.parquet"
    empty = tmp_path / "empty.parquet"
    frame = pl.DataFrame(
        {
            "query_name": ["anchor-1"],
            "species": ["Homo sapiens"],
            "sequence": ["A" * 255],
        }
    )
    frame.write_parquet(populated)
    frame.clear().write_parquet(empty)

    output = tmp_path / "combined.parquet"
    combine_sequence_parquets([str(populated), str(empty)], output)

    assert pl.read_parquet(output).to_dicts() == frame.to_dicts()


def test_combine_sequence_parquets_rejects_empty_input_with_wrong_schema(
    tmp_path: Path,
) -> None:
    populated = tmp_path / "populated.parquet"
    malformed = tmp_path / "malformed.parquet"
    pl.DataFrame(
        {
            "query_name": ["anchor-1"],
            "species": ["Homo sapiens"],
            "sequence": ["A" * 255],
        }
    ).write_parquet(populated)
    pl.DataFrame(schema={"query_name": pl.String}).write_parquet(malformed)

    with pytest.raises(AssertionError):
        combine_sequence_parquets(
            [str(populated), str(malformed)], tmp_path / "combined.parquet"
        )
