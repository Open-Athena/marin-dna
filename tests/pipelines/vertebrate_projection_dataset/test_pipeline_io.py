from __future__ import annotations

from pathlib import Path

import polars as pl

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
