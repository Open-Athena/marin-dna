from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from marin_dna.pipelines.vertebrate_projection_dataset.pipeline_io import (
    combine_sequence_parquets,
    write_filtered_anchor_bed,
)


def test_write_filtered_anchor_bed_is_valid_deterministic_gzip(
    tmp_path: Path,
) -> None:
    scored_paths: list[str] = []
    for chrom, offset in [("chr1", 0), ("chr2", 1_000)]:
        path = tmp_path / f"{chrom}.parquet"
        pl.DataFrame(
            {
                "chrom": [chrom, chrom],
                "start": [offset, offset + 255],
                "end": [offset + 255, offset + 510],
                "name": [f"{chrom}-keep", f"{chrom}-drop"],
                "proportion_conserved": [0.8, 0.2],
            }
        ).write_parquet(path)
        scored_paths.append(str(path))

    first = tmp_path / "first.bed.gz"
    second = tmp_path / "second.bed.gz"
    for output in [first, second]:
        write_filtered_anchor_bed(
            scored_paths,
            output,
            min_proportion_conserved=0.65,
        )

    assert first.read_bytes() == second.read_bytes()
    assert pl.read_csv(
        first,
        separator="\t",
        has_header=False,
        new_columns=["chrom", "start", "end", "name"],
    ).to_dicts() == [
        {"chrom": 1, "start": 0, "end": 255, "name": "chr1-keep"},
        {"chrom": 2, "start": 1_000, "end": 1_255, "name": "chr2-keep"},
    ]


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
