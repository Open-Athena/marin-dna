"""Tests for ``marin_dna_zoonomia_projection.projection.hal``."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from marin_dna_zoonomia_projection.projection.hal import (
    HALLIFTOVER_BED_SCHEMA,
    attach_src_size,
    parse_halliftover_bed,
)


def _write_bed(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def test_parse_empty_bed_returns_schema_correct_frame(tmp_path: Path) -> None:
    bed = tmp_path / "empty.bed"
    bed.write_text("")
    df = parse_halliftover_bed(bed, species="Mus_musculus")
    assert df.height == 0
    expected = {**HALLIFTOVER_BED_SCHEMA, "species": pl.Utf8}
    assert dict(df.schema) == expected


def test_parse_nonempty_bed(tmp_path: Path) -> None:
    bed = tmp_path / "lift.bed"
    _write_bed(
        bed,
        [
            "chr1\t100\t355\twin_1_000000001\t0\t+",
            "chr1\t500\t755\twin_1_000000002\t0\t-",
        ],
    )
    df = parse_halliftover_bed(bed, species="Mus_musculus")
    assert df.height == 2
    assert df["species"].to_list() == ["Mus_musculus", "Mus_musculus"]
    assert df["query_name"].to_list() == ["win_1_000000001", "win_1_000000002"]
    assert df["t_strand"].to_list() == ["+", "-"]
    assert df["t_start"].to_list() == [100, 500]
    assert df["t_end"].to_list() == [355, 755]


def test_attach_src_size_inner_joins_and_drops_missing_chroms(
    tmp_path: Path,
) -> None:
    sizes = tmp_path / "chrom.sizes"
    sizes.write_text("chr1\t1000000\nchr2\t500000\n")
    records = pl.DataFrame(
        {
            "t_chrom": ["chr1", "chr2", "chr_missing"],
            "t_start": [10, 20, 30],
            "t_end": [40, 50, 60],
            "query_name": ["a", "b", "c"],
            "species": ["sp", "sp", "sp"],
            "score": [0, 0, 0],
            "t_strand": ["+", "-", "+"],
        }
    )
    out = attach_src_size(records, sizes)
    # chr_missing gets dropped by the inner join.
    assert out.height == 2
    assert set(out["t_chrom"].to_list()) == {"chr1", "chr2"}
    assert set(out["t_src_size"].to_list()) == {1000000, 500000}


def test_parse_bed_dtypes_are_canonical(tmp_path: Path) -> None:
    """Schema should not silently shift to Int32 etc on small files."""
    bed = tmp_path / "lift.bed"
    _write_bed(bed, ["chr1\t10\t20\tn\t0\t+"])
    df = parse_halliftover_bed(bed, species="X")
    assert df.schema["t_start"] == pl.Int64
    assert df.schema["t_end"] == pl.Int64
    assert df.schema["score"] == pl.Int64
    assert df.schema["t_chrom"] == pl.Utf8


@pytest.mark.parametrize("species", ["Homo_sapiens", "Mus_musculus", "Bos_taurus"])
def test_parse_bed_species_column_is_constant(tmp_path: Path, species: str) -> None:
    bed = tmp_path / "lift.bed"
    _write_bed(
        bed,
        [
            "chr1\t10\t20\tq1\t0\t+",
            "chr2\t30\t40\tq2\t0\t-",
            "chr3\t50\t60\tq3\t0\t+",
        ],
    )
    df = parse_halliftover_bed(bed, species=species)
    assert df["species"].unique().to_list() == [species]
