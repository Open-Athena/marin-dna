"""Tests for ``marin_dna_zoonomia_projection.projection.filter``."""

from __future__ import annotations

import polars as pl
from marin_dna_zoonomia_projection.projection.filter import (
    filter_length,
    filter_single_chrom_strand,
)


def _records(rows: list[dict]) -> pl.DataFrame:
    schema = {
        "query_name": pl.Utf8,
        "species": pl.Utf8,
        "t_chrom": pl.Utf8,
        "t_start": pl.Int64,
        "t_end": pl.Int64,
        "t_strand": pl.Utf8,
        "t_src_size": pl.Int64,
    }
    return pl.DataFrame(rows, schema=schema)


def test_single_block_passes_through() -> None:
    df = _records(
        [
            {
                "query_name": "q1",
                "species": "Mus_musculus",
                "t_chrom": "chr1",
                "t_start": 100,
                "t_end": 355,
                "t_strand": "+",
                "t_src_size": 100000,
            }
        ]
    )
    out = filter_single_chrom_strand(df)
    assert out.height == 1
    row = out.row(0, named=True)
    assert (row["t_chrom"], row["t_start"], row["t_end"], row["t_strand"]) == (
        "chr1",
        100,
        355,
        "+",
    )


def test_two_blocks_same_chrom_strand_merge_to_min_max() -> None:
    df = _records(
        [
            {
                "query_name": "q1",
                "species": "Mus_musculus",
                "t_chrom": "chr1",
                "t_start": 100,
                "t_end": 200,
                "t_strand": "+",
                "t_src_size": 100000,
            },
            {
                "query_name": "q1",
                "species": "Mus_musculus",
                "t_chrom": "chr1",
                "t_start": 250,
                "t_end": 355,
                "t_strand": "+",
                "t_src_size": 100000,
            },
        ]
    )
    out = filter_single_chrom_strand(df)
    assert out.height == 1
    row = out.row(0, named=True)
    assert row["t_start"] == 100
    assert row["t_end"] == 355


def test_multi_chrom_group_dropped() -> None:
    df = _records(
        [
            {
                "query_name": "q1",
                "species": "Mus_musculus",
                "t_chrom": "chr1",
                "t_start": 100,
                "t_end": 200,
                "t_strand": "+",
                "t_src_size": 100000,
            },
            {
                "query_name": "q1",
                "species": "Mus_musculus",
                "t_chrom": "chr2",
                "t_start": 100,
                "t_end": 200,
                "t_strand": "+",
                "t_src_size": 100000,
            },
        ]
    )
    out = filter_single_chrom_strand(df)
    assert out.height == 0


def test_multi_strand_group_dropped() -> None:
    df = _records(
        [
            {
                "query_name": "q1",
                "species": "Mus_musculus",
                "t_chrom": "chr1",
                "t_start": 100,
                "t_end": 200,
                "t_strand": "+",
                "t_src_size": 100000,
            },
            {
                "query_name": "q1",
                "species": "Mus_musculus",
                "t_chrom": "chr1",
                "t_start": 300,
                "t_end": 400,
                "t_strand": "-",
                "t_src_size": 100000,
            },
        ]
    )
    out = filter_single_chrom_strand(df)
    assert out.height == 0


def test_empty_input_returns_empty_with_schema() -> None:
    df = _records([])
    out = filter_single_chrom_strand(df)
    assert out.height == 0
    assert set(out.columns) >= {
        "query_name",
        "species",
        "t_chrom",
        "t_start",
        "t_end",
        "t_strand",
        "t_src_size",
    }


def test_different_queries_same_species_kept_separate() -> None:
    df = _records(
        [
            {
                "query_name": "q1",
                "species": "Mus_musculus",
                "t_chrom": "chr1",
                "t_start": 100,
                "t_end": 200,
                "t_strand": "+",
                "t_src_size": 100000,
            },
            {
                "query_name": "q2",
                "species": "Mus_musculus",
                "t_chrom": "chr1",
                "t_start": 500,
                "t_end": 600,
                "t_strand": "+",
                "t_src_size": 100000,
            },
        ]
    )
    out = filter_single_chrom_strand(df)
    assert out.height == 2
    assert set(out["query_name"].to_list()) == {"q1", "q2"}


def test_filter_length_inclusive_boundaries() -> None:
    df = _records(
        [
            {
                "query_name": f"q{i}",
                "species": "X",
                "t_chrom": "chr1",
                "t_start": 0,
                "t_end": L,
                "t_strand": "+",
                "t_src_size": 100000,
            }
            for i, L in enumerate([127, 128, 255, 512, 513])
        ]
    )
    out = filter_length(df, min_len=128, max_len=512)
    kept_lengths = (out["t_end"] - out["t_start"]).to_list()
    assert kept_lengths == [128, 255, 512]


def test_filter_length_default_range() -> None:
    df = _records(
        [
            {
                "query_name": "q1",
                "species": "X",
                "t_chrom": "chr1",
                "t_start": 0,
                "t_end": 250,
                "t_strand": "+",
                "t_src_size": 1000,
            },
            {
                "query_name": "q2",
                "species": "X",
                "t_chrom": "chr1",
                "t_start": 0,
                "t_end": 50,
                "t_strand": "+",
                "t_src_size": 1000,
            },
        ]
    )
    out = filter_length(df)
    assert out["query_name"].to_list() == ["q1"]
