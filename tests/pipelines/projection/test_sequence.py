"""Tests for ``marin_dna.pipelines.projection.sequence``."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from marin_dna.pipelines.projection.sequence import (
    attach_sequences_to_parquet,
    parquet_to_bed6,
    parse_bedtools_getfasta_output,
)


def _make_parquet(tmp_path: Path, rows: list[dict]) -> Path:
    schema = {
        "query_name": pl.Utf8,
        "species": pl.Utf8,
        "t_chrom": pl.Utf8,
        "t_start": pl.Int64,
        "t_end": pl.Int64,
        "t_strand": pl.Utf8,
        "t_src_size": pl.Int64,
    }
    pq = tmp_path / "proj.parquet"
    pl.DataFrame(rows, schema=schema).write_parquet(pq)
    return pq


def test_parquet_to_bed6_writes_six_cols(tmp_path: Path) -> None:
    pq = _make_parquet(
        tmp_path,
        [
            {
                "query_name": "win_1_000000001",
                "species": "Mus_musculus",
                "t_chrom": "chr5",
                "t_start": 100,
                "t_end": 355,
                "t_strand": "+",
                "t_src_size": 1_000_000,
            },
            {
                "query_name": "win_1_000000002",
                "species": "Mus_musculus",
                "t_chrom": "chr5",
                "t_start": 500,
                "t_end": 755,
                "t_strand": "-",
                "t_src_size": 1_000_000,
            },
        ],
    )
    bed = tmp_path / "out.bed"
    n = parquet_to_bed6(pq, bed)
    assert n == 2
    assert bed.read_text() == (
        "chr5\t100\t355\twin_1_000000001\t0\t+\nchr5\t500\t755\twin_1_000000002\t0\t-\n"
    )


def test_parquet_to_bed6_empty(tmp_path: Path) -> None:
    pq = _make_parquet(tmp_path, [])
    bed = tmp_path / "out.bed"
    n = parquet_to_bed6(pq, bed)
    assert n == 0
    assert bed.exists()
    assert bed.read_text() == ""


def test_parse_bedtools_getfasta_output_basic(tmp_path: Path) -> None:
    """Two-line-per-record FASTA, single line per sequence (bedtools default)."""
    fa = tmp_path / "seqs.fa"
    fa.write_text(
        ">win_1_000000001(+)\nACGTACGTACGT\n>win_1_000000002(-)\nTTTTAAAACCCC\n"
    )
    seqs = parse_bedtools_getfasta_output(fa)
    assert seqs == ["ACGTACGTACGT", "TTTTAAAACCCC"]


def test_parse_bedtools_getfasta_output_empty(tmp_path: Path) -> None:
    fa = tmp_path / "empty.fa"
    fa.write_text("")
    assert parse_bedtools_getfasta_output(fa) == []


def test_parse_bedtools_getfasta_output_rejects_malformed(tmp_path: Path) -> None:
    """Header line missing where one is expected → loud failure."""
    fa = tmp_path / "bad.fa"
    fa.write_text(
        ">name1\n"
        "ACGT\n"
        "no_header_here\n"  # should have been ">name2"
        "TTTT\n"
    )
    with pytest.raises(AssertionError, match="expected header"):
        parse_bedtools_getfasta_output(fa)


def test_parse_bedtools_getfasta_output_rejects_empty_sequence(tmp_path: Path) -> None:
    """Empty sequence line → loud failure pointing at the malformed FASTA row.

    Without this guard, a blank sequence would silently pass through and
    be detected only by ``attach_sequences_to_parquet``'s length check —
    which then points at the wrong stage.
    """
    fa = tmp_path / "bad.fa"
    fa.write_text(">name1\n\n")
    with pytest.raises(AssertionError, match="empty sequence"):
        parse_bedtools_getfasta_output(fa)


def test_attach_sequences_to_parquet_writes_extended_schema(tmp_path: Path) -> None:
    pq = _make_parquet(
        tmp_path,
        [
            {
                "query_name": "win_a",
                "species": "Mus_musculus",
                "t_chrom": "chr5",
                "t_start": 100,
                "t_end": 355,
                "t_strand": "+",
                "t_src_size": 1_000_000,
            },
            {
                "query_name": "win_b",
                "species": "Mus_musculus",
                "t_chrom": "chr5",
                "t_start": 500,
                "t_end": 755,
                "t_strand": "-",
                "t_src_size": 1_000_000,
            },
        ],
    )
    out = tmp_path / "with_seq.parquet"
    n = attach_sequences_to_parquet(
        pq,
        ["A" * 255, "T" * 255],
        out,
        target_len=255,
    )
    assert n == 2
    df = pl.read_parquet(out)
    assert df.height == 2
    assert "sequence" in df.columns
    assert df["sequence"].to_list() == ["A" * 255, "T" * 255]


def test_attach_sequences_to_parquet_length_mismatch_raises(tmp_path: Path) -> None:
    pq = _make_parquet(
        tmp_path,
        [
            {
                "query_name": "win_a",
                "species": "X",
                "t_chrom": "chr1",
                "t_start": 0,
                "t_end": 255,
                "t_strand": "+",
                "t_src_size": 1000,
            }
        ],
    )
    out = tmp_path / "out.parquet"
    with pytest.raises(AssertionError, match="unexpected sequence length"):
        attach_sequences_to_parquet(pq, ["A" * 9], out, target_len=255)


def test_attach_sequences_to_parquet_count_mismatch_raises(tmp_path: Path) -> None:
    pq = _make_parquet(
        tmp_path,
        [
            {
                "query_name": "win_a",
                "species": "X",
                "t_chrom": "chr1",
                "t_start": 0,
                "t_end": 4,
                "t_strand": "+",
                "t_src_size": 1000,
            }
        ],
    )
    out = tmp_path / "out.parquet"
    # Pass 2 sequences but Parquet has 1 row.
    with pytest.raises(AssertionError, match="!="):
        attach_sequences_to_parquet(pq, ["ACGT", "TTTT"], out, target_len=4)


def test_attach_sequences_to_parquet_empty(tmp_path: Path) -> None:
    pq = _make_parquet(tmp_path, [])
    out = tmp_path / "out.parquet"
    n = attach_sequences_to_parquet(pq, [], out, target_len=255)
    assert n == 0
    df = pl.read_parquet(out)
    assert df.height == 0
    assert "sequence" in df.columns
