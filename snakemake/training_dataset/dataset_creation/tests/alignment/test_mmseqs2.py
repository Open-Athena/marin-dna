"""Tests for marin_dna_training_dataset.alignment.mmseqs2 hits parsing / projection helpers."""

from __future__ import annotations

import polars as pl
from marin_dna_training_dataset.alignment.mmseqs2 import (
    MMSEQS2_HITS_SCHEMA,
    PROJECTED_SCHEMA,
    best_hit_per_query,
    parse_mmseqs2_hits,
    project_hits_to_intervals,
)


def _hit_line(
    query: str,
    target: str,
    tstart: int,
    tend: int,
    bits: float,
    evalue: float = 1e-10,
    fident: float = 0.9,
    qcov: float = 1.0,
    tcov: float = 1.0,
) -> str:
    fields = [query, target, tstart, tend, bits, evalue, fident, qcov, tcov]
    return "\t".join(str(f) for f in fields) + "\n"


def test_parse_mmseqs2_hits_empty(tmp_path):
    hits = tmp_path / "empty.tsv"
    hits.write_text("")
    df = parse_mmseqs2_hits(hits)
    assert df.height == 0
    assert df.schema == MMSEQS2_HITS_SCHEMA


def test_parse_mmseqs2_hits_basic(tmp_path):
    hits = tmp_path / "basic.tsv"
    hits.write_text(_hit_line("q1", "chr1", 201, 300, 123.4))
    df = parse_mmseqs2_hits(hits)
    assert df.height == 1
    row = df.to_dicts()[0]
    assert row["query"] == "q1"
    assert row["target"] == "chr1"
    assert row["tstart"] == 201
    assert row["tend"] == 300
    assert row["bits"] == 123.4


def test_project_forward_strand_1based_to_0based_half_open():
    df = pl.DataFrame(
        [{"query": "q1", "target": "chr1", "tstart": 201, "tend": 300, "bits": 100.0}],
        schema=MMSEQS2_HITS_SCHEMA,
    )
    proj = project_hits_to_intervals(df)
    assert proj.schema == PROJECTED_SCHEMA
    row = proj.to_dicts()[0]
    # 1-based [201, 300] closed → 0-based [200, 300) half-open.
    assert row["start"] == 200
    assert row["end"] == 300
    assert row["rev_strand"] is False
    assert row["chrom"] == "chr1"


def test_project_reverse_strand_normalizes_coords():
    df = pl.DataFrame(
        [{"query": "q1", "target": "chr1", "tstart": 500, "tend": 401, "bits": 80.0}],
        schema=MMSEQS2_HITS_SCHEMA,
    )
    proj = project_hits_to_intervals(df)
    row = proj.to_dicts()[0]
    # min(tstart, tend) = 401 → 0-based start = 400; max = 500 → end = 500.
    assert row["start"] == 400
    assert row["end"] == 500
    assert row["rev_strand"] is True


def test_project_empty_preserves_schema():
    empty = pl.DataFrame(schema=MMSEQS2_HITS_SCHEMA)
    proj = project_hits_to_intervals(empty)
    assert proj.height == 0
    assert proj.schema == PROJECTED_SCHEMA


def test_best_hit_per_query_picks_highest_bits():
    proj = pl.DataFrame(
        [
            {
                "query": "q1",
                "chrom": "chr1",
                "start": 0,
                "end": 100,
                "rev_strand": False,
                "bits": 50.0,
            },
            {
                "query": "q1",
                "chrom": "chr1",
                "start": 500,
                "end": 600,
                "rev_strand": False,
                "bits": 90.0,
            },
            {
                "query": "q2",
                "chrom": "chr3",
                "start": 10,
                "end": 90,
                "rev_strand": True,
                "bits": 70.0,
            },
        ],
        schema=PROJECTED_SCHEMA,
    )
    best = best_hit_per_query(proj)
    assert best.height == 2
    q1 = best.filter(pl.col("query") == "q1").to_dicts()[0]
    assert q1["bits"] == 90.0
    assert q1["start"] == 500
    q2 = best.filter(pl.col("query") == "q2").to_dicts()[0]
    assert q2["chrom"] == "chr3"
    assert q2["rev_strand"] is True


def test_best_hit_per_query_empty():
    empty = pl.DataFrame(schema=PROJECTED_SCHEMA)
    out = best_hit_per_query(empty)
    assert out.height == 0
    assert out.schema == PROJECTED_SCHEMA
