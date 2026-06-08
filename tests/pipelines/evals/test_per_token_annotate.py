"""Tests for per-position codon annotation (issue #296).

The off-by-one gate for the headline metric (LL gap restricted to codon
positions 1+2). Covers + and − strand numbering, a codon split across a splice
junction on both strands, the GTF-frame cross-check, and the left-join contract.
"""

from __future__ import annotations

import polars as pl
import pytest

from marin_dna.pipelines.evals.per_token_annotate import (
    assign_codon_positions,
    cds_codon_positions,
)


def _cp_map(cds: pl.DataFrame) -> dict[int, int]:
    out = cds_codon_positions(cds).sort("genomic_pos")
    return dict(zip(out["genomic_pos"].to_list(), out["codon_pos"].to_list()))


def test_plus_strand_single_segment():
    cds = pl.DataFrame(
        {
            "chrom": ["1"],
            "start": [10],
            "end": [16],
            "strand": ["+"],
            "transcript_id": ["T1"],
        }
    )
    # Reading 5'→3' left-to-right from the CDS start: 1,2,3,1,2,3.
    assert _cp_map(cds) == {10: 1, 11: 2, 12: 3, 13: 1, 14: 2, 15: 3}


def test_minus_strand_single_segment():
    cds = pl.DataFrame(
        {
            "chrom": ["1"],
            "start": [10],
            "end": [16],
            "strand": ["-"],
            "transcript_id": ["T1"],
        }
    )
    # − strand reads high→low: the highest coordinate is codon position 1.
    assert _cp_map(cds) == {15: 1, 14: 2, 13: 3, 12: 1, 11: 2, 10: 3}


def test_plus_strand_split_codon_across_junction():
    # Two CDS segments of one transcript; the reading frame must continue across
    # the intron: base 11 (pos 2) and base 20 (pos 3) are the same codon.
    cds = pl.DataFrame(
        {
            "chrom": ["1", "1"],
            "start": [10, 20],
            "end": [12, 23],
            "strand": ["+", "+"],
            "transcript_id": ["T2", "T2"],
        }
    )
    assert _cp_map(cds) == {10: 1, 11: 2, 20: 3, 21: 1, 22: 2}


def test_minus_strand_split_codon_across_junction():
    cds = pl.DataFrame(
        {
            "chrom": ["1", "1"],
            "start": [10, 20],
            "end": [12, 23],
            "strand": ["-", "-"],
            "transcript_id": ["T3", "T3"],
        }
    )
    # Transcription order: 22,21,20 (seg2, higher coords) then 11,10 (seg1).
    assert _cp_map(cds) == {22: 1, 21: 2, 20: 3, 11: 1, 10: 2}


def test_frame_cross_check_passes_skips_and_fails():
    # + transcript, segments at offsets 0 and 2 → expected GTF frame 0 and 1.
    good = pl.DataFrame(
        {
            "chrom": ["1", "1"],
            "start": [10, 20],
            "end": [12, 23],
            "strand": ["+", "+"],
            "transcript_id": ["T2", "T2"],
            "frame": ["0", "1"],
        }
    )
    cds_codon_positions(good)  # consistent → no raise

    bad = good.with_columns(pl.Series("frame", ["0", "2"]))  # seg2 should be 1
    with pytest.raises(AssertionError, match="codon-frame cross-check"):
        cds_codon_positions(bad)

    dotted = good.with_columns(pl.Series("frame", [".", "."]))  # '.' phases skipped
    cds_codon_positions(dotted)  # no raise


def test_assign_left_join_marks_noncoding_null():
    cds = pl.DataFrame(
        {
            "chrom": ["1"],
            "start": [10],
            "end": [13],
            "strand": ["+"],
            "transcript_id": ["T1"],
        }
    )
    q = pl.DataFrame({"chrom": ["1", "1", "1"], "genomic_pos": [10, 12, 99]})
    out = assign_codon_positions(cds, q).sort("genomic_pos")
    cp = dict(zip(out["genomic_pos"].to_list(), out["codon_pos"].to_list()))
    assert cp[10] == 1 and cp[12] == 3
    assert cp[99] is None  # non-coding → null
