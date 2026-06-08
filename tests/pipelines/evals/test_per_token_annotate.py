"""Tests for per-position codon annotation (issue #296).

The off-by-one gate for the headline metric (LL gap restricted to codon
positions 1+2). Covers + and − strand numbering from per-segment GTF phase, a
codon split across a splice junction on both strands, partial / 5'-incomplete
CDS (non-zero phase), invalid phase, and the left-join contract.
"""

from __future__ import annotations

import polars as pl
import pytest

from marin_dna.pipelines.evals.per_token_annotate import (
    assign_codon_positions,
    cds_codon_positions,
    flag_fourfold_degenerate,
    intron_splice_regions,
)

_SCHEMA = ["chrom", "start", "end", "strand", "transcript_id", "frame"]
_EXON_SCHEMA = ["chrom", "start", "end", "strand", "transcript_id"]


def _cds(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=_SCHEMA, orient="row")


def _exons(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=_EXON_SCHEMA, orient="row")


def _cp_map(cds: pl.DataFrame) -> dict[int, int]:
    out = cds_codon_positions(cds).sort("genomic_pos")
    return dict(zip(out["genomic_pos"].to_list(), out["codon_pos"].to_list()))


def test_plus_strand_single_segment():
    # Complete CDS (phase 0): 5'→3' left-to-right gives 1,2,3,1,2,3.
    assert _cp_map(_cds([("1", 10, 16, "+", "T1", "0")])) == {
        10: 1,
        11: 2,
        12: 3,
        13: 1,
        14: 2,
        15: 3,
    }


def test_minus_strand_single_segment():
    # − strand reads high→low: the highest coordinate is codon position 1.
    assert _cp_map(_cds([("1", 10, 16, "-", "T1", "0")])) == {
        15: 1,
        14: 2,
        13: 3,
        12: 1,
        11: 2,
        10: 3,
    }


def test_plus_strand_split_codon_across_junction():
    # seg1 phase 0 (2 bases); seg2 phase 1 — its first base finishes the codon
    # started in seg1 (base 11 = pos 2, base 20 = pos 3).
    cds = _cds([("1", 10, 12, "+", "T2", "0"), ("1", 20, 23, "+", "T2", "1")])
    assert _cp_map(cds) == {10: 1, 11: 2, 20: 3, 21: 1, 22: 2}


def test_minus_strand_split_codon_across_junction():
    cds = _cds([("1", 10, 12, "-", "T3", "0"), ("1", 20, 23, "-", "T3", "0")])
    # Transcription order 22,21,20 (5' segment) then 11,10.
    assert _cp_map(cds) == {22: 1, 21: 2, 20: 3, 11: 1, 10: 2}


def test_partial_cds_nonzero_phase():
    # A 5'-incomplete CDS (GTF phase 2, e.g. a TR/IG segment): the first base is
    # the 2nd base of a codon — frame-free numbering would mislabel it as pos 1.
    assert _cp_map(_cds([("1", 10, 13, "+", "T4", "2")])) == {10: 2, 11: 3, 12: 1}


def test_invalid_frame_raises():
    with pytest.raises(AssertionError, match="phase 0/1/2"):
        cds_codon_positions(_cds([("1", 10, 13, "+", "T5", ".")]))


def test_intron_splice_regions_plus_strand_donor_acceptor():
    # + strand, intron [50,100): low side (donor) [50,70), high side (acceptor)
    # [80,100).
    ex = _exons([("1", 0, 50, "+", "T1"), ("1", 100, 150, "+", "T1")])
    out = intron_splice_regions(ex, flank=20).sort("start")
    assert out.select(["start", "end", "side", "strand"]).rows() == [
        (50, 70, "donor", "+"),
        (80, 100, "acceptor", "+"),
    ]


def test_intron_splice_regions_minus_strand_flips_sides():
    # Same intron on − strand: donor/acceptor swap (gene reads high→low).
    ex = _exons([("1", 0, 50, "-", "T1"), ("1", 100, 150, "-", "T1")])
    out = intron_splice_regions(ex, flank=20).sort("start")
    assert out.select(["start", "end", "side", "strand"]).rows() == [
        (50, 70, "acceptor", "-"),
        (80, 100, "donor", "-"),
    ]


def test_intron_splice_regions_short_intron_collapses():
    # Intron [50,60) shorter than 2·flank → whole intron on both sides.
    ex = _exons([("1", 0, 50, "+", "T1"), ("1", 60, 100, "+", "T1")])
    out = intron_splice_regions(ex, flank=20).sort("side")
    assert set(out.select(["start", "end", "side"]).rows()) == {
        (50, 60, "donor"),
        (50, 60, "acceptor"),
    }


def test_intron_splice_regions_single_exon_empty():
    assert len(intron_splice_regions(_exons([("1", 0, 50, "+", "T1")]))) == 0


def _coding(rows: list[tuple]) -> pl.DataFrame:
    # (chrom, genomic_pos, transcript_id, codon_id, codon_pos, ref_base)
    return pl.DataFrame(
        rows,
        schema=[
            "chrom",
            "genomic_pos",
            "transcript_id",
            "codon_id",
            "codon_pos",
            "ref_base",
        ],
        orient="row",
    )


def test_flag_fourfold_degenerate():
    # Codon 1 = GTC (Val, GT* 4-fold → pos3 @102 is 4-fold);
    # codon 2 = ATG (Met, AT* not 4-fold → pos3 @202 not 4-fold);
    # codon 3 missing base 2 (window edge) → dropped.
    coding = _coding(
        [
            ("1", 100, "T1", 1, 1, "G"),
            ("1", 101, "T1", 1, 2, "T"),
            ("1", 102, "T1", 1, 3, "C"),
            ("1", 200, "T1", 2, 1, "A"),
            ("1", 201, "T1", 2, 2, "T"),
            ("1", 202, "T1", 2, 3, "G"),
            ("1", 301, "T1", 3, 1, "C"),  # codon 3 has no pos-2 base
            ("1", 302, "T1", 3, 3, "A"),
        ]
    )
    out = flag_fourfold_degenerate(coding).sort("genomic_pos")
    got = dict(zip(out["genomic_pos"].to_list(), out["is_4fold"].to_list()))
    assert got == {102: True, 202: False}  # codon 3 dropped (incomplete)


def test_assign_left_join_marks_noncoding_null():
    cds = _cds([("1", 10, 13, "+", "T1", "0")])
    q = pl.DataFrame({"chrom": ["1", "1", "1"], "genomic_pos": [10, 12, 99]})
    out = assign_codon_positions(cds, q).sort("genomic_pos")
    cp = dict(zip(out["genomic_pos"].to_list(), out["codon_pos"].to_list()))
    assert cp[10] == 1 and cp[12] == 3
    assert cp[99] is None  # non-coding → null
