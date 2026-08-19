from __future__ import annotations

import numpy as np
import polars as pl
from marin_dna_evals.cds_annotations import (
    annotate_cds_windows,
    canonical_splice_segments,
)


def _cds(rows: list[tuple[object, ...]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema=["chrom", "start", "end", "strand", "transcript_id", "frame"],
        orient="row",
    )


def _exons(rows: list[tuple[object, ...]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema=["chrom", "start", "end", "strand", "transcript_id"],
        orient="row",
    )


def test_codon_position_uses_phase_and_strand() -> None:
    cds = _cds(
        [
            ("1", 10, 13, "+", "plus", 2),
            ("1", 20, 23, "-", "minus", 0),
        ]
    )
    exons = _exons([("1", 10, 13, "+", "plus"), ("1", 20, 23, "-", "minus")])
    codon, strand, splice, _ = annotate_cds_windows([("1", 10, 23)], cds, exons)
    assert codon[0].tolist() == [2, 3, 1] + [0] * 7 + [3, 2, 1]
    assert strand[0].tolist() == [1, 1, 1] + [0] * 7 + [-1, -1, -1]
    assert not splice[0].any()


def test_split_codon_phase_matches_across_exons() -> None:
    cds = _cds(
        [
            ("1", 10, 12, "+", "tx", 0),
            ("1", 20, 23, "+", "tx", 1),
        ]
    )
    exons = _exons([("1", 10, 12, "+", "tx"), ("1", 20, 23, "+", "tx")])
    codon, _, _, _ = annotate_cds_windows([("1", 10, 23)], cds, exons)
    expected = [1, 2] + [0] * 8 + [3, 1, 2]
    assert codon[0].tolist() == expected


def test_canonical_two_base_splice_sites_flip_on_minus_strand() -> None:
    plus = _exons([("1", 0, 10, "+", "p"), ("1", 20, 30, "+", "p")])
    minus = _exons([("2", 0, 10, "-", "m"), ("2", 20, 30, "-", "m")])
    out_plus = canonical_splice_segments(plus)["1"]
    out_minus = canonical_splice_segments(minus)["2"]
    assert [(x.start, x.end, x.label) for x in out_plus] == [
        (10, 12, 1),
        (18, 20, 2),
    ]
    assert [(x.start, x.end, x.label) for x in out_minus] == [
        (10, 12, 2),
        (18, 20, 1),
    ]


def test_overlapping_transcript_disagreement_is_ambiguous() -> None:
    cds = _cds(
        [
            ("1", 10, 13, "+", "a", 0),
            ("1", 10, 13, "+", "b", 1),
        ]
    )
    exons = _exons([("1", 10, 13, "+", "a"), ("1", 10, 13, "+", "b")])
    codon, strand, _, _ = annotate_cds_windows([("1", 10, 13)], cds, exons)
    np.testing.assert_array_equal(codon[0], [-1, -1, -1])
    np.testing.assert_array_equal(strand[0], [2, 2, 2])


def test_agreeing_overlaps_remain_resolved() -> None:
    cds = _cds(
        [
            ("1", 10, 13, "+", "a", 0),
            ("1", 10, 13, "+", "b", 0),
        ]
    )
    exons = _exons([("1", 10, 13, "+", "a"), ("1", 10, 13, "+", "b")])
    codon, strand, _, _ = annotate_cds_windows([("1", 10, 13)], cds, exons)
    np.testing.assert_array_equal(codon[0], [1, 2, 3])
    np.testing.assert_array_equal(strand[0], [1, 1, 1])


def test_splice_annotation_is_two_bases_not_twenty() -> None:
    cds = _cds([("1", 0, 10, "+", "tx", 0), ("1", 20, 30, "+", "tx", 1)])
    exons = _exons([("1", 0, 10, "+", "tx"), ("1", 20, 30, "+", "tx")])
    _, _, splice, strand = annotate_cds_windows([("1", 8, 22)], cds, exons)
    assert splice[0].tolist() == [0, 0, 1, 1] + [0] * 6 + [2, 2, 0, 0]
    assert strand[0].tolist() == [0, 0, 1, 1] + [0] * 6 + [1, 1, 0, 0]
