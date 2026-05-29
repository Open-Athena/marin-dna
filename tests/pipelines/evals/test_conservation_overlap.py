"""Tests for ``marin_dna.pipelines.evals.conservation_overlap``.

Focus: the 1-based ``pos`` -> 0-based ``[pos-1, pos)`` boundary (the one
coordinate conversion in the module) and the NaN-as-non-conserved
convention inherited from ``score_positions``.
"""

import math

import polars as pl
import pyBigWig
import pytest

from marin_dna.pipelines.evals.conservation_overlap import (
    add_base_conservation,
    add_window_overlap,
    overlap_summary,
)


@pytest.fixture
def synthetic_bigwig(tmp_path):
    """``chr1`` bigWig: [0,30) = 2.0, [30,60) = NaN gap, [60,100) = -1.0."""
    bw_path = tmp_path / "test.bw"
    bw = pyBigWig.open(str(bw_path), "w")
    bw.addHeader([("chr1", 100)])
    bw.addEntries(
        ["chr1"] * 30, list(range(0, 30)), ends=list(range(1, 31)), values=[2.0] * 30
    )
    bw.addEntries(
        ["chr1"] * 40,
        list(range(60, 100)),
        ends=list(range(61, 101)),
        values=[-1.0] * 40,
    )
    bw.close()
    return bw_path


# --- add_window_overlap: coordinate boundary ---


def test_window_overlap_boundaries() -> None:
    """A variant base [pos-1, pos) is inside [start, end) iff start <= pos-1 < end.

    Region [10, 20): 1-based pos 11..20 are inside (bases 10..19); pos 10
    (base 9) and pos 21 (base 20) are outside.
    """
    regions = pl.DataFrame({"chrom": ["1"], "start": [10], "end": [20]})
    variants = pl.DataFrame({"chrom": ["1"] * 5, "pos": [10, 11, 15, 20, 21]})
    out = add_window_overlap(variants, regions)
    assert out["in_conserved_window"].to_list() == [False, True, True, True, False]


def test_window_overlap_multi_region_and_order() -> None:
    """Multiple regions; output preserves input row order."""
    regions = pl.DataFrame({"chrom": ["1", "1"], "start": [10, 50], "end": [20, 60]})
    variants = pl.DataFrame({"chrom": ["1", "1", "1", "1"], "pos": [55, 5, 11, 100]})
    out = add_window_overlap(variants, regions)
    # pos 55 -> base 54 in [50,60); pos 5 -> base 4 outside; pos 11 -> in
    # [10,20); pos 100 -> base 99 outside.
    assert out["in_conserved_window"].to_list() == [True, False, True, False]


def test_window_overlap_wrong_chrom() -> None:
    """A variant on a chrom with no regions is not flagged."""
    regions = pl.DataFrame({"chrom": ["1"], "start": [10], "end": [20]})
    variants = pl.DataFrame({"chrom": ["2"], "pos": [15]})
    out = add_window_overlap(variants, regions)
    assert out["in_conserved_window"].to_list() == [False]


def test_window_overlap_empty_regions() -> None:
    """No regions -> nothing flagged, original rows preserved."""
    regions = pl.DataFrame(
        {"chrom": [], "start": [], "end": []},
        schema={"chrom": pl.Utf8, "start": pl.Int64, "end": pl.Int64},
    )
    variants = pl.DataFrame({"chrom": ["1", "1"], "pos": [15, 99]})
    out = add_window_overlap(variants, regions)
    assert out["in_conserved_window"].to_list() == [False, False]
    assert out.height == 2


# --- add_base_conservation: 1-based pos lookup ---


def test_base_conservation_lookup(synthetic_bigwig) -> None:
    """1-based pos -> base [pos-1, pos): pos 1 -> base 0 (val 2.0), pos 31 ->
    base 30 (NaN gap), pos 61 -> base 60 (val -1.0)."""
    variants = pl.DataFrame({"chrom": ["1", "1", "1"], "pos": [1, 31, 61]})
    out = add_base_conservation(variants, synthetic_bigwig, threshold=1.0)
    vals = out["base_phylop"].to_list()
    assert math.isclose(vals[0], 2.0, abs_tol=1e-6)
    assert vals[1] is None or math.isnan(vals[1])
    assert math.isclose(vals[2], -1.0, abs_tol=1e-6)
    assert out["base_conserved"].to_list() == [True, False, False]


def test_base_conservation_threshold_inclusive(synthetic_bigwig) -> None:
    variants = pl.DataFrame({"chrom": ["1"], "pos": [1]})  # base 0 -> 2.0
    out = add_base_conservation(variants, synthetic_bigwig, threshold=2.0)
    assert out["base_conserved"].to_list() == [True]


# --- overlap_summary ---


def test_overlap_summary_fractions() -> None:
    df = pl.DataFrame(
        {
            "label": [True, True, True, True, False, False],
            "consequence_group": ["a", "a", "b", "b", "a", "a"],
            "flag": [True, True, True, False, False, False],
        }
    )
    overall = overlap_summary(df, "flag").sort("label")
    # negatives: 0/2 = 0.0 ; positives: 3/4 = 0.75
    neg = overall.filter(~pl.col("label")).row(0, named=True)
    pos = overall.filter(pl.col("label")).row(0, named=True)
    assert neg["n"] == 2 and neg["n_in"] == 0 and math.isclose(neg["frac_in"], 0.0)
    assert pos["n"] == 4 and pos["n_in"] == 3 and math.isclose(pos["frac_in"], 0.75)

    by_group = overlap_summary(df, "flag", by=("consequence_group",))
    pos_b = by_group.filter(pl.col("label") & (pl.col("consequence_group") == "b")).row(
        0, named=True
    )
    assert (
        pos_b["n"] == 2 and pos_b["n_in"] == 1 and math.isclose(pos_b["frac_in"], 0.5)
    )
