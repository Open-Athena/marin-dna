"""Tests for ``marin_dna.pipelines.conservation.scoring``."""

import math

import numpy as np
import polars as pl
import pyBigWig
import pytest

from marin_dna.pipelines.conservation.scoring import score_windows


@pytest.fixture
def synthetic_bigwig(tmp_path):
    """Build a tiny ``chr1`` bigWig with three regions:
    - [0, 30):    value 2.0   (above any threshold below 2.0)
    - [30, 60):   no value    (NaN — gap in bigWig)
    - [60, 100):  value -1.0  (below any threshold above -1.0)
    """
    bw_path = tmp_path / "test.bw"
    bw = pyBigWig.open(str(bw_path), "w")
    bw.addHeader([("chr1", 100)])
    bw.addEntries(
        ["chr1"] * 30,
        list(range(30)),
        ends=list(range(1, 31)),
        values=[2.0] * 30,
    )
    bw.addEntries(
        ["chr1"] * 40,
        list(range(60, 100)),
        ends=list(range(61, 101)),
        values=[-1.0] * 40,
    )
    bw.close()
    return bw_path


def test_score_windows_basic(synthetic_bigwig) -> None:
    """Single 100 bp window covering the whole synthetic chromosome."""
    windows = pl.DataFrame({"chrom": ["1"], "start": [0], "end": [100], "name": ["w1"]})
    result = score_windows(synthetic_bigwig, windows, threshold=1.0)
    row = result.row(0, named=True)
    assert row["conserved_bases"] == 30
    assert math.isclose(row["proportion_conserved"], 0.30, abs_tol=1e-5)
    assert row["n_valid_bases"] == 70
    # mean over 70 valid bases: (30*2 + 40*-1) / 70 = 20/70 ≈ 0.2857
    assert math.isclose(row["mean_phylop"], 20 / 70, abs_tol=1e-3)


def test_score_windows_nan_counted_as_zero(synthetic_bigwig) -> None:
    """A window of pure NaN scores 0 conserved bases, n_valid 0."""
    windows = pl.DataFrame(
        {"chrom": ["1"], "start": [30], "end": [60], "name": ["w_nan"]}
    )
    result = score_windows(synthetic_bigwig, windows, threshold=0.0)
    row = result.row(0, named=True)
    assert row["conserved_bases"] == 0
    assert math.isclose(row["proportion_conserved"], 0.0, abs_tol=1e-5)
    assert row["n_valid_bases"] == 0
    assert math.isnan(row["mean_phylop"])


def test_score_windows_threshold_inclusive(synthetic_bigwig) -> None:
    """A value exactly at threshold is conserved (>= semantics)."""
    windows = pl.DataFrame(
        {"chrom": ["1"], "start": [0], "end": [30], "name": ["w_above"]}
    )
    result = score_windows(synthetic_bigwig, windows, threshold=2.0)
    row = result.row(0, named=True)
    assert row["conserved_bases"] == 30


def test_score_windows_chrom_prefix(synthetic_bigwig) -> None:
    """Both ``"1"`` and ``"chr1"`` chrom labels resolve to the same bigWig chrom."""
    bare = score_windows(
        synthetic_bigwig,
        pl.DataFrame({"chrom": ["1"], "start": [0], "end": [30], "name": ["w"]}),
        threshold=1.0,
    )
    chr_prefixed = score_windows(
        synthetic_bigwig,
        pl.DataFrame({"chrom": ["chr1"], "start": [0], "end": [30], "name": ["w"]}),
        threshold=1.0,
    )
    assert bare.row(0, named=True)["conserved_bases"] == 30
    assert chr_prefixed.row(0, named=True)["conserved_bases"] == 30


def test_score_windows_unknown_chrom(synthetic_bigwig) -> None:
    """A chrom not in the bigWig produces zeros and NaN mean (instead of crashing)."""
    windows = pl.DataFrame(
        {"chrom": ["99"], "start": [0], "end": [30], "name": ["w_missing"]}
    )
    result = score_windows(synthetic_bigwig, windows, threshold=1.0)
    row = result.row(0, named=True)
    assert row["conserved_bases"] == 0
    assert row["n_valid_bases"] == 0
    assert math.isnan(row["mean_phylop"])


def test_score_windows_dtypes(synthetic_bigwig) -> None:
    """Schema is the contract used by downstream Parquet / filter rules."""
    windows = pl.DataFrame({"chrom": ["1"], "start": [0], "end": [100], "name": ["w1"]})
    result = score_windows(synthetic_bigwig, windows, threshold=1.0)
    assert result.schema["conserved_bases"] == pl.Int32
    assert result.schema["proportion_conserved"] == pl.Float32
    assert result.schema["mean_phylop"] == pl.Float32
    assert result.schema["n_valid_bases"] == pl.Int32


def test_score_windows_proportion_consistent_with_count(synthetic_bigwig) -> None:
    """``proportion_conserved == conserved_bases / (end - start)`` exactly."""
    rng = np.random.default_rng(0)
    rows = [
        {
            "chrom": "1",
            "start": int(s),
            "end": int(s + 50),
            "name": f"w{i}",
        }
        for i, s in enumerate(rng.integers(0, 50, size=10))
    ]
    windows = pl.DataFrame(rows)
    result = score_windows(synthetic_bigwig, windows, threshold=1.0)
    for row in result.iter_rows(named=True):
        size = row["end"] - row["start"]
        assert math.isclose(
            row["proportion_conserved"], row["conserved_bases"] / size, abs_tol=1e-5
        )
