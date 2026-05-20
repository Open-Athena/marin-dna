"""Tests for ``marin_dna.pipelines.conservation.calibration``."""

import numpy as np
import pytest

from marin_dna.pipelines.conservation.calibration import calibrate_to_match_proportion
from marin_dna.pipelines.conservation.histogram import PhylopHistogram


@pytest.fixture
def edges() -> np.ndarray:
    return np.linspace(-20.0, 20.0, 1001)


def _hist(values: np.ndarray, edges: np.ndarray, n_nan: int = 0) -> PhylopHistogram:
    counts, _ = np.histogram(np.clip(values, edges[0], edges[-1]), bins=edges)
    return PhylopHistogram(counts=counts.astype(np.int64), edges=edges, n_nan=n_nan)


def test_calibrate_identity(edges: np.ndarray) -> None:
    """Same histogram on both sides → matched threshold equals the reference."""
    rng = np.random.default_rng(0)
    values = rng.normal(0.0, 2.0, size=200_000)
    hist = _hist(values, edges)
    out = calibrate_to_match_proportion(hist, hist, ref_threshold=2.27)
    # Must be within one bin's worth (0.04) of the reference
    assert abs(out["target"]["threshold"] - 2.27) < 0.05


def test_calibrate_known_shift(edges: np.ndarray) -> None:
    """Reference is N(0, 2); target is N(1, 2). Calibrated threshold should be
    ~ 1 above the reference threshold (since target distribution is shifted +1)."""
    rng = np.random.default_rng(0)
    n = 500_000
    ref_vals = rng.normal(0.0, 2.0, size=n)
    tgt_vals = rng.normal(1.0, 2.0, size=n)
    ref_hist = _hist(ref_vals, edges)
    tgt_hist = _hist(tgt_vals, edges)

    out = calibrate_to_match_proportion(tgt_hist, ref_hist, ref_threshold=2.27)
    assert abs(out["target"]["threshold"] - (2.27 + 1.0)) < 0.1


def test_calibrate_proportion_invariant_to_total_size(edges: np.ndarray) -> None:
    """Doubling the target's total non-NaN bases (with the same distribution
    shape) should NOT change the calibrated threshold — proportion-based
    calibration is invariant to total coverage. Count-based would scale."""
    rng = np.random.default_rng(0)
    ref_vals = rng.normal(0.0, 2.0, size=200_000)
    ref_hist = _hist(ref_vals, edges)

    tgt_vals_small = rng.normal(0.0, 2.0, size=200_000)
    tgt_vals_big = rng.normal(0.0, 2.0, size=400_000)
    tgt_hist_small = _hist(tgt_vals_small, edges)
    tgt_hist_big = _hist(tgt_vals_big, edges)

    out_small = calibrate_to_match_proportion(
        tgt_hist_small, ref_hist, ref_threshold=2.27
    )
    out_big = calibrate_to_match_proportion(tgt_hist_big, ref_hist, ref_threshold=2.27)
    # Same distribution → same threshold within sampling noise
    assert abs(out_small["target"]["threshold"] - out_big["target"]["threshold"]) < 0.1


def test_calibrate_n_nan_does_not_affect_proportion(edges: np.ndarray) -> None:
    """``n_nan`` is tracked separately from ``counts``; it should not enter
    the proportion calculation. Adding NaN bases to either side must leave
    the calibrated threshold unchanged."""
    rng = np.random.default_rng(0)
    ref_vals = rng.normal(0.0, 2.0, size=200_000)
    tgt_vals = rng.normal(1.0, 2.0, size=200_000)

    out_no_nan = calibrate_to_match_proportion(
        _hist(tgt_vals, edges, n_nan=0),
        _hist(ref_vals, edges, n_nan=0),
        ref_threshold=2.27,
    )
    out_with_nan = calibrate_to_match_proportion(
        _hist(tgt_vals, edges, n_nan=10_000_000),
        _hist(ref_vals, edges, n_nan=5_000_000),
        ref_threshold=2.27,
    )
    # Identical thresholds (NaN counts not in numerator or denominator)
    assert out_no_nan["target"]["threshold"] == out_with_nan["target"]["threshold"]


def test_calibration_dict_shape(edges: np.ndarray) -> None:
    """Returned dict has the expected structure for downstream JSON output."""
    rng = np.random.default_rng(0)
    hist = _hist(rng.normal(0, 2, size=100_000), edges, n_nan=42)
    out = calibrate_to_match_proportion(
        hist,
        hist,
        ref_threshold=2.0,
        target_name="phyloP_447m",
        ref_name="phyloP_241m",
    )
    assert set(out.keys()) == {"phyloP_241m", "phyloP_447m", "hist_meta"}
    assert set(out["phyloP_241m"]) == {
        "threshold",
        "count",
        "proportion",
        "total_bases",
        "n_nan",
    }
    assert set(out["phyloP_447m"]) == {
        "threshold",
        "count",
        "count_target",
        "proportion",
        "abs_relative_error",
        "total_bases",
        "n_nan",
    }
    assert set(out["hist_meta"]) == {"n_bins", "min", "max", "bin_width"}
    assert out["phyloP_447m"]["abs_relative_error"] < 0.01
    assert 0.0 <= out["phyloP_241m"]["proportion"] <= 1.0
    assert 0.0 <= out["phyloP_447m"]["proportion"] <= 1.0
