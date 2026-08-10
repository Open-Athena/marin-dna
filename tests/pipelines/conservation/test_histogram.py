"""Tests for ``marin_dna.pipelines.conservation.histogram``."""

from itertools import pairwise

import numpy as np
import pytest

from marin_dna.pipelines.conservation.histogram import PhylopHistogram


@pytest.fixture
def edges() -> np.ndarray:
    return np.linspace(-20.0, 20.0, 1001)  # 1000 bins, width 0.04


@pytest.fixture
def synthetic_values() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(loc=0.0, scale=2.0, size=100_000).astype(np.float64)


def _hist_from_values(values: np.ndarray, edges: np.ndarray) -> PhylopHistogram:
    counts, _ = np.histogram(np.clip(values, edges[0], edges[-1]), bins=edges)
    return PhylopHistogram(counts=counts.astype(np.int64), edges=edges, n_nan=0)


def test_count_ge_matches_naive(
    edges: np.ndarray, synthetic_values: np.ndarray
) -> None:
    # Linear interpolation introduces an error bounded by the count density
    # times the bin width — for a Gaussian on 100k samples that's well
    # under 200.
    hist = _hist_from_values(synthetic_values, edges)
    for t in (-5.0, -1.0, 0.0, 1.5, 3.0, 5.0):
        assert abs(hist.count_ge(t) - int((synthetic_values >= t).sum())) < 200


def test_count_ge_extremes(edges: np.ndarray) -> None:
    hist = _hist_from_values(np.array([0.0, 1.0, 2.0, 3.0]), edges)
    # Below the histogram range: all 4 are above
    assert hist.count_ge(-1000.0) == 4
    # Above the histogram range: zero
    assert hist.count_ge(1000.0) == 0


def test_threshold_for_count_inverse(
    edges: np.ndarray, synthetic_values: np.ndarray
) -> None:
    hist = _hist_from_values(synthetic_values, edges)
    total = hist.total()
    # Round-trip a sample of target counts
    for target in (10, 100, 1_000, 10_000, 50_000, total - 100):
        t = hist.threshold_for_count(target)
        # Re-counting should give a count near the target (within bin precision).
        recovered = hist.count_ge(t)
        # Within ~1% or 50 bases, whichever is larger.
        tolerance = max(50, target // 100)
        assert abs(recovered - target) < tolerance, (
            f"target={target}, threshold={t}, recovered={recovered}"
        )


def test_threshold_for_count_monotone(
    edges: np.ndarray, synthetic_values: np.ndarray
) -> None:
    """As the target count increases, the matching threshold must not increase."""
    hist = _hist_from_values(synthetic_values, edges)
    targets = [100, 1_000, 10_000, 50_000]
    thresholds = [hist.threshold_for_count(t) for t in targets]
    for a, b in pairwise(thresholds):
        assert a >= b, f"non-monotone: targets {targets}, thresholds {thresholds}"


def test_threshold_for_count_extremes(
    edges: np.ndarray, synthetic_values: np.ndarray
) -> None:
    hist = _hist_from_values(synthetic_values, edges)
    # target_count == 0 → threshold at the top of the range
    assert hist.threshold_for_count(0) == edges[-1]
    # target_count >= total → threshold at the bottom
    assert hist.threshold_for_count(hist.total() + 100) == edges[0]


def test_add_histograms(edges: np.ndarray) -> None:
    h1 = _hist_from_values(np.array([1.0, 2.0, 3.0]), edges)
    h1.n_nan = 7
    h2 = _hist_from_values(np.array([1.0, 4.0]), edges)
    h2.n_nan = 3
    h12 = h1 + h2
    assert h12.n_nan == 10
    assert h12.total() == 5
    assert h12.count_ge(2.5) == 2  # values 3.0 and 4.0


def test_add_histograms_edge_mismatch(edges: np.ndarray) -> None:
    h1 = _hist_from_values(np.array([1.0]), edges)
    other_edges = np.linspace(-10.0, 10.0, 501)
    h2 = _hist_from_values(np.array([1.0]), other_edges)
    with pytest.raises(AssertionError, match="same edges"):
        _ = h1 + h2


def test_save_load_roundtrip(edges: np.ndarray, tmp_path) -> None:
    h = _hist_from_values(np.array([0.0, 1.0, 2.0, 3.0, 4.0]), edges)
    h.n_nan = 42
    p = tmp_path / "h.npz"
    h.save(p)
    loaded = PhylopHistogram.load(p)
    assert np.array_equal(loaded.counts, h.counts)
    assert np.array_equal(loaded.edges, h.edges)
    assert loaded.n_nan == h.n_nan
    assert loaded.total() == h.total()
    assert loaded.count_ge(2.5) == h.count_ge(2.5)
