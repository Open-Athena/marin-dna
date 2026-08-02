from __future__ import annotations

import numpy as np
from scipy import sparse, stats
from sklearn.metrics import average_precision_score

import complete_family_association as module
from complete_family_association import (
    benjamini_hochberg,
    directional_average_precision,
    kruskal_omnibus,
    mann_whitney_one_vs_rest,
    sparse_rank_deviations,
    welch_one_vs_rest,
)


def synthetic() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.array(
        [
            [0, -2],
            [1, 0],
            [2, 3],
            [0, 1],
            [4, 0],
            [5, -1],
        ],
        dtype=np.float32,
    )
    encoded = np.array([0, 0, 0, 1, 1, 1], dtype=np.int16)
    indicator = np.zeros((6, 2), dtype=np.float64)
    indicator[np.arange(6), encoded] = 1
    return values, encoded, indicator


def test_bh_preserves_complete_family_shape() -> None:
    values = np.array([[0.01, 0.04], [0.03, 0.002]])
    np.testing.assert_allclose(
        benjamini_hochberg(values), [[0.02, 0.04], [0.04, 0.008]]
    )


def test_vectorized_welch_matches_scipy(monkeypatch) -> None:
    values, _, indicator = synthetic()
    monkeypatch.setattr(module, "EXPECTED_ROWS", 6)
    monkeypatch.setattr(module, "EXPECTED_CLASSES", 2)
    monkeypatch.setattr(module, "EXPECTED_PER_CLASS", 3)
    pvalue, difference, _ = welch_one_vs_rest(sparse.csr_matrix(values), indicator)
    for feature in range(values.shape[1]):
        expected = stats.ttest_ind(
            values[:3, feature], values[3:, feature], equal_var=False
        )
        np.testing.assert_allclose(pvalue[feature, 0], expected.pvalue, atol=2e-6)
        np.testing.assert_allclose(
            difference[feature, 0],
            values[:3, feature].mean() - values[3:, feature].mean(),
        )


def test_rank_sum_mwu_and_kruskal_match_scipy(monkeypatch) -> None:
    values, _, indicator = synthetic()
    monkeypatch.setattr(module, "EXPECTED_ROWS", 6)
    monkeypatch.setattr(module, "EXPECTED_CLASSES", 2)
    monkeypatch.setattr(module, "EXPECTED_PER_CLASS", 3)
    deviations, zero_ranks, ties = sparse_rank_deviations(sparse.csr_matrix(values))
    pvalue, _, rank_sum = mann_whitney_one_vs_rest(
        deviations, zero_ranks, ties, indicator
    )
    for feature in range(values.shape[1]):
        expected = stats.mannwhitneyu(
            values[:3, feature],
            values[3:, feature],
            alternative="two-sided",
            method="asymptotic",
            use_continuity=False,
        )
        np.testing.assert_allclose(pvalue[feature, 0], expected.pvalue, atol=1e-12)
    kruskal_p, kruskal_h, valid = kruskal_omnibus(rank_sum, ties)
    assert valid.all()
    for feature in range(values.shape[1]):
        expected = stats.kruskal(values[:3, feature], values[3:, feature])
        np.testing.assert_allclose(kruskal_h[feature], expected.statistic, atol=2e-6)
        np.testing.assert_allclose(kruskal_p[feature], expected.pvalue, atol=2e-6)


def test_directional_average_precision_matches_sklearn(monkeypatch) -> None:
    values, encoded, _ = synthetic()
    monkeypatch.setattr(module, "EXPECTED_CLASSES", 2)
    direction = np.array([[1, -1], [-1, 1]], dtype=np.float64)
    observed = directional_average_precision(
        sparse.csr_matrix(values), encoded, direction
    )
    for feature in range(values.shape[1]):
        for target in range(2):
            sign = 1 if direction[feature, target] >= 0 else -1
            expected = average_precision_score(
                encoded == target, sign * values[:, feature]
            )
            np.testing.assert_allclose(observed[feature, target], expected, atol=1e-12)
