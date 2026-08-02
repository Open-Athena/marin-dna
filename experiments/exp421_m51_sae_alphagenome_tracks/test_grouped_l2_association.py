from __future__ import annotations

import numpy as np
from grouped_l2_association import (
    benjamini_hochberg,
    correlation_pvalues,
    positive_sparse_rank_deviations,
    sparse_pearson,
    sparse_rank_correlation,
)
from scipy import sparse, stats


def test_bh_corrects_the_complete_shape() -> None:
    pvalues = np.array([[0.01, 0.04], [0.03, 0.002]], dtype=np.float64)
    observed = benjamini_hochberg(pvalues)
    np.testing.assert_allclose(observed, [[0.02, 0.04], [0.04, 0.008]])
    assert observed.shape == pvalues.shape


def test_sparse_pearson_matches_scipy_with_implicit_zeros() -> None:
    dense = np.array([[0, 1], [2, 0], [0, 4], [3, 2], [1, 0], [0, 5]], dtype=np.float32)
    outcomes = np.array(
        [[1, 8], [2, 3], [5, 1], [4, 6], [8, 2], [9, 4]], dtype=np.float32
    )
    observed, ss = sparse_pearson(sparse.csr_matrix(dense), outcomes)
    expected = np.array(
        [
            [
                stats.pearsonr(dense[:, feature], outcomes[:, target]).statistic
                for target in range(outcomes.shape[1])
            ]
            for feature in range(dense.shape[1])
        ]
    )
    np.testing.assert_allclose(observed, expected, atol=1e-7)
    assert (ss > 0).all()
    pvalues = correlation_pvalues(observed, len(dense))
    expected_p = np.array(
        [
            [
                stats.pearsonr(dense[:, feature], outcomes[:, target]).pvalue
                for target in range(outcomes.shape[1])
            ]
            for feature in range(dense.shape[1])
        ]
    )
    np.testing.assert_allclose(pvalues, expected_p, atol=2e-7)


def test_sparse_spearman_matches_scipy_with_ties() -> None:
    dense = np.array([[0, 1], [2, 0], [0, 4], [2, 2], [1, 0], [0, 4]], dtype=np.float32)
    outcomes = np.array(
        [[1, 8], [2, 3], [5, 1], [4, 6], [8, 2], [9, 4]], dtype=np.float32
    )
    deviations, feature_ss = positive_sparse_rank_deviations(sparse.csr_matrix(dense))
    observed = sparse_rank_correlation(deviations, feature_ss, outcomes)
    expected = np.array(
        [
            [
                stats.spearmanr(dense[:, feature], outcomes[:, target]).statistic
                for target in range(outcomes.shape[1])
            ]
            for feature in range(dense.shape[1])
        ]
    )
    np.testing.assert_allclose(observed, expected, atol=1e-7)


def test_log1p_changes_pearson_but_not_sparse_ranks() -> None:
    dense = np.array([[0], [1], [3], [10], [0], [2]], dtype=np.float32)
    outcomes = np.arange(6, dtype=np.float32)[:, None]
    raw_r, _ = sparse_pearson(sparse.csr_matrix(dense), outcomes)
    log_r, _ = sparse_pearson(sparse.csr_matrix(np.log1p(dense)), outcomes)
    assert not np.isclose(raw_r.item(), log_r.item())
    raw_d, raw_ss = positive_sparse_rank_deviations(sparse.csr_matrix(dense))
    log_d, log_ss = positive_sparse_rank_deviations(sparse.csr_matrix(np.log1p(dense)))
    np.testing.assert_array_equal(raw_d.toarray(), log_d.toarray())
    np.testing.assert_array_equal(raw_ss, log_ss)
