from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy import sparse, stats
from sklearn.metrics import average_precision_score

from analyze import (
    benjamini_hochberg,
    binary_moments,
    mann_whitney_binary,
    sparse_average_precision,
    sparse_correlations,
)
from analyze_direction_pilot import top_direction_hits


def test_benjamini_hochberg_matches_known_values() -> None:
    observed = benjamini_hochberg(np.array([0.01, 0.04, 0.03, 0.002]))
    np.testing.assert_allclose(observed, [0.02, 0.04, 0.04, 0.008])


def test_sparse_binary_statistics_match_dense_references() -> None:
    dense = np.array(
        [
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.5],
            [0.0, 3.0, 0.0],
            [4.0, 0.0, 1.5],
            [1.0, 2.0, 0.0],
            [5.0, 0.0, 2.5],
        ]
    )
    labels = np.array([False, False, False, True, True, True])
    matrix = sparse.csc_matrix(dense)

    mean1, mean0, difference, _, statistic, pvalue = binary_moments(matrix, labels)
    rank_biserial, mann_whitney_p = mann_whitney_binary(matrix, labels)
    auprc = sparse_average_precision(matrix, labels)
    for feature in range(dense.shape[1]):
        positive = dense[labels, feature]
        negative = dense[~labels, feature]
        welch = stats.ttest_ind(positive, negative, equal_var=False)
        mann_whitney = stats.mannwhitneyu(
            positive,
            negative,
            alternative="two-sided",
            method="asymptotic",
            use_continuity=False,
        )
        np.testing.assert_allclose(mean1[feature], positive.mean())
        np.testing.assert_allclose(mean0[feature], negative.mean())
        np.testing.assert_allclose(
            difference[feature], positive.mean() - negative.mean()
        )
        np.testing.assert_allclose(statistic[feature], welch.statistic)
        np.testing.assert_allclose(pvalue[feature], welch.pvalue)
        np.testing.assert_allclose(
            rank_biserial[feature],
            2 * mann_whitney.statistic / (len(positive) * len(negative)) - 1,
        )
        np.testing.assert_allclose(mann_whitney_p[feature], mann_whitney.pvalue)
        np.testing.assert_allclose(
            auprc[feature], average_precision_score(labels, dense[:, feature])
        )


def test_sparse_correlations_match_scipy_with_zero_ties() -> None:
    dense = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, -1.0],
            [2.0, 0.0, 0.0],
            [-1.0, 1.0, 2.0],
            [0.0, 0.0, 0.0],
            [3.0, -2.0, 1.0],
            [0.0, 0.0, -2.0],
            [-2.0, 3.0, 0.0],
            [1.5, 0.0, 3.0],
            [0.0, -3.0, 0.0],
            [2.5, 1.5, -3.0],
        ]
    )
    effect = np.array([-2.1, -1.6, -1.0, -0.4, 0.1, 0.5, 0.9, 1.2, 1.7, 2.2, 2.8, 3.1])
    pearson, pearson_p, spearman, spearman_p = sparse_correlations(
        sparse.csc_matrix(dense), effect
    )
    for feature in range(dense.shape[1]):
        expected_pearson = stats.pearsonr(dense[:, feature], effect)
        expected_spearman = stats.spearmanr(dense[:, feature], effect)
        np.testing.assert_allclose(pearson[feature], expected_pearson.statistic)
        np.testing.assert_allclose(pearson_p[feature], expected_pearson.pvalue)
        np.testing.assert_allclose(spearman[feature], expected_spearman.statistic)
        np.testing.assert_allclose(
            spearman_p[feature], expected_spearman.pvalue, rtol=1e-12
        )


def test_top_direction_hits_ranks_absolute_effects() -> None:
    frame = pl.DataFrame(
        {
            "outcome": ["direction"] * 4,
            "feature_id": [1, 2, 3, 4],
            "pearson": [0.1, -0.8, 0.6, -0.2],
            "spearman": [-0.9, 0.2, 0.7, 0.1],
        }
    )
    result = top_direction_hits(frame, top_k=2)
    assert result.filter(pl.col("ranking_metric") == "pearson")[
        "feature_id"
    ].to_list() == [2, 3]
    assert result.filter(pl.col("ranking_metric") == "spearman")[
        "feature_id"
    ].to_list() == [1, 3]


def test_benjamini_hochberg_rejects_nonfinite_values() -> None:
    with pytest.raises(AssertionError):
        benjamini_hochberg(np.array([0.1, np.nan]))
