from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import mannwhitneyu, ttest_ind
from sklearn.metrics import average_precision_score

from analyze import bh_adjust, feature_statistics, sparse_view


def test_bh_adjust_matches_known_example() -> None:
    observed = bh_adjust(np.array([0.01, 0.04, 0.03, 0.002]))
    np.testing.assert_allclose(observed, [0.02, 0.04, 0.04, 0.008])


def test_zero_aware_statistics_match_dense_references() -> None:
    dense = np.array([0, 1, 2, 0, 0, 3, 0, 4, 0], dtype=float)
    class_codes = np.repeat(np.arange(3), 3)
    nonzero = dense > 0
    observed = feature_statistics(
        dense[nonzero], class_codes[nonzero], np.array([3, 3, 3])
    )

    for class_index in range(3):
        labels = class_codes == class_index
        welch = ttest_ind(dense[labels], dense[~labels], equal_var=False)
        mwu = mannwhitneyu(
            dense[labels],
            dense[~labels],
            alternative="two-sided",
            method="asymptotic",
            use_continuity=False,
        )
        ap = average_precision_score(labels, dense)
        assert np.isclose(observed["welch_t"][class_index], welch.statistic)
        assert np.isclose(observed["welch_p"][class_index], welch.pvalue)
        assert np.isclose(observed["mwu_u"][class_index], mwu.statistic)
        assert np.isclose(observed["mwu_p"][class_index], mwu.pvalue)
        assert np.isclose(observed["auprc"][class_index], ap)
        expected_rb = 2 * mwu.statistic / (labels.sum() * (~labels).sum()) - 1
        assert np.isclose(observed["rank_biserial"][class_index], expected_rb)


def test_sparse_views_include_implicit_zeros() -> None:
    forward = pl.DataFrame(
        {
            "panel_row": [0, 1],
            "feature_id": [3, 4],
            "activation": [2.0, 6.0],
        },
        schema={
            "panel_row": pl.UInt32,
            "feature_id": pl.UInt32,
            "activation": pl.Float32,
        },
    )
    reverse = pl.DataFrame(
        {
            "panel_row": [0, 2],
            "feature_id": [3, 4],
            "activation": [4.0, 8.0],
        },
        schema={
            "panel_row": pl.UInt32,
            "feature_id": pl.UInt32,
            "activation": pl.Float32,
        },
    )
    mean = sparse_view(forward, reverse, "same_id_mean").sort("panel_row", "feature_id")
    maximum = sparse_view(forward, reverse, "same_id_max").sort(
        "panel_row", "feature_id"
    )
    assert mean["activation"].to_list() == [3.0, 3.0, 4.0]
    assert maximum["activation"].to_list() == [4.0, 6.0, 8.0]


def test_constant_dense_feature_is_a_valid_null() -> None:
    observed = feature_statistics(
        np.ones(6), np.repeat(np.arange(3), 2), np.array([2, 2, 2])
    )
    np.testing.assert_allclose(observed["welch_p"], 1.0)
    np.testing.assert_allclose(observed["mwu_p"], 1.0)
    np.testing.assert_allclose(observed["rank_biserial"], 0.0)
