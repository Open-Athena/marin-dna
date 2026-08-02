from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.sparse import csr_matrix

from association_common import bh_adjust, comparison_metrics
from extract_common import D_SAE


def test_sparse_metrics_match_reference_statistics() -> None:
    dense = np.zeros((10, D_SAE), dtype=np.float32)
    dense[:, 0] = [0, 1, 1, 3, 0, 0, 1, 2, 2, 0]
    dense[:, 1] = [5, 0, 2, 0, 1, 0, 0, 0, 1, 1]
    dense[:, 2] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    observed = comparison_metrics(
        csr_matrix(dense),
        np.arange(5),
        np.arange(5, 10),
        minimum_nonzero_support=1,
    ).sort("feature_id")
    assert observed["feature_id"].to_list() == [0, 1, 2]
    expected_ap = [(0.6, 0.5285714285714285), (0.72, 0.42388888888888887), (0.5, 0.5)]
    for feature_id in range(3):
        row = observed.row(feature_id, named=True)
        positive = dense[:5, feature_id]
        negative = dense[5:, feature_id]
        welch = stats.ttest_ind(positive, negative, equal_var=False)
        mann = stats.mannwhitneyu(
            positive, negative, alternative="two-sided", method="asymptotic"
        )
        if feature_id < 2:
            assert np.isclose(row["welch_p"], welch.pvalue, equal_nan=False)
        else:
            assert np.isnan(welch.pvalue) and row["welch_p"] == 1.0
        assert np.isclose(row["u_statistic"], mann.statistic)
        if feature_id < 2:
            assert np.isclose(row["mann_whitney_p"], mann.pvalue)
        else:
            assert np.isnan(mann.pvalue) and row["mann_whitney_p"] == 1.0
        assert np.isclose(row["auprc"], expected_ap[feature_id][0])
        assert np.isclose(row["auprc_negated"], expected_ap[feature_id][1])


def test_bh_adjust_is_monotone_in_rank() -> None:
    values = np.array([0.04, 0.001, 0.03, 0.2])
    adjusted = bh_adjust(values)
    order = np.argsort(values)
    assert np.all(adjusted[order][:-1] <= adjusted[order][1:])
    assert np.all(adjusted >= values)
