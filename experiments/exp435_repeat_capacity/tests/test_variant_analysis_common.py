from __future__ import annotations

import numpy as np

from variant_analysis_common import (
    average_precision,
    binary_feature_metrics,
    with_fdr,
)


def test_average_precision_groups_ties() -> None:
    scores = np.array([1.0, 1.0, 0.0, 0.0])
    labels = np.array([True, False, True, False])
    assert average_precision(scores, labels) == 0.5
    assert average_precision(np.ones(4), labels) == 0.5


def test_signed_binary_metrics_and_fdr_call_concordant_direction() -> None:
    positive = np.column_stack((np.linspace(2, 4, 20), np.zeros(20)))
    negative = np.column_stack((np.linspace(-4, -2, 20), np.zeros(20)))
    positive[-1, 1] = 1
    negative[-1, 1] = 1
    matrix = np.concatenate((positive, negative)).astype(np.float32)
    frame = binary_feature_metrics(
        matrix,
        np.array([7, 11]),
        np.arange(20),
        np.arange(20, 40),
        minimum_nonzero_support=2,
    )
    assert frame["feature_id"].to_list() == [7, 11]
    first = frame.row(0, named=True)
    assert first["mean_difference"] > 0
    assert first["rank_biserial"] > 0
    assert first["auprc"] == 1.0
    called = with_fdr(frame, response="delta")
    assert called.filter(called["feature_id"] == 7)["concordant_association"].item()
    assert not called.filter(called["feature_id"] == 7)[
        "positive_mutation_association"
    ].item()


def test_activation_support_is_combined_not_group_specific() -> None:
    matrix = np.array([[1.0], [2.0], [0.0], [0.0]], dtype=np.float32)
    frame = binary_feature_metrics(
        matrix,
        np.array([5]),
        np.array([0, 1]),
        np.array([2, 3]),
        minimum_nonzero_support=2,
    )
    assert frame.height == 1
    assert frame["positive_nonzero"].item() == 2
    assert frame["negative_nonzero"].item() == 0
