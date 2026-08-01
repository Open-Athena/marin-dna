from __future__ import annotations

import numpy as np

from subset_analysis import (
    categorical_rate_scores,
    cluster_bootstrap_binary,
    conditional_auc,
    select_feature,
)


def test_select_feature_uses_validation_to_reject_discovery_reversal() -> None:
    target = np.asarray([1, 1, 0, 0] * 2, dtype=np.bool_)
    splits = np.asarray(["discovery"] * 4 + ["validation"] * 4)
    task_rows = np.ones(8, dtype=np.bool_)
    matrix = np.asarray(
        [
            [5, 4, 1],
            [6, 5, 1],
            [1, 2, 1],
            [2, 3, 1],
            [1, 4, 1],
            [2, 5, 1],
            [5, 2, 1],
            [6, 3, 1],
        ],
        dtype=np.float32,
    )

    selection, candidates = select_feature(
        matrix,
        target,
        splits,
        task_rows,
        top_k=2,
        min_nonzero=2,
    )

    assert selection.feature_id == 1
    assert selection.direction == 1
    assert selection.validation_direction_consistent
    assert any(
        row["feature_id"] == 0 and not row["validation_direction_consistent"]
        for row in candidates
    )


def test_conditional_auc_weights_comparable_pairs_only() -> None:
    target = np.asarray([1, 0, 1, 0, 1], dtype=np.bool_)
    scores = np.asarray([3.0, 1.0, 0.0, 2.0, 9.0])
    strata = np.asarray(["a", "a", "b", "b", "c"])

    result = conditional_auc(target, scores, strata)

    assert result["auc"] == 0.5
    assert result["pairs"] == 2
    assert result["covered_rows"] == 4
    assert result["covered_fraction"] == 0.8


def test_categorical_rates_do_not_use_test_targets() -> None:
    train_target = np.asarray([1, 1, 0, 0], dtype=np.bool_)
    train_strata = np.asarray(["a", "a", "a", "b"])
    test_strata = np.asarray(["a", "b", "unseen"])

    scores = categorical_rate_scores(train_target, train_strata, test_strata)

    np.testing.assert_allclose(scores, [3 / 5, 1 / 3, 1 / 2])


def test_cluster_bootstrap_binary_is_deterministic() -> None:
    target = np.asarray([1, 1, 0, 0], dtype=np.bool_)
    scores = np.asarray([4.0, 3.0, 2.0, 1.0])
    groups = np.asarray(["p1", "p2", "n1", "n2"])

    first = cluster_bootstrap_binary(target, scores, groups, seed=17, samples=100)
    second = cluster_bootstrap_binary(target, scores, groups, seed=17, samples=100)

    assert first == second
    assert first["auc"] == 1.0
    assert first["mean_difference"] == 2.0
