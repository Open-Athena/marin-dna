from __future__ import annotations

import numpy as np

from prediction_targets import (
    bootstrap_label_auprc,
    bootstrap_subset_auprc,
    multiclass_auprc,
    select_matched_label_feature,
)


def test_label_bootstrap_preserves_fixed_prevalence_and_is_deterministic() -> None:
    groups = np.repeat(np.arange(4), 10)
    labels = np.tile(np.asarray([1] + [0] * 9, dtype=np.int8), 4)
    scores = labels.astype(np.float64)

    first = bootstrap_label_auprc(labels, scores, groups, seed=1, samples=50)
    second = bootstrap_label_auprc(labels, scores, groups, seed=1, samples=50)

    assert first == second
    assert first["auprc"] == 1.0
    assert first["prevalence"] == 0.1


def test_matched_label_selection_rejects_validation_reversal() -> None:
    groups = np.repeat(np.arange(8), 10)
    labels = np.tile(np.asarray([1] + [0] * 9, dtype=np.int8), 8)
    splits = np.repeat(np.asarray(["discovery"] * 4 + ["validation"] * 4), 10)
    matrix = np.zeros((80, 3), dtype=np.float32)
    values = [5.0, 6.0, 4.0, 7.0]
    stable = [4.0, 5.0, 3.0, 6.0]
    for group in range(8):
        rows = groups == group
        positive = np.flatnonzero(rows & (labels == 1))[0]
        if group < 4:
            matrix[positive, 0] = values[group]
        else:
            matrix[positive, 0] = -values[group - 4]
            matrix[rows & (labels == 0), 0] = 10
        matrix[positive, 1] = stable[group % 4]

    selection, candidates = select_matched_label_feature(
        matrix,
        labels,
        groups,
        splits,
        top_k=2,
        min_nonzero_groups=2,
    )

    assert selection.feature_id == 1
    assert selection.validation_direction_consistent
    assert any(
        row["feature_id"] == 0 and not row["validation_direction_consistent"]
        for row in candidates
    )


def test_multiclass_auprc_and_bootstrap() -> None:
    classes = ("a", "b")
    truth = np.repeat(np.asarray(classes), 100)
    groups = np.repeat(np.arange(20), 10)
    scores = np.column_stack((truth == "a", truth == "b")).astype(np.float64)

    per_class, macro = multiclass_auprc(truth, scores, classes)
    result = bootstrap_subset_auprc(
        truth,
        scores,
        groups,
        classes,
        seed=2,
        samples=50,
    )

    assert per_class == {"a": 1.0, "b": 1.0}
    assert macro == 1.0
    assert result["macro_auprc"] == 1.0
    assert result["macro_random_baseline"] == 0.5
