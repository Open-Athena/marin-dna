from __future__ import annotations

import numpy as np
import polars as pl
from scipy import sparse

from analyze import (
    Activations,
    correlations_to_all,
    cross_feature_matches,
    make_views,
    paired_state_summary,
    paired_states,
    same_id_correlations,
    select_feature,
)


def test_make_views_distinguishes_cancellation_from_magnitude() -> None:
    forward = sparse.csr_matrix([[2.0, -3.0], [0.0, 4.0]])
    reverse = sparse.csr_matrix([[-2.0, 1.0], [0.0, -4.0]])
    views = make_views(forward, reverse)

    np.testing.assert_allclose(views["signed_mean"].toarray(), [[0, -1], [0, 0]])
    np.testing.assert_allclose(views["mean_abs"].toarray(), [[2, 2], [0, 4]])
    np.testing.assert_allclose(views["rms"].toarray(), [[2, np.sqrt(5)], [0, 4]])
    np.testing.assert_allclose(views["max_abs"].toarray(), [[2, 3], [0, 4]])


def test_paired_states_are_exclusive() -> None:
    ref = sparse.csr_matrix([[1, 0, 2, 0], [0, 3, 0, 0]], dtype=np.float32)
    alt = sparse.csr_matrix([[1, 4, 0, 0], [0, 5, 0, 6]], dtype=np.float32)
    activations = Activations(ref=ref, alt=alt, delta=alt - ref)
    states = paired_states(activations)

    np.testing.assert_array_equal(
        states["both"].toarray(), [[1, 0, 0, 0], [0, 1, 0, 0]]
    )
    np.testing.assert_array_equal(
        states["turn_on"].toarray(), [[0, 1, 0, 0], [0, 0, 0, 1]]
    )
    np.testing.assert_array_equal(
        states["turn_off"].toarray(), [[0, 0, 1, 0], [0, 0, 0, 0]]
    )


def test_same_id_correlations_identical_columns_equal_one() -> None:
    matrix = sparse.csr_matrix([[0.0, 1.0, 0.0], [2.0, 0.0, 0.0], [1.0, 3.0, 0.0]])
    pearson, cosine = same_id_correlations(matrix, matrix, rows=matrix.shape[0])
    np.testing.assert_allclose(pearson[:2], 1.0)
    np.testing.assert_allclose(cosine[:2], 1.0)
    assert np.isnan(pearson[2]) and np.isnan(cosine[2])


def test_correlations_to_all_finds_cross_id_match() -> None:
    source = np.array([0.0, 1.0, 2.0, 3.0])
    target = sparse.csr_matrix(
        np.column_stack(
            (
                [3.0, 0.0, 1.0, 0.0],
                [0.0, 2.0, 4.0, 6.0],
                [1.0, 1.0, 1.0, 1.0],
            )
        )
    )
    correlations = correlations_to_all(source, target)
    assert int(np.nanargmax(np.abs(correlations))) == 1
    np.testing.assert_allclose(correlations[1], 1.0)
    assert np.isnan(correlations[2])


def test_select_feature_keeps_test_held_out() -> None:
    rng = np.random.default_rng(424)
    labels: list[str] = []
    splits: list[str] = []
    rows: list[np.ndarray] = []
    for split_name, count in (("discovery", 40), ("validation", 24), ("test", 24)):
        for class_name in ("signal", "other"):
            for _ in range(count):
                row = rng.normal(0, 0.1, size=8).astype(np.float32)
                if class_name == "signal":
                    row[6] += 3
                rows.append(row)
                labels.append(class_name)
                splits.append(split_name)
    result = select_feature(
        sparse.csr_matrix(np.stack(rows)),
        np.asarray(labels),
        np.asarray(splits),
        "signal",
    )
    assert result["feature_id"] == 6
    assert result["direction"] == 1
    assert result["validation_average_precision"] > 0.99
    assert result["test_average_precision"] > 0.99


def test_cross_feature_match_is_selected_on_discovery_and_transfers() -> None:
    rng = np.random.default_rng(9)
    split = np.repeat(["discovery", "validation", "test"], 20)
    source_signal = rng.normal(size=60)
    forward = np.column_stack((source_signal, rng.normal(size=60)))
    reverse = np.column_stack((rng.normal(size=60), -2 * source_signal))
    prior = pl.DataFrame(
        {
            "class": ["a", "a"],
            "orientation": ["forward", "reverse_complement"],
            "dimension": [0, 1],
        }
    )
    matches = cross_feature_matches(
        sparse.csr_matrix(forward),
        sparse.csr_matrix(reverse),
        prior,
        split,
    )
    forward_row = matches.filter(pl.col("source_orientation") == "forward")
    assert forward_row["matched_feature_id"].item() == 1
    assert forward_row["matched_validation_pearson"].item() < -0.99
    assert forward_row["matched_test_pearson"].item() < -0.99


def test_paired_state_summary_reports_ref_alt_transitions() -> None:
    split = np.array(["discovery", "validation", "test", "test"])
    labels = np.array(["a", "a", "a", "b"])
    ref = sparse.csr_matrix([[0], [0], [0], [2]], dtype=np.float32)
    alt = sparse.csr_matrix([[0], [0], [3], [2]], dtype=np.float32)
    activations = Activations(ref=ref, alt=alt, delta=alt - ref)
    selected = pl.DataFrame({"class": ["a"], "feature_id": [0], "view": ["mean_abs"]})
    summary = paired_state_summary(activations, activations, selected, labels, split)
    class_forward = summary.filter(
        (pl.col("orientation") == "forward") & (pl.col("group") == "class")
    )
    other_forward = summary.filter(
        (pl.col("orientation") == "forward") & (pl.col("group") == "other")
    )
    assert class_forward["turn_on_fraction"].item() == 1
    assert other_forward["both_fraction"].item() == 1
