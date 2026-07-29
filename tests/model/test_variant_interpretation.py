"""Tests for variant-score bundle presentation helpers."""

import numpy as np
import pytest

from marin_dna.model.variant_interpretation import (
    fwd_rc_average_fp32,
    variant_score_bundle_view,
)


def test_fwd_rc_average_fp32_upcasts_before_averaging():
    fwd = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
    rc = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float16)
    actual = fwd_rc_average_fp32([fwd, rc])
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(
        actual,
        np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
    )


def test_variant_score_bundle_view_retains_atoms_and_averages_embedding_blocks():
    n_rows, hidden_size = 3, 2
    fwd = np.zeros((n_rows, 2 + 2 * hidden_size), dtype=np.float32)
    rc = np.zeros_like(fwd)
    fwd[:, :2] = [[1.0, 0.1], [2.0, 0.2], [3.0, 0.3]]
    rc[:, :2] = [[-1.0, 0.4], [-2.0, 0.5], [-3.0, 0.6]]
    fwd[:, 2 : 2 + hidden_size] = [[1, 2], [3, 4], [5, 6]]
    rc[:, 2 : 2 + hidden_size] = [[7, 8], [9, 10], [11, 12]]
    fwd[:, 2 + hidden_size :] = [[2, 4], [6, 8], [10, 12]]
    rc[:, 2 + hidden_size :] = [[14, 16], [18, 20], [22, 24]]

    view = variant_score_bundle_view(
        {"fwd": fwd, "rc": rc},
        hidden_size=hidden_size,
    )

    assert list(view.scores.columns) == [
        "llr_fwd",
        "jsd_fwd",
        "llr_rc",
        "jsd_rc",
    ]
    np.testing.assert_array_equal(view.scores["llr_fwd"], fwd[:, 0])
    np.testing.assert_array_equal(view.scores["jsd_rc"], rc[:, 1])
    assert view.ref_embeddings is not None
    assert view.alt_embeddings is not None
    assert view.ref_embeddings.dtype == np.float32
    assert view.alt_embeddings.dtype == np.float32
    np.testing.assert_array_equal(
        view.ref_embeddings,
        (fwd[:, 2 : 2 + hidden_size] + rc[:, 2 : 2 + hidden_size]) / 2,
    )
    np.testing.assert_array_equal(
        view.alt_embeddings,
        (fwd[:, 2 + hidden_size :] + rc[:, 2 + hidden_size :]) / 2,
    )


def test_variant_score_bundle_view_handles_forward_only_scores():
    fwd = np.array([[1.0, 0.1], [2.0, 0.2]], dtype=np.float32)
    view = variant_score_bundle_view({"fwd": fwd})
    assert list(view.scores.columns) == ["llr_fwd", "jsd_fwd"]
    assert view.ref_embeddings is None
    assert view.alt_embeddings is None


def test_variant_score_bundle_view_handles_both_strands_without_embeddings():
    fwd = np.array([[1.0, 0.1], [2.0, 0.2]], dtype=np.float32)
    rc = np.array([[-1.0, 0.3], [-2.0, 0.4]], dtype=np.float32)

    view = variant_score_bundle_view({"fwd": fwd, "rc": rc})

    assert list(view.scores.columns) == [
        "llr_fwd",
        "jsd_fwd",
        "llr_rc",
        "jsd_rc",
    ]
    np.testing.assert_array_equal(view.scores["llr_fwd"], fwd[:, 0])
    np.testing.assert_array_equal(view.scores["jsd_fwd"], fwd[:, 1])
    np.testing.assert_array_equal(view.scores["llr_rc"], rc[:, 0])
    np.testing.assert_array_equal(view.scores["jsd_rc"], rc[:, 1])
    assert view.ref_embeddings is None
    assert view.alt_embeddings is None


@pytest.mark.parametrize(
    ("results", "hidden_size", "message"),
    [
        ({}, None, "empty"),
        ({"rc": np.zeros((2, 2))}, None, "forward"),
        ({"fwd": np.zeros((2, 3))}, None, "width"),
        (
            {"fwd": np.zeros((2, 6)), "rc": np.zeros((3, 6))},
            2,
            "row count",
        ),
        (
            {"fwd": np.zeros((2, 6)), "rc": np.full((2, 6), np.nan)},
            2,
            "non-finite",
        ),
        ({"fwd": np.zeros((2, 6))}, 2, "both forward"),
    ],
)
def test_variant_score_bundle_view_rejects_corrupt_layout(
    results,
    hidden_size,
    message,
):
    with pytest.raises(AssertionError, match=message):
        variant_score_bundle_view(results, hidden_size=hidden_size)
