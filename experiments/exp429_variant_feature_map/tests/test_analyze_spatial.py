from __future__ import annotations

import numpy as np

from analyze_spatial import (
    aligned_orientation_profile,
    choose_by_validation,
    oriented_profile,
    spatial_scores,
)


def test_aligned_orientation_profile_reverses_rc_positions() -> None:
    forward = np.array([[1, 2, 3]], dtype=np.float32)
    reverse = np.array([[4, 5, 6]], dtype=np.float32)

    np.testing.assert_array_equal(
        aligned_orientation_profile(forward, reverse, "reverse_complement"),
        np.array([[6, 5, 4]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        aligned_orientation_profile(forward, reverse, "mean"),
        np.array([[3.5, 3.5, 3.5]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        aligned_orientation_profile(forward, reverse, "max_absolute"),
        np.array([[6, 5, 4]], dtype=np.float32),
    )


def test_spatial_scores_apply_frozen_transform_and_direction() -> None:
    raw = np.array([[-2, 1, 3], [4, -5, 2]], dtype=np.float32)
    profile = oriented_profile(raw, transform="absolute", direction=-1)

    np.testing.assert_array_equal(spatial_scores(profile, "focal"), [-1, -5])
    np.testing.assert_array_equal(spatial_scores(profile, "local_max"), [-1, -2])
    np.testing.assert_array_equal(spatial_scores(profile, "local_sum"), [-6, -11])


def test_choose_by_validation_never_uses_test_ap() -> None:
    rows = [
        {
            "class": "a",
            "orientation": "forward",
            "spatial_metric": "focal",
            "dimension": 1,
            "validation_average_precision": 0.8,
            "test_average_precision": 0.1,
        },
        {
            "class": "a",
            "orientation": "forward",
            "spatial_metric": "local_max",
            "dimension": 1,
            "validation_average_precision": 0.7,
            "test_average_precision": 0.9,
        },
    ]

    selected = choose_by_validation(rows, ("class", "orientation"))

    assert len(selected) == 1
    assert selected[0]["spatial_metric"] == "focal"
