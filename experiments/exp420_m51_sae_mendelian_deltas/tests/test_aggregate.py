from __future__ import annotations

import numpy as np

from aggregate import group_center, standardized_orientation_mean


def test_group_center_uses_each_match_group_without_labels() -> None:
    groups = np.repeat(np.asarray(["a", "b"]), 10)
    values = np.concatenate(
        [np.arange(10, dtype=np.float64), np.arange(20, 30, dtype=np.float64)]
    )

    centered = group_center(values, groups)

    np.testing.assert_allclose(centered[:10], np.arange(10) - 4.5)
    np.testing.assert_allclose(centered[10:], np.arange(10) - 4.5)
    assert centered[:10].mean() == centered[10:].mean() == 0


def test_orientation_aggregate_standardizes_on_non_test_only() -> None:
    groups = np.repeat(np.asarray(["discovery", "test"]), 10)
    forward = np.tile(np.arange(10, dtype=np.float64), 2)
    reverse = -forward
    non_test = groups == "discovery"

    forward_z, reverse_z, aggregate, forward_scale, reverse_scale = (
        standardized_orientation_mean(
            forward,
            reverse,
            groups,
            non_test,
        )
    )

    assert forward_scale == reverse_scale
    np.testing.assert_allclose(forward_z, -reverse_z)
    np.testing.assert_allclose(aggregate, 0)
