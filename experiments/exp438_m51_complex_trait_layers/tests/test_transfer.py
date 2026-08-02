from __future__ import annotations

import numpy as np

from analyze_feature1662_transfer import association_row
from extract_feature1662_transfer import dense_feature_table
from transfer_common import bh_adjust


def test_bh_adjust_known_values() -> None:
    observed = bh_adjust(np.array([0.01, 0.04, 0.03, 0.20]))
    np.testing.assert_allclose(observed, [0.04, 0.0533333333, 0.0533333333, 0.20])


def test_dense_feature_table_preserves_rows_and_delta() -> None:
    table = dense_feature_table(
        np.array([3, 4], dtype=np.uint32),
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([2.5, 4.0], dtype=np.float32),
    )
    assert table.to_pydict() == {
        "panel_row": [3, 4],
        "ref_activation": [1.0, 0.0],
        "alt_activation": [2.5, 4.0],
        "delta": [1.5, 4.0],
    }


def test_association_row_uses_fixed_higher_magnitude_direction() -> None:
    labels = np.array([True] * 30 + [False] * 30)
    response = np.array(list(range(30, 60)) + list(range(30)), dtype=np.float64)
    row = association_row(
        labels,
        response,
        orientation="forward",
        target="missense_variant",
    )
    assert row["standardized_mean_difference"] > 0
    assert row["rank_biserial"] > 0
    assert row["auprc"] == 1.0
