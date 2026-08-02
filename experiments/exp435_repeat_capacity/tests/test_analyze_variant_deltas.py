from __future__ import annotations

import numpy as np
import polars as pl

from analyze_variant_deltas import (
    capacity_rows,
    run_contrast,
    targeted_delta_matrix,
)


def test_targeted_matrix_and_metadata_preserve_feature_ids() -> None:
    table = pl.DataFrame(
        {
            "panel_row": [0, 1, 1],
            "feature_id": [7, 7, 11],
            "ref_activation": [0.0, 2.0, 1.0],
            "alt_activation": [3.0, 0.0, 1.5],
            "delta": [3.0, -2.0, 0.5],
        }
    )
    matrix, mapping = targeted_delta_matrix(table, np.array([7, 11]))
    assert matrix.shape == (16_140, 2)
    assert matrix[0, mapping[7]] == 3
    assert matrix[1, mapping[7]] == -2
    assert matrix[1, mapping[11]] == 0.5

    matrix[:40, mapping[7]] = np.concatenate(
        (np.linspace(2, 4, 20), np.linspace(-4, -2, 20))
    )
    result = run_contrast(
        matrix,
        mapping,
        np.array([7]),
        np.arange(20),
        np.arange(20, 40),
        response="delta",
        minimum_support=16,
        metadata={
            "arm": "block19-25m",
            "orientation": "forward",
            "scope": "broad",
            "hierarchy": "repeat",
            "target": "repeat_vs_repeat_free",
            "variant_sensitivity": "all",
        },
    )
    assert result is not None
    assert result["feature_id"].item() == 7
    assert result["block"].item() == 19
    assert result["response"].item() == "delta"


def test_capacity_summary_uses_nonzero_delta_denominator() -> None:
    table = pl.DataFrame(
        {
            "panel_row": [0, 0, 1, 2],
            "feature_id": [7, 11, 7, 7],
            "ref_activation": [0.0, 1.0, 1.0, 1.0],
            "alt_activation": [2.0, 1.0, 0.0, 3.0],
            "delta": [2.0, 0.0, -1.0, 2.0],
        }
    )
    panel = pl.DataFrame(
        {
            "position_status": [
                "focal_repeat",
                "near_repeat",
                "repeat_free_window",
            ]
        }
    )
    rows = capacity_rows(
        table,
        panel,
        np.array([7]),
        arm="block19-25m",
        orientation="forward",
    )
    focal = next(row for row in rows if row["position_status"] == "focal_repeat")
    assert focal["total_nonzero_delta_slots"] == 1
    assert focal["repeat_feature_nonzero_delta_slots"] == 1
    assert focal["repeat_feature_fraction_abs_delta_mass"] == 1.0
    assert focal["inactive_to_active_slots"] == 1
