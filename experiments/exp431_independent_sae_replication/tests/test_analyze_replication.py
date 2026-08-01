from __future__ import annotations

import numpy as np
import polars as pl

from analyze_replication import (
    SPATIAL_RADIUS,
    bootstrap_mean,
    context_effects,
    response_views,
    signed_peak,
)


def test_signed_peak_preserves_sign() -> None:
    values = np.array(
        [
            [[1.0, -4.0], [-3.0, 2.0], [2.0, 1.0]],
            [[-5.0, 1.0], [4.0, -6.0], [1.0, 2.0]],
        ]
    )
    np.testing.assert_array_equal(signed_peak(values), [[-3.0, -4.0], [-5.0, -6.0]])


def test_response_views_reverse_rc_position_axis() -> None:
    positions = 2 * SPATIAL_RADIUS + 1
    fwd = np.zeros((1, positions, 1), dtype=np.float32)
    rc = np.zeros_like(fwd)
    fwd[0, SPATIAL_RADIUS, 0] = 2.0
    rc[0, SPATIAL_RADIUS, 0] = 4.0
    rc[0, 0, 0] = -9.0
    views = response_views(fwd, rc)
    assert views["mean_focal"].item() == 3.0
    assert views["rc_local_peak"].item() == -9.0
    assert views["maxabs_local_peak"].item() == -9.0


def test_context_effects_equal_weights_sources() -> None:
    panel = pl.DataFrame(
        {
            "perturbation_row": [0, 1, 2, 3, 4, 5],
            "class": ["stop_gained"] * 6,
            "source_panel_row": [10, 10, 10, 20, 20, 20],
            "expected_consequence": [
                "stop_gained",
                "missense_variant",
                "missense_variant",
                "stop_gained",
                "synonymous_variant",
                "missense_variant",
            ],
            "relative_position": [0] * 6,
        }
    )
    scores = np.array([[5.0], [1.0], [3.0], [7.0], [4.0], [2.0]])
    sources, effects = context_effects(panel, scores, concept="stop_creation")
    np.testing.assert_array_equal(sources, [10, 20])
    np.testing.assert_allclose(effects[:, 0], [3.0, 4.0])


def test_bootstrap_mean_is_reproducible() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    first = bootstrap_mean(values, seed=431, samples=100)
    second = bootstrap_mean(values, seed=431, samples=100)
    assert first == second
    assert first[0] == 2.5
