from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score

from analyze_focal import (
    MIN_NONZERO_SUPPORT,
    Target,
    analyze_target,
    average_precision_both_directions,
    bh_adjust,
)


def test_bh_adjust_preserves_order_and_nan() -> None:
    values = np.array([0.01, 0.04, 0.03, np.nan])
    observed = bh_adjust(values)
    np.testing.assert_allclose(observed[:3], [0.03, 0.04, 0.04])
    assert np.isnan(observed[3])


def test_average_precision_matches_sklearn_with_large_ties() -> None:
    labels = np.array([0, 1, 0, 1, 0, 1], dtype=np.uint8)
    scores = np.array(
        [
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    raw, negated = average_precision_both_directions(labels, scores, chunk_size=1)
    for feature in range(scores.shape[1]):
        assert raw[feature] == average_precision_score(labels, scores[:, feature])
        assert negated[feature] == average_precision_score(labels, -scores[:, feature])


def test_analyze_target_excludes_rare_response_and_finds_signal(
    monkeypatch,
) -> None:
    rows = 100
    features = 4
    labels = np.array([1] * 40 + [0] * 60, dtype=np.uint8)
    response = np.zeros((rows, features), dtype=np.float32)
    response[:40, 0] = 2
    response[40:, 0] = np.arange(60, dtype=np.float32) / 100
    response[:, 1] = np.linspace(0, 1, rows)
    response[: MIN_NONZERO_SUPPORT - 1, 2] = 1
    response[:, 3] = 1
    target = Target(
        kind="overall",
        name="overall",
        indices=np.arange(rows),
        labels=labels,
    )
    monkeypatch.setattr("analyze_focal.EXPECTED_ROWS", rows)
    monkeypatch.setattr("analyze_focal.D_SAE", features)
    result = analyze_target(
        response,
        target,
        arm="block19-25m",
        block=19,
        budget=25_000_200,
        orientation="forward",
        response_name="abs_delta",
        ap_chunk_size=2,
    )
    assert result["feature_id"].to_list() == [0, 1]
    signal = result.filter(pl.col("feature_id") == 0).row(0, named=True)
    assert signal["mean_difference"] > 1
    assert signal["best_auprc"] > 0.9
    assert signal["best_auprc_direction"] == "higher"
