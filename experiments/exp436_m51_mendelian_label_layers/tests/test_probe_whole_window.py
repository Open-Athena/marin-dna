from __future__ import annotations

import numpy as np
import pandas as pd

from probe_whole_window import (
    build_probe_feature,
    metric_table,
    pooled_diagnostics,
    run_cell,
)


def test_build_probe_feature_drops_only_constant_components() -> None:
    ref = np.array([[1, 0, 5, 7], [2, 0, 5, 7], [3, 0, 5, 7]], dtype=np.float32)
    alt = np.array([[1, 1, 6, 7], [2, 2, 4, 7], [3, 3, 5, 7]], dtype=np.float32)

    features, mapping = build_probe_feature(ref, alt)

    assert features.dtype == np.float32
    assert mapping["component"].to_list() == ["ref", "delta", "delta"]
    assert mapping["feature_id"].to_list() == [0, 1, 2]
    np.testing.assert_array_equal(features[:, 0], [1, 2, 3])
    np.testing.assert_array_equal(features[:, 1], [1, 2, 3])
    np.testing.assert_array_equal(features[:, 2], [1, -1, 0])


def test_nested_probe_and_metrics_smoke() -> None:
    rng = np.random.default_rng(436)
    chrom = np.repeat(["1", "2", "3"], 20)
    label = np.tile(np.repeat([0, 1], 10), 3)
    features = rng.normal(size=(60, 4)).astype(np.float32)
    features[:, 0] += 2 * label

    predictions, summary, classifier = run_cell(
        features,
        label,
        chrom,
        c_grid=np.array([0.01, 1.0]),
        n_jobs=1,
    )

    assert predictions.shape == (60,)
    assert np.isfinite(predictions).all()
    assert summary["full_c"] in (0.01, 1.0)
    assert classifier.predict_proba(features).shape == (60, 2)

    frame = pd.DataFrame(
        {
            "chrom": chrom,
            "label": label,
            "subset": ["subset"] * 60,
            "sparse_probe_score": predictions,
            "sparse_global_probe_score": predictions,
            "official_probe_score": predictions,
            "minus_llr_avg": predictions,
        }
    )
    metrics = metric_table(frame, "forward")
    assert {"per_subset", "global_raw_matched", "global_all_rows"} <= set(
        metrics["scope"]
    )
    diagnostics = pooled_diagnostics(frame, "forward")
    assert diagnostics.height == 5
