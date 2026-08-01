from __future__ import annotations

import numpy as np
from scipy import sparse

from analyze import make_views, select_feature


def test_make_views_uses_signed_mean_and_max_abs() -> None:
    forward = sparse.csr_matrix([[2.0, -3.0], [0.0, 4.0]])
    reverse = sparse.csr_matrix([[-2.0, 1.0], [0.0, -4.0]])
    views = make_views(forward, reverse)
    np.testing.assert_allclose(views["signed_mean"].toarray(), [[0, -1], [0, 0]])
    np.testing.assert_allclose(views["max_abs"].toarray(), [[2, 3], [0, 4]])


def test_select_feature_keeps_test_held_out() -> None:
    rng = np.random.default_rng(426)
    labels: list[str] = []
    splits: list[str] = []
    rows: list[np.ndarray] = []
    for split_name, count in (("discovery", 256), ("validation", 128), ("test", 128)):
        for class_name in ("signal", "other"):
            for _ in range(count):
                row = rng.normal(0, 0.1, size=8).astype(np.float32)
                if class_name == "signal":
                    row[6] += 3
                rows.append(row)
                labels.append(class_name)
                splits.append(split_name)
    result, scores, positive = select_feature(
        sparse.csr_matrix(np.stack(rows)),
        np.asarray(labels),
        np.asarray(splits),
        "signal",
    )
    assert result["feature_id"] == 6
    assert result["direction"] == 1
    assert result["validation_average_precision"] > 0.99
    assert result["test_average_precision"] > 0.99
    assert scores.shape == positive.shape == (256,)
