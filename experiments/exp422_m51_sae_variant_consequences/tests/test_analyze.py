from __future__ import annotations

import numpy as np
from scipy import sparse

from analyze import (
    bootstrap_block_ap,
    choose_transform,
    column_stats,
    select_individual_feature,
)


def test_column_stats_sparse_matches_dense() -> None:
    dense = np.array([[0, 1, -2], [3, 0, -1], [0, 2, 0], [4, 0, 1]], dtype=np.float32)
    indices = np.array([0, 1, 3])
    dense_stats = column_stats(dense, indices)
    sparse_stats = column_stats(sparse.csr_matrix(dense), indices)
    for dense_value, sparse_value in zip(dense_stats, sparse_stats, strict=True):
        np.testing.assert_allclose(dense_value, sparse_value, rtol=1e-6, atol=1e-6)


def test_select_individual_feature_uses_discovery_and_validation() -> None:
    rng = np.random.default_rng(422)
    labels: list[str] = []
    splits: list[str] = []
    rows: list[np.ndarray] = []
    for split, per_class in (("discovery", 256), ("validation", 128), ("test", 128)):
        for class_name in ("signal", "other"):
            for _ in range(per_class):
                row = rng.normal(0, 0.2, size=6).astype(np.float32)
                if class_name == "signal":
                    row[4] += 3
                rows.append(row)
                labels.append(class_name)
                splits.append(split)
    result = select_individual_feature(
        sparse.csr_matrix(np.stack(rows)),
        np.asarray(labels),
        np.asarray(splits),
        "signal",
    )
    assert result["dimension"] == 4
    assert result["direction"] == 1
    assert result["validation_direction_consistent"]
    assert result["test_average_precision"] > 0.99


def test_bootstrap_block_ap_returns_finite_interval() -> None:
    scores = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.6, 0.3, 0.4])
    positive = np.array([1, 1, 0, 0, 1, 1, 0, 0], dtype=bool)
    blocks = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    low, high, n_blocks, positive_blocks = bootstrap_block_ap(
        scores, positive, blocks, seed=1, samples=50
    )
    assert low is not None and high is not None
    assert 0 <= low <= high <= 1
    assert n_blocks == positive_blocks == 2


def test_bootstrap_block_ap_omits_interval_without_spatial_replication() -> None:
    scores = np.array([0.9, 0.8, 0.1, 0.2, 0.3, 0.4])
    positive = np.array([1, 1, 0, 0, 0, 0], dtype=bool)
    blocks = np.array([1, 1, 2, 2, 3, 3])

    low, high, n_blocks, positive_blocks = bootstrap_block_ap(
        scores, positive, blocks, seed=1, samples=50
    )

    assert low is None and high is None
    assert n_blocks == 3
    assert positive_blocks == 1


def test_choose_transform_uses_validation_not_test() -> None:
    import polars as pl

    frame = pl.DataFrame(
        {
            "class": ["a", "a"],
            "orientation": ["forward", "forward"],
            "space": ["sae", "sae"],
            "transform": ["signed", "absolute"],
            "validation_oriented_t": [2.0, 1.0],
            "test_average_precision": [0.1, 0.9],
        }
    )
    selected = choose_transform(frame)
    assert selected.height == 1
    assert selected["transform"].item() == "signed"
