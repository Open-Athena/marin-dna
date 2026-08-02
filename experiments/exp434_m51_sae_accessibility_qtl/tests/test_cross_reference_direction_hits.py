import numpy as np
import polars as pl

from cross_reference_direction_hits import (
    _component_members,
    dense_delta_matrix,
    response_redundancy,
)


def test_dense_delta_matrix_materializes_selected_sparse_features() -> None:
    sparse = pl.DataFrame(
        {
            "panel_row": [0, 2, 1, 2],
            "feature_id": [11, 11, 22, 22],
            "delta": [1.0, -2.0, 3.0, 4.0],
        }
    )

    observed = dense_delta_matrix(
        panel_rows=3,
        sparse=sparse,
        feature_ids=[22, 11],
    )

    np.testing.assert_array_equal(
        observed,
        np.array([[0.0, 1.0], [3.0, 0.0], [4.0, -2.0]]),
    )


def test_component_members_uses_absolute_correlation() -> None:
    correlation = np.array(
        [
            [1.0, -0.95, 0.2],
            [-0.95, 1.0, 0.1],
            [0.2, 0.1, 1.0],
        ]
    )

    assert _component_members(
        correlation,
        [10, 20, 30],
        threshold=0.9,
    ) == [[10, 20], [30]]


def test_response_redundancy_reports_pearson_and_spearman() -> None:
    matrix = np.array(
        [
            [0.0, 5.0, 0.0],
            [1.0, 4.0, 1.0],
            [2.0, 3.0, 0.0],
            [3.0, 2.0, 1.0],
            [4.0, 1.0, 0.0],
            [5.0, 0.0, 1.0],
        ]
    )

    pairs, components, summary = response_redundancy(matrix, [1, 2, 3])

    first = pairs.row(0, named=True)
    assert {first["feature_id_left"], first["feature_id_right"]} == {1, 2}
    assert np.isclose(first["absolute_pearson"], 1.0)
    assert np.isclose(first["absolute_spearman"], 1.0)
    assert summary["features"] == 3
    assert summary["component_metric"] == "absolute Pearson correlation"
    assert set(summary["absolute_pairwise_correlation_quantiles"]) == {
        "pearson",
        "spearman",
    }
    assert summary["threshold_components"]["0.9"]["component_sizes"] == [2, 1]
    assert components.filter(pl.col("absolute_pearson_threshold") == 0.9).group_by(
        "component_index"
    ).agg(pl.col("feature_id").sort()).sort("component_index")[
        "feature_id"
    ].to_list() == [[1, 2], [3]]
