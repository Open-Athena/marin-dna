from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.metrics import roc_auc_score

from analyze import bootstrap_mean_ap, make_views, select_feature
from controls import make_baseline_designs, matched_substitution_auc
from pairwise import bootstrap_pairwise_metrics, load_selected_activations


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


def test_mean_ap_bootstrap_preserves_spatially_clustered_classes() -> None:
    labels = np.asarray(["a"] * 4 + ["b"] * 4)
    blocks = np.asarray([0, 0, 0, 0, 2, 2, 3, 3])
    positive_a = labels == "a"
    positive_b = labels == "b"
    scores_a = np.asarray([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])
    scores_b = 1.0 - scores_a
    low, high = bootstrap_mean_ap(
        {
            "a": (scores_a, positive_a),
            "b": (scores_b, positive_b),
        },
        blocks,
        seed=426,
        samples=100,
    )
    assert 0 <= low <= high <= 1
    assert low > 0.9


def test_load_selected_activations_remaps_panel_rows(tmp_path) -> None:
    path = tmp_path / "activations.parquet"
    table = pa.table(
        {
            "panel_row": [2, 2, 8, 8, 11],
            "feature_id": [1, 3, 0, 2, 1],
            "ref_activation": np.asarray([1, 0, 2, 1, 4], dtype=np.float32),
            "alt_activation": np.asarray([3, 4, 1, 0, 7], dtype=np.float32),
            "delta": np.asarray([2, 4, -1, -1, 3], dtype=np.float32),
        }
    )
    pq.write_table(table, path, row_group_size=2)
    loaded = load_selected_activations(
        path, selected_rows=np.asarray([2, 8]), columns=4
    )
    np.testing.assert_array_equal(
        loaded.ref.toarray(), np.asarray([[0, 1, 0, 0], [2, 0, 1, 0]])
    )
    np.testing.assert_array_equal(
        loaded.alt.toarray(), np.asarray([[0, 3, 0, 4], [1, 0, 0, 0]])
    )
    np.testing.assert_array_equal(
        loaded.delta.toarray(), (loaded.alt - loaded.ref).toarray()
    )


def test_pairwise_bootstrap_reports_perfect_separation() -> None:
    positive = np.asarray([True, True, True, True, False, False, False, False])
    blocks = np.asarray([0, 0, 1, 1, 2, 2, 3, 3])
    scores = np.asarray([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])
    result = bootstrap_pairwise_metrics(scores, positive, blocks, seed=426, samples=100)
    assert result == {
        "test_average_precision_ci95_low": 1.0,
        "test_average_precision_ci95_high": 1.0,
        "test_auroc_ci95_low": 1.0,
        "test_auroc_ci95_high": 1.0,
    }


def test_baseline_designs_keep_focal_context_and_alternate() -> None:
    contexts = np.asarray(["A" * 15 + "C" + "G" * 15, "T" * 31])
    alternate = np.asarray(["A", "G"])
    designs = make_baseline_designs(contexts, alternate)
    np.testing.assert_array_equal(designs["centered_1mer_alt"].ravel(), ["C>A", "T>G"])
    np.testing.assert_array_equal(
        designs["centered_3mer_alt"].ravel(), ["ACG>A", "TTT>G"]
    )
    assert designs["positional_31bp_alt"].shape == (2, 32)
    np.testing.assert_array_equal(designs["positional_31bp_alt"][:, -1], alternate)


def test_matched_substitution_auc_removes_allele_spectrum_signal() -> None:
    substitutions = np.asarray(["A>C"] * 4 + ["G>T"] * 4)
    positive = np.asarray([True, False, False, False, True, True, True, False])
    scores = np.asarray([0.0] * 4 + [1.0] * 4)
    assert roc_auc_score(positive, scores) > 0.5
    assert matched_substitution_auc(scores, positive, substitutions) == 0.5
