from __future__ import annotations

import numpy as np

from analyze import (
    BASELINE_WINDOW_BP,
    bootstrap_matched_aucs,
    encode_positional_contexts,
    feature_score_views,
    fit_sequence_baseline,
    matched_auc,
    stratified_block_resample_indices,
)


def test_matched_auc_weights_comparable_pairs() -> None:
    scores = np.asarray([0, 1, 0, 1, 0, 1, 2, 3], dtype=float)
    positive = np.asarray([False, True, False, True, True, False, False, True])
    strata = np.asarray(["a", "a", "a", "a", "b", "b", "b", "b"])
    # Stratum a is perfect; stratum b has two correct of four pairs.
    assert matched_auc(scores, positive, strata) == 0.75


def test_block_bootstrap_never_mixes_fixed_label_strata() -> None:
    strata = np.asarray(["0|a", "0|a", "1|a", "1|a", "0|b", "1|b"])
    blocks = np.asarray([1, 2, 1, 2, 3, 3])
    indices = stratified_block_resample_indices(
        strata, blocks, np.random.default_rng(7)
    )
    assert set(strata[indices]) == set(strata)
    assert len(indices) > 0


def test_bootstrap_intervals_cover_deterministic_perfect_score() -> None:
    positive = np.asarray([False, True] * 4)
    strata = np.asarray(["a", "a", "a", "a", "b", "b", "b", "b"])
    blocks = np.asarray([1, 1, 2, 2, 3, 3, 4, 4])
    scores = positive.astype(float)[:, None]
    result = bootstrap_matched_aucs(
        scores, ["perfect"], positive, strata, blocks, samples=20, seed=1
    )
    assert result["conditional_auc_ci_low"].item() == 1.0
    assert result["conditional_auc_ci_high"].item() == 1.0


def test_feature_views_keep_orientations_separate_before_aggregation() -> None:
    forward = np.asarray([2.0, -3.0])
    reverse = np.asarray([-4.0, 1.0])
    strand = np.asarray(["+", "-"])
    views = feature_score_views(forward, reverse, strand, direction=1)
    np.testing.assert_array_equal(views["forward"], forward)
    np.testing.assert_array_equal(views["reverse_complement"], reverse)
    np.testing.assert_array_equal(views["coding_aligned"], [2.0, 1.0])
    np.testing.assert_array_equal(views["anti_aligned"], [-4.0, -3.0])
    np.testing.assert_array_equal(views["signed_mean"], [-1.0, -1.0])
    np.testing.assert_array_equal(views["max_abs"], [4.0, 3.0])


def test_positional_baseline_encodes_every_position_and_alt() -> None:
    contexts = ["A" * BASELINE_WINDOW_BP, "C" * BASELINE_WINDOW_BP]
    design = encode_positional_contexts(contexts, ["G", "T"])
    assert design.shape == (2, BASELINE_WINDOW_BP * 4 + 4)
    np.testing.assert_array_equal(
        design[:, : BASELINE_WINDOW_BP * 4].sum(axis=1),
        [BASELINE_WINDOW_BP, BASELINE_WINDOW_BP],
    )
    np.testing.assert_array_equal(design[:, -4:].sum(axis=1), [1, 1])


def test_baseline_selection_does_not_use_test_labels() -> None:
    rng = np.random.default_rng(3)
    design = rng.normal(size=(24, 4))
    split = np.asarray(["discovery"] * 8 + ["validation"] * 8 + ["test"] * 8)
    strata = np.asarray(["a", "a", "b", "b"] * 6)
    positive = np.asarray([False, True, False, True] * 6)
    _, first = fit_sequence_baseline(design, positive, split, strata)
    changed = positive.copy()
    changed[split == "test"] = ~changed[split == "test"]
    _, second = fit_sequence_baseline(design, changed, split, strata)
    assert first["selected_c"] == second["selected_c"]
    assert (
        first["candidate_validation_metrics"] == second["candidate_validation_metrics"]
    )
