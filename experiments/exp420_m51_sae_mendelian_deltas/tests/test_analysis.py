from __future__ import annotations

import numpy as np
import polars as pl

from analysis import (
    CHROM_SPLITS,
    FOCAL_INDEX,
    TEST_CHROMS,
    _substitution_scores,
    bootstrap_mean_interval,
    matched_contrasts,
    matched_permutation_pvalue,
    select_candidate,
    split_for_chrom,
    variant_sequences,
)


def test_chromosome_splits_are_disjoint_and_cover_dataset_chromosomes() -> None:
    all_chroms = set().union(*CHROM_SPLITS.values())
    assert all_chroms == {
        "1",
        "3",
        "5",
        "7",
        "9",
        "11",
        "13",
        "15",
        "17",
        "19",
        "21",
        "X",
    }
    assert sum(len(chroms) for chroms in CHROM_SPLITS.values()) == len(all_chroms)
    assert TEST_CHROMS == {"11", "X"}
    for split, chroms in CHROM_SPLITS.items():
        assert all(split_for_chrom(chrom) == split for chrom in chroms)


def test_variant_sequences_changes_only_zero_based_focal_base() -> None:
    reference = "A" * FOCAL_INDEX + "C" + "G" * FOCAL_INDEX
    observed_reference, alternate = variant_sequences(reference, "c", "t")

    assert observed_reference == reference
    assert alternate[FOCAL_INDEX] == "T"
    changed = [
        index
        for index, (left, right) in enumerate(
            zip(observed_reference, alternate, strict=True)
        )
        if left != right
    ]
    assert changed == [FOCAL_INDEX]


def test_matched_contrasts_preserve_first_group_order() -> None:
    groups = np.repeat(np.asarray(["b", "a"]), 10)
    labels = np.tile(np.asarray([1] + [0] * 9, dtype=np.int8), 2)
    scores = np.zeros((20, 2), dtype=np.float32)
    scores[0] = [5, -2]
    scores[10] = [3, 7]

    contrasts, order = matched_contrasts(scores, labels, groups)

    assert order.tolist() == ["b", "a"]
    np.testing.assert_allclose(contrasts, [[5, -2], [3, 7]])


def test_selection_ranks_discovery_then_uses_validation_direction() -> None:
    discovery = np.asarray(
        [
            [2.0, -3.0, 0.0],
            [2.2, -2.0, 0.1],
            [1.8, -4.0, -0.1],
            [2.1, -3.5, 0.2],
        ],
        dtype=np.float32,
    )
    validation = np.asarray(
        [
            [-0.4, -2.0, 0.0],
            [-0.2, -2.2, 0.1],
            [0.1, -1.8, -0.1],
            [0.0, -2.1, 0.0],
        ],
        dtype=np.float32,
    )

    selected, direction, replicated, candidates = select_candidate(
        discovery, validation, top_k=2
    )

    assert {row["dimension"] for row in candidates} == {0, 1}
    assert selected == 1
    assert direction == -1
    assert replicated


def test_selection_marks_failed_direction_gate_without_using_test_data() -> None:
    discovery = np.asarray([[2.0], [2.1], [1.9], [2.2]], dtype=np.float32)
    validation = -discovery

    selected, direction, replicated, candidates = select_candidate(
        discovery, validation, top_k=1
    )

    assert selected == 0
    assert direction == 1
    assert not replicated
    assert not candidates[0]["direction_consistent"]


def test_substitution_baseline_falls_back_for_unseen_test_change() -> None:
    frame = pl.DataFrame(
        {
            "ref": ["A", "A", "G", "C"],
            "alt": ["C", "C", "T", "A"],
            "label": [1, 1, 0, 1],
            "subset": ["s"] * 4,
            "split": ["discovery", "validation", "discovery", "test"],
        }
    )

    scores = _substitution_scores(frame, "s")

    assert scores[0] == scores[1] == 0.75
    assert scores[2] == 1 / 3
    assert scores[3] == 0.6


def test_bootstrap_and_group_permutation_are_deterministic() -> None:
    groups = np.repeat(np.arange(8), 10)
    labels = np.tile(np.asarray([1] + [0] * 9, dtype=np.int8), 8)
    scores = labels.astype(np.float64)

    interval_a = bootstrap_mean_interval(
        np.arange(1, 9, dtype=np.float64), seed=17, samples=500
    )
    interval_b = bootstrap_mean_interval(
        np.arange(1, 9, dtype=np.float64), seed=17, samples=500
    )
    pvalue_a = matched_permutation_pvalue(
        scores, labels, groups, seed=19, permutations=500
    )
    pvalue_b = matched_permutation_pvalue(
        scores, labels, groups, seed=19, permutations=500
    )

    assert interval_a == interval_b
    assert pvalue_a == pvalue_b
    assert 0 < pvalue_a < 0.05
