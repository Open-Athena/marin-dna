from __future__ import annotations

import numpy as np
import pytest

from window_conservation.evaluate import metrics


def test_matched_control_removes_composition_only_signal() -> None:
    n = 100
    values = {
        "start": np.arange(n) * 100,
        "end": (np.arange(n) + 1) * 100,
        "label": np.r_[np.ones(50), np.zeros(50)],
        "strata": np.r_[np.ones(50, dtype=int), np.zeros(50, dtype=int)],
        "tie": np.arange(n),
        "gc": np.ones(n),
        "repeat": np.zeros(n),
        "entropy": np.ones(n),
        "seeds": np.ones(n),
    }
    result = metrics(values, values["label"], 0.1)
    assert result["random_enrichment"] == 2
    assert result["matched_enrichment"] == 1


def test_fixed_budget_and_ties_do_not_read_labels() -> None:
    n = 100
    values = {
        "start": np.arange(n) * 100,
        "end": (np.arange(n) + 1) * 100,
        "label": np.arange(n) / 100,
        "strata": np.zeros(n, dtype=int),
        "tie": np.arange(n)[::-1],
        "gc": np.ones(n),
        "repeat": np.zeros(n),
        "entropy": np.ones(n),
        "seeds": np.ones(n),
    }
    result = metrics(values, np.zeros(n), 0.05)
    assert result["selected_windows"] == 5
    assert result["selected_annotated_fraction"] == pytest.approx(0.97)
    weighted = metrics(values, np.zeros(n), 0.05, np.full(n, 2.0))
    assert weighted["selected_windows"] == 10
    assert weighted["selected_annotated_fraction"] == pytest.approx(0.97)


def test_stretches_merge_adjacent_bins_without_bridging_gaps(tmp_path) -> None:
    from window_conservation.evaluate import write_stretches

    values = {
        "start": np.array([0, 100, 200, 400, 500]),
        "end": np.array([100, 200, 300, 500, 600]),
        "tie": np.arange(5),
    }
    output = tmp_path / "stretches.bed"
    write_stretches(output, "chr1", values, np.array([1.0, 0.9, 0.0, 0.8, 0.7]), 0.8)
    assert output.read_text().splitlines() == [
        "chr1\t0\t200\tcandidate_1\t950\t.\t0.95\t2",
        "chr1\t400\t600\tcandidate_2\t750\t.\t0.75\t2",
    ]


def test_linear_selection_matches_ranked_budget_with_ties() -> None:
    from window_conservation.evaluate import select_indices

    rng = np.random.default_rng(577)
    score = rng.integers(0, 7, 1000).astype(float)
    tie = rng.permutation(1000)
    for count in [1, 50, 500, 1000]:
        expected = np.sort(np.lexsort((tie, -score))[:count])
        np.testing.assert_array_equal(select_indices(score, tie, count), expected)
