from __future__ import annotations

import numpy as np
import pytest

from window_conservation.evaluate import interval_coverage, metrics


def test_coverage_merges_overlaps_and_respects_boundaries() -> None:
    result = interval_coverage(np.array([0, 10, 20, 30]), np.array([10, 20, 30, 40]),
                               [(4, 12), (8, 14), (20, 25), (25, 28), (40, 50)])
    assert result.tolist() == [6, 4, 8, 0]


def test_matched_control_removes_composition_only_signal() -> None:
    n = 100
    values = {"label": np.r_[np.ones(50), np.zeros(50)], "strata": np.r_[np.ones(50, dtype=int), np.zeros(50, dtype=int)],
              "tie": np.arange(n), "gc": np.ones(n), "repeat": np.zeros(n), "entropy": np.ones(n), "seeds": np.ones(n)}
    result = metrics(values, values["label"], .1)
    assert result["random_enrichment"] == 2
    assert result["matched_enrichment"] == 1


def test_fixed_budget_and_ties_do_not_read_labels() -> None:
    n = 100
    values = {"label": np.arange(n)/100, "strata": np.zeros(n, dtype=int), "tie": np.arange(n)[::-1],
              "gc": np.ones(n), "repeat": np.zeros(n), "entropy": np.ones(n), "seeds": np.ones(n)}
    result = metrics(values, np.zeros(n), .05)
    assert result["selected_windows"] == 5
    assert result["selected_annotated_fraction"] == pytest.approx(.97)
    weighted = metrics(values, np.zeros(n), .05, np.full(n, 2.0))
    assert weighted["selected_windows"] == 10
    assert weighted["selected_annotated_fraction"] == pytest.approx(.97)
