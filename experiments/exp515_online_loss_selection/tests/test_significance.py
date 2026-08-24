"""Tests for the issue #515 matched-group significance gate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from glm_experiments.exp515.significance import (
    _batched_average_precision,
    holm_adjust,
    paired_group_swap_p_worse,
)


def test_batched_average_precision_preserves_ties() -> None:
    labels = np.array([True, False, True, False])
    scores = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
        ]
    )
    observed = _batched_average_precision(scores, labels)
    expected = np.array([average_precision_score(labels, row) for row in scores])
    assert np.allclose(observed, expected)


def test_holm_adjustment_is_monotone_in_sorted_order() -> None:
    observed = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert observed == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})


def _write_evaluation(
    path: Path, *, positive_score: float, negative_score: float
) -> None:
    rows = []
    for group in range(12):
        rows.extend(
            [
                {
                    "chrom": "1",
                    "pos": group * 2 + 1,
                    "ref": "A",
                    "alt": "C",
                    "label": True,
                    "match_group": group,
                    "minus_llr_score": positive_score,
                },
                {
                    "chrom": "1",
                    "pos": group * 2 + 2,
                    "ref": "A",
                    "alt": "G",
                    "label": False,
                    "match_group": group,
                    "minus_llr_score": negative_score,
                },
            ]
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_paired_group_swap_detects_a_clearly_worse_candidate(
    tmp_path: Path,
) -> None:
    bridge = tmp_path / "bridge.csv"
    candidate = tmp_path / "candidate.csv"
    _write_evaluation(bridge, positive_score=1.0, negative_score=0.0)
    _write_evaluation(candidate, positive_score=0.0, negative_score=1.0)
    observed = paired_group_swap_p_worse(
        candidate,
        bridge,
        permutations=5_000,
        seed=515,
        batch_size=250,
    )
    assert observed["delta_auprc"] < 0
    assert observed["p_worse_one_sided"] < 0.05
