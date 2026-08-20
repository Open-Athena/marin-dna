from __future__ import annotations

import pandas as pd
import pytest

from exp479_mntp.config import NUCLEOTIDE_LENGTH
from exp479_mntp.context_window_diagnostics import (
    CENTER,
    CONDITIONS,
    ablate_context,
    stability_rows,
)


def test_context_ablation_preserves_target_and_opposite_flank() -> None:
    sequence = ("ACGT" * 64)[:NUCLEOTIDE_LENGTH]
    left = ablate_context(sequence, "left")
    right = ablate_context(sequence, "right")

    assert left[:CENTER] == "N" * CENTER
    assert left[CENTER:] == sequence[CENTER:]
    assert right[: CENTER + 1] == sequence[: CENTER + 1]
    assert right[CENTER + 1 :] == "N" * (NUCLEOTIDE_LENGTH - CENTER - 1)
    with pytest.raises(ValueError, match="unsupported"):
        ablate_context(sequence, "both")


def test_diagnostic_conditions_keep_shifted_variant_in_bounds() -> None:
    assert [condition.name for condition in CONDITIONS] == [
        "centered_full",
        "left_context_ablated",
        "right_context_ablated",
        "window_shift_upstream_64",
        "window_shift_downstream_64",
    ]
    assert all(0 <= condition.variant_index < NUCLEOTIDE_LENGTH for condition in CONDITIONS)


def test_stability_rows_use_centered_scores_as_baseline() -> None:
    scores = pd.DataFrame(
        {
            "centered_full": [0.0, 1.0, 2.0],
            "left_context_ablated": [0.0, 1.0, 2.0],
            "right_context_ablated": [2.0, 1.0, 0.0],
            "window_shift_upstream_64": [0.0, 1.0, 2.0],
            "window_shift_downstream_64": [0.0, 1.0, 2.0],
        }
    )
    rows = pd.DataFrame(stability_rows(scores, dataset="test")).set_index("condition")
    assert rows.loc["centered_full", "spearman_vs_centered"] == pytest.approx(1.0)
    assert rows.loc["left_context_ablated", "mean_absolute_llr_change"] == 0.0
    assert rows.loc["right_context_ablated", "spearman_vs_centered"] == pytest.approx(-1.0)
