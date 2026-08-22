from __future__ import annotations

import pandas as pd
import pytest

from exp479_mntp.attention_anneal_diagnostic import (
    _endpoint_delta,
    summarize_annealing,
)


def _scores() -> pd.DataFrame:
    rows = []
    for probability, ce, accuracy in ((0.0, 1.0, 0.5), (0.5, 1.2, 0.4), (1.0, 1.4, 0.3)):
        for replicate in range(2):
            for sample_id in range(3):
                rows.append(
                    {
                        "future_edge_probability": probability,
                        "mask_replicate": replicate,
                        "sample_id": sample_id,
                        "component": "test",
                        "target_nucleotide_index": sample_id + 1,
                        "left_context_bases": sample_id + 1,
                        "right_context_bases": 10 - sample_id,
                        "target_base": "A",
                        "repeat_masked_target": False,
                        "nucleotide_ce": ce,
                        "nucleotide_correct": accuracy,
                        "full_vocab_ce": ce + 0.1,
                        "full_vocab_correct": accuracy,
                    }
                )
    return pd.DataFrame(rows)


def test_annealing_summary_preserves_replicates_targets_and_degradation_scale() -> None:
    replicate_summary, trajectory, target_means = summarize_annealing(_scores())
    assert len(replicate_summary) == 6
    assert len(trajectory) == 3
    assert len(target_means) == 9
    assert trajectory["mask_replicates"].tolist() == [2, 2, 2]
    assert trajectory["ce_degradation_fraction"].tolist() == pytest.approx([0.0, 0.5, 1.0])
    assert trajectory["accuracy_degradation_fraction"].tolist() == pytest.approx([0.0, 0.5, 1.0])
    assert target_means.groupby("future_edge_probability").size().tolist() == [3, 3, 3]


def test_annealing_summary_rejects_incomplete_target_panel() -> None:
    scores = _scores()
    scores = scores.drop(index=scores.index[-1])
    with pytest.raises(RuntimeError, match="incomplete target panels"):
        summarize_annealing(scores)


def test_endpoint_delta_reports_loss_and_prediction_disagreement() -> None:
    standard = pd.DataFrame(
        {
            "sample_id": [0, 1],
            "target_nucleotide_index": [3, 4],
            "nucleotide_ce": [1.0, 2.0],
            "nucleotide_correct": [1.0, 0.0],
        }
    )
    custom = standard.copy()
    custom["nucleotide_ce"] = [1.1, 1.8]
    custom["nucleotide_correct"] = [1.0, 1.0]

    delta = _endpoint_delta(custom, standard)

    assert delta["maximum_absolute_nucleotide_ce_delta"] == pytest.approx(0.2)
    assert delta["mean_absolute_nucleotide_ce_delta"] == pytest.approx(0.15)
    assert delta["mean_signed_nucleotide_ce_delta"] == pytest.approx(-0.05)
    assert delta["nucleotide_prediction_mismatches"] == 1
    assert delta["nucleotide_predictions_identical"] is False
