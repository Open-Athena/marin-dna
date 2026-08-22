from __future__ import annotations

import pandas as pd

from exp479_mntp.paired_nucleotide_gate import (
    information_gate,
    paired_comparison,
    summarize_readouts,
)


def _scores() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sample_id, baseline_ce, candidate_ce, baseline_correct, candidate_correct in (
        (0, 1.0, 0.8, 0.0, 1.0),
        (1, 1.2, 0.9, 1.0, 1.0),
        (2, 0.8, 0.7, 0.0, 0.0),
        (3, 1.4, 1.0, 1.0, 1.0),
    ):
        for readout, ce, correct in (
            ("baseline", baseline_ce, baseline_correct),
            ("candidate", candidate_ce, candidate_correct),
        ):
            rows.append(
                {
                    "readout": readout,
                    "sample_id": sample_id,
                    "nucleotide_ce": ce,
                    "nucleotide_correct": correct,
                    "full_vocab_ce": ce + 0.1,
                    "full_vocab_correct": correct,
                }
            )
    return pd.DataFrame(rows)


def test_summary_keeps_unweighted_target_mean() -> None:
    summary = summarize_readouts(_scores()).set_index("readout")
    assert summary.loc["baseline", "n_targets"] == 4
    assert summary.loc["baseline", "nucleotide_ce"] == 1.1
    assert summary.loc["candidate", "nucleotide_accuracy"] == 0.75


def test_paired_information_gate_requires_both_metrics_and_confidence() -> None:
    comparison = paired_comparison(
        _scores(),
        candidate="candidate",
        baseline="baseline",
        n_bootstrap=200,
    )
    assert comparison["nucleotide_ce_delta"] < 0
    assert comparison["nucleotide_accuracy_delta"] > 0
    gate = information_gate(comparison)
    assert gate["point_estimate_passed"] is True
    assert gate["confidence_supported"] is True
    assert gate["passed"] is True


def test_information_gate_rejects_accuracy_regression() -> None:
    gate = information_gate(
        {
            "candidate": "candidate",
            "baseline": "baseline",
            "nucleotide_ce_delta": -0.1,
            "nucleotide_ce_delta_ci95_high": -0.01,
            "nucleotide_accuracy_delta": -0.01,
            "nucleotide_accuracy_delta_ci95_low": -0.02,
        }
    )
    assert gate["passed"] is False
