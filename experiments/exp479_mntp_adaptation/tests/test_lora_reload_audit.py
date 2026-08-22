from __future__ import annotations

import pandas as pd
import pytest

from exp479_mntp.lora_reload_audit import paired_score_parity


def _scores(*, ce_delta: float = 0.0, correctness_delta: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [0, 1],
            "target_nucleotide_index": [17, 29],
            "target_base": ["A", "T"],
            "nucleotide_ce": [1.0 + ce_delta, 1.2 + ce_delta],
            "nucleotide_correct": [1.0, 0.0 + correctness_delta],
            "full_vocab_ce": [2.0 + ce_delta, 2.2 + ce_delta],
            "full_vocab_correct": [0.0, 0.0 + correctness_delta],
        }
    )


def test_paired_score_parity_accepts_exact_roundtrip() -> None:
    checks = paired_score_parity(_scores(), _scores(), ce_tolerance=0.0)
    assert checks["passed"] is True
    assert checks["n_targets"] == 2
    assert checks["nucleotide_ce_maximum_absolute_delta"] == 0.0
    assert checks["nucleotide_correctness_mismatches"] == 0


def test_paired_score_parity_rejects_ce_or_correctness_change() -> None:
    ce_checks = paired_score_parity(_scores(), _scores(ce_delta=0.01), ce_tolerance=1e-3)
    correctness_checks = paired_score_parity(
        _scores(),
        _scores(correctness_delta=1.0),
        ce_tolerance=0.0,
    )
    assert ce_checks["passed"] is False
    assert correctness_checks["passed"] is False
    assert correctness_checks["nucleotide_correctness_mismatches"] == 1


def test_paired_score_parity_rejects_changed_target_identity() -> None:
    changed = _scores()
    changed.loc[0, "target_nucleotide_index"] = 18
    with pytest.raises(RuntimeError, match="identical paired targets"):
        paired_score_parity(_scores(), changed, ce_tolerance=0.0)


def test_paired_score_parity_rejects_negative_tolerance() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        paired_score_parity(_scores(), _scores(), ce_tolerance=-1.0)
