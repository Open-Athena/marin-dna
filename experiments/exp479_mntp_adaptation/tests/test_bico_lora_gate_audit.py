from __future__ import annotations

import pandas as pd
import pytest

from exp479_mntp.bico_lora_gate_audit import corrected_trajectory_tables

STEPS = (0, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1_000)


def _readout(name: str, *, nucleotide_ce: float, correct: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "readout": name,
            "sample_id": range(640),
            "component": ["cds"] * 640,
            "target_nucleotide_index": range(640),
            "left_context_bases": range(640),
            "right_context_bases": range(640),
            "target_base": ["A"] * 640,
            "repeat_masked_target": [False] * 640,
            "nucleotide_ce": [nucleotide_ce] * 640,
            "nucleotide_correct": [correct] * 640,
            "full_vocab_ce": [nucleotide_ce] * 640,
            "full_vocab_correct": [correct] * 640,
        }
    )


def _stored_trajectory() -> pd.DataFrame:
    frames = []
    for step in STEPS:
        name = f"lora_full_step{step:04d}"
        frame = _readout(name, nucleotide_ce=1.4, correct=0.0)
        frame["optimizer_step"] = step
        frames.append(frame)
    frames.append(_readout("source_causal_adapter_disabled_step0", nucleotide_ce=1.4, correct=0.0))
    return pd.concat(frames, ignore_index=True)


def test_corrected_tables_replace_invalid_source_and_reloaded_final() -> None:
    source = _readout("source_causal_standard_sdpa", nucleotide_ce=1.0, correct=0.0)
    final = _readout("reloaded_final", nucleotide_ce=0.9, correct=1.0)

    scores, summary, comparisons, gate = corrected_trajectory_tables(
        _stored_trajectory(),
        source,
        final,
        n_bootstrap=20,
    )

    indexed = summary.set_index("readout")
    assert indexed.loc["source_causal_adapter_disabled_step0", "nucleotide_ce"] == 1.0
    assert indexed.loc["lora_full_step1000", "nucleotide_ce"] == 0.9
    assert len(scores) == 14 * 640
    assert len(comparisons) == len(STEPS)
    assert gate["passed"] is True


def test_corrected_tables_reject_incomplete_retained_trajectory() -> None:
    stored = _stored_trajectory()
    stored = stored[stored["readout"] != "lora_full_step0900"]
    source = _readout("source", nucleotide_ce=1.0, correct=1.0)
    final = _readout("final", nucleotide_ce=1.0, correct=1.0)

    with pytest.raises(RuntimeError, match="trajectory is incomplete"):
        corrected_trajectory_tables(stored, source, final, n_bootstrap=20)
