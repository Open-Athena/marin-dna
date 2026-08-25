from __future__ import annotations

from pathlib import Path

import pytest

from marin_dna_linclust_conservation.controls import (
    check_release_gate,
    reverse_complement,
    synthetic_sequences,
)


def test_synthetic_controls_cover_required_cases() -> None:
    controls = synthetic_sequences()
    assert controls["exact_reverse_complement"] == reverse_complement(controls["base"])
    assert len(controls["base"]) == 255
    assert len(controls["short_insertion"]) == 255
    assert len(controls["short_deletion"]) == 240
    assert sum(base.islower() for base in controls["soft_masked_25pct"]) == 64


def test_release_gate_requires_reverse_complement_cluster(tmp_path: Path) -> None:
    passing = tmp_path / "passing.tsv"
    passing.write_text(
        "base\tbase\n"
        "base\texact_duplicate_a\n"
        "base\texact_duplicate_b\n"
        "base\texact_reverse_complement\n"
    )
    assert check_release_gate(str(passing))["release_gate_passed"] is True

    failing = tmp_path / "failing.tsv"
    failing.write_text(
        "base\tbase\n"
        "base\texact_duplicate_a\n"
        "base\texact_duplicate_b\n"
        "exact_reverse_complement\texact_reverse_complement\n"
    )
    with pytest.raises(AssertionError, match="release gate failed"):
        check_release_gate(str(failing))
