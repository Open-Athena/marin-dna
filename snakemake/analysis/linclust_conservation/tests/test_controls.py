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
    controls = synthetic_sequences()
    passing = tmp_path / "passing.tsv"
    exact_members = {
        "base",
        "exact_duplicate_a",
        "exact_duplicate_b",
        "exact_reverse_complement",
    }
    representatives = {
        name: "base" if name in exact_members else name for name in controls
    }
    passing.write_text(
        "".join(f"{representatives[name]}\t{name}\n" for name in controls)
    )
    alignments = tmp_path / "alignments.tsv"
    alignments.write_text(
        "".join(
            f"{representatives[name]}\t{name}\t1\t255\t1\t1\t1\t255\t"
            + ("255\t1" if name == "exact_reverse_complement" else "1\t255")
            + "\t0\t500\n"
            for name in controls
        )
    )
    receipt = check_release_gate(str(passing), str(alignments))
    assert receipt["release_gate_passed"] is True
    assert receipt["reverse_complement_alignment_verified"] is True

    failing = tmp_path / "failing.tsv"
    failing.write_text("".join(passing.read_text().splitlines(keepends=True)[:-1]))
    with pytest.raises(AssertionError, match="fixture"):
        check_release_gate(str(failing))

    separated = tmp_path / "separated.tsv"
    separated.write_text(
        passing.read_text().replace(
            "base\texact_reverse_complement",
            "exact_reverse_complement\texact_reverse_complement",
        )
    )
    with pytest.raises(AssertionError, match="release gate failed"):
        check_release_gate(str(separated))
