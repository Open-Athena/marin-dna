from __future__ import annotations

import polars as pl

from build_panel import FOCAL_INDEX, WINDOW_BP, materialize_sequences, variant_sequences


def test_variant_sequences_changes_only_focal_base() -> None:
    reference = "A" * FOCAL_INDEX + "C" + "G" * FOCAL_INDEX
    ref, alt = variant_sequences(reference, "C", "T")
    assert ref == reference
    assert alt[FOCAL_INDEX] == "T"
    assert sum(a != b for a, b in zip(ref, alt, strict=True)) == 1


def test_materialize_sequences_uses_zero_based_half_open_window() -> None:
    calls: list[tuple[str, int, int, str]] = []

    def genome(chrom: str, start: int, end: int, strand: str = "+") -> str:
        calls.append((chrom, start, end, strand))
        return "A" * FOCAL_INDEX + "C" + "G" * FOCAL_INDEX

    frame = pl.DataFrame({"chrom": ["1"], "pos": [1_000], "ref": ["C"], "alt": ["T"]})
    result = materialize_sequences(frame, genome)
    pos0 = 999
    assert calls == [("1", pos0 - FOCAL_INDEX, pos0 + FOCAL_INDEX + 1, "+")]
    assert len(result["ref_sequence"].item()) == WINDOW_BP
    assert result["alt_sequence"].item()[FOCAL_INDEX] == "T"
