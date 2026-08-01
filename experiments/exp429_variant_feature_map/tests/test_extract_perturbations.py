from __future__ import annotations

import polars as pl

from extract_perturbations import build_state_table


def test_build_state_table_deduplicates_repeated_reference_sequences() -> None:
    panel = pl.DataFrame(
        {
            "chrom": ["21", "21"],
            "window_start0": [100, 100],
            "window_end0": [355, 355],
            "reference_sequence": ["A" * 255, "A" * 255],
            "alternate_sequence": ["C" + "A" * 254, "G" + "A" * 254],
        }
    )

    states, reference_indices, alternate_indices = build_state_table(panel)

    assert states.height == 3
    assert reference_indices.tolist() == [0, 0]
    assert alternate_indices.tolist() == [1, 2]
    sequences = states["sequence"].to_list()
    assert [sequences[index] for index in reference_indices] == panel[
        "reference_sequence"
    ].to_list()
    assert [sequences[index] for index in alternate_indices] == panel[
        "alternate_sequence"
    ].to_list()
