from __future__ import annotations

import polars as pl

from extract_candidates import build_state_table, validate_state_table


def test_state_validation_allows_cross_pair_deduplication() -> None:
    panel = pl.DataFrame(
        {
            "chrom": ["21", "21"],
            "window_start0": [100, 100],
            "window_end0": [355, 355],
            "reference_sequence": ["A" * 255, "C" + "A" * 254],
            "alternate_sequence": ["C" + "A" * 254, "A" * 255],
        }
    )

    states, reference_indices, alternate_indices = build_state_table(panel)

    assert states.height == panel.height
    validate_state_table(panel, states, reference_indices, alternate_indices)
