from __future__ import annotations

import polars as pl
from marin_dna.data.dna import reverse_complement

from design_perturbations import (
    FOCAL_INDEX,
    coding_codon_start_index,
    codon_consequence,
    codon_sweep_rows,
    perturbation_frame,
    select_contexts,
    splice_saturation_rows,
)


def test_perturbation_frame_infers_mixed_columns_beyond_first_rows() -> None:
    rows = [
        {"perturbation_type": "splice_saturation", "reference_codon": None}
        for _ in range(101)
    ]
    rows.append({"perturbation_type": "codon_sweep", "reference_codon": "CAG"})

    frame = perturbation_frame(rows)

    assert frame.height == 102
    assert frame["reference_codon"].tail(1).item() == "CAG"


def test_select_contexts_uses_top_and_rank_spaced_controls() -> None:
    frame = pl.DataFrame(
        {
            "class": ["x"] * 12,
            "rank": list(range(1, 13)),
            "is_top": [True] * 4 + [False] * 8,
            "panel_row": list(range(12)),
        }
    )

    selected = select_contexts(frame, class_name="x", contexts_per_group=3)

    assert selected.filter(pl.col("context_group") == "top")["rank"].to_list() == [
        1,
        2,
        3,
    ]
    assert selected.filter(pl.col("context_group") == "rank_spaced_control")[
        "rank"
    ].to_list() == [5, 8, 12]


def test_splice_saturation_maps_reverse_positions_and_bases() -> None:
    sequence = "A" * FOCAL_INDEX + "C" + "A" * (FOCAL_INDEX)
    oriented = reverse_complement(sequence[FOCAL_INDEX - 15 : FOCAL_INDEX + 16])
    source = {
        "class": "splice_acceptor_variant",
        "feature_id": 11698,
        "panel_row": 1,
        "rank": 1,
        "context_group": "top",
        "response_orientation": "reverse_complement",
        "chrom": "21",
        "pos": 1000,
        "ref_context": oriented,
    }

    rows = splice_saturation_rows(
        source,
        sequence,
        window_start0=872,
        window_end0=1127,
        relative_positions=[1],
    )

    assert len(rows) == 3
    assert {row["source_state"] for row in rows} == {"T"}
    assert {row["alternate_state"] for row in rows} == {"A", "C", "G"}
    for row in rows:
        changed = [
            index
            for index, (ref, alt) in enumerate(
                zip(sequence, row["alternate_sequence"], strict=True)
            )
            if ref != alt
        ]
        assert changed == [FOCAL_INDEX - 1]


def test_codon_sweep_reconstructs_minus_strand_codon() -> None:
    sequence = list("A" * 255)
    codon_position = 2
    start = coding_codon_start_index(codon_position, "-")
    sequence[start : start + 3] = reverse_complement("CAG")
    sequence = "".join(sequence)
    source = {
        "class": "synonymous_variant",
        "feature_id": 6072,
        "panel_row": 2,
        "rank": 1,
        "context_group": "top",
        "consensus_strand": "-",
        "consensus_codon_position": codon_position,
        "consensus_ref_codon": "CAG",
        "chrom": "21",
        "pos": 1000,
        "ref": "T",
    }

    rows = codon_sweep_rows(
        source,
        sequence,
        window_start0=872,
        window_end0=1127,
    )

    assert len(rows) == 63
    synonymous = next(row for row in rows if row["alternate_codon"] == "CAA")
    assert synonymous["expected_consequence"] == "synonymous_variant"
    assert synonymous["edit_distance"] == 1
    assert (
        reverse_complement(synonymous["alternate_sequence"][start : start + 3]) == "CAA"
    )
    stop = next(row for row in rows if row["alternate_codon"] == "TAG")
    assert stop["expected_consequence"] == "stop_gained"


def test_codon_consequence_distinguishes_stop_synonymous_and_missense() -> None:
    assert codon_consequence("CAG", "TAG") == "stop_gained"
    assert codon_consequence("CAG", "CAA") == "synonymous_variant"
    assert codon_consequence("CAG", "CAC") == "missense_variant"
