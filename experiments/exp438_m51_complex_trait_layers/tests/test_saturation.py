from __future__ import annotations

import numpy as np
import polars as pl

from analyze_feature1662_saturation import (
    add_bh,
    codon_phase_tests,
    consequence_tests,
    generic_focal_tests,
)
from saturation_states import (
    MUTATIONS_PER_CONTEXT,
    build_response_table,
    enumerate_context_states,
)
from saturation_common import (
    POSITIONS,
    bh_adjust,
    codon_genomic_offsets,
    mutate_transcript_codon,
    parse_codon_change,
)


def context_row(*, context_index: int = 0, strand: int = 1) -> dict[str, object]:
    return {
        "context_index": context_index,
        "ref": "A" if strand == 1 else "T",
        "strand": strand,
        "focal_codon_position": 3,
        "ref_codon": "GAA",
    }


def test_codon_helpers_handle_both_strands() -> None:
    assert parse_codon_change("Gaa/Gag") == ("GAA", "GAG", 2)
    assert codon_genomic_offsets(3, 1) == (-2, -1, 0)
    assert codon_genomic_offsets(3, -1) == (2, 1, 0)
    assert mutate_transcript_codon(
        "GAA",
        focal_codon_position=3,
        strand=1,
        genomic_offset=0,
        genomic_alt="G",
    ) == (3, "GAG", "E", "synonymous")
    assert mutate_transcript_codon(
        "GAA",
        focal_codon_position=3,
        strand=-1,
        genomic_offset=0,
        genomic_alt="C",
    ) == (3, "GAG", "E", "synonymous")


def test_enumerate_context_states_is_complete_and_single_edit() -> None:
    reference = "A" * 255
    states = enumerate_context_states(context_row(), reference)
    assert len(states) == 1 + MUTATIONS_PER_CONTEXT
    assert states[0]["is_baseline"]
    mutations = states[1:]
    assert {state["genomic_offset"] for state in mutations} == set(POSITIONS)
    assert all(
        sum(
            left != right
            for left, right in zip(reference, state["sequence"], strict=True)
        )
        == 1
        for state in mutations
    )
    center = [state for state in mutations if state["genomic_offset"] == 0]
    assert sorted(state["consequence"] for state in center) == [
        "missense",
        "missense",
        "synonymous",
    ]


def test_build_response_table_pairs_each_mutant_with_context_baseline() -> None:
    rows = enumerate_context_states(context_row(), "A" * 255)
    for state_index, row in enumerate(rows):
        row["state_index"] = state_index
        row["baseline_state_index"] = 0
    states = pl.DataFrame(rows).select(
        "state_index",
        "baseline_state_index",
        "context_index",
        "is_baseline",
        "genomic_offset",
        "transcript_offset",
        "genomic_ref",
        "genomic_alt",
        "edited_codon_position",
        "alternate_codon",
        "alternate_amino_acid",
        "consequence",
        "sequence",
    )
    forward = np.arange(states.height, dtype=np.float32)
    reverse = forward + 10
    result = build_response_table(
        states, {"forward": forward, "reverse_complement": reverse}
    )
    assert result.height == 2 * MUTATIONS_PER_CONTEXT
    assert set(result["baseline_activation"].unique()) == {0.0, 10.0}
    assert result.filter(pl.col("orientation") == "forward")["delta"].to_list() == list(
        np.arange(1, states.height, dtype=np.float32)
    )


def test_generic_focal_test_detects_center_sensitivity() -> None:
    rows = []
    for orientation in ("forward", "reverse_complement"):
        for context_index in range(40):
            for offset in (-1, 0, 1):
                for alternate_index in range(3):
                    rows.append(
                        {
                            "orientation": orientation,
                            "context_index": context_index,
                            "genomic_offset": offset,
                            "abs_delta": (
                                2.0 + context_index / 100 + alternate_index / 1000
                                if offset == 0
                                else 1.0 + context_index / 200 + alternate_index / 1000
                            ),
                        }
                    )
    observed = generic_focal_tests(pl.DataFrame(rows))
    assert observed.height == 2
    assert observed.select(
        (pl.col("mean_difference") > 0).all(),
        (pl.col("t_q") < 0.05).all(),
        (pl.col("rank_q") < 0.05).all(),
    ).row(0) == (True, True, True)


def test_codon_phase_and_consequence_tests_use_frozen_directions() -> None:
    contexts = pl.DataFrame(
        {
            "context_index": np.arange(384),
            "focal_codon_position": np.repeat([1, 2, 3], 128),
        }
    )
    phase_rows = []
    for orientation in ("forward", "reverse_complement"):
        for context_index, position in contexts.iter_rows():
            for alternate_index in range(3):
                phase_rows.append(
                    {
                        "orientation": orientation,
                        "context_index": context_index,
                        "genomic_offset": 0,
                        "abs_delta": (
                            (3.0 if position == 2 else 1.0)
                            + context_index / 10_000
                            + alternate_index / 100_000
                        ),
                    }
                )
    phase = codon_phase_tests(contexts, pl.DataFrame(phase_rows))
    assert phase.height == 4
    assert phase.select(
        (pl.col("mean_difference") > 0).all(),
        (pl.col("t_q") < 0.05).all(),
        (pl.col("rank_q") < 0.05).all(),
    ).row(0) == (True, True, True)

    consequence_rows = []
    for orientation in ("forward", "reverse_complement"):
        for context_index in range(40):
            consequence_rows.extend(
                [
                    {
                        "orientation": orientation,
                        "context_index": context_index,
                        "genomic_offset": 0,
                        "consequence": "synonymous",
                        "abs_delta": 1.0 + context_index / 100,
                    },
                    {
                        "orientation": orientation,
                        "context_index": context_index,
                        "genomic_offset": 0,
                        "consequence": "missense",
                        "abs_delta": 2.0 + context_index / 50,
                    },
                ]
            )
    consequence = consequence_tests(pl.DataFrame(consequence_rows))
    assert consequence.height == 2
    assert consequence.select(
        pl.col("minimum_pairs_met").all(),
        (pl.col("mean_difference") > 0).all(),
        (pl.col("t_q") < 0.05).all(),
        (pl.col("rank_q") < 0.05).all(),
    ).row(0) == (True, True, True, True)


def test_bh_with_missing_underpowered_test_is_null_safe() -> None:
    frame = pl.DataFrame({"p": [0.01, None, 0.04]})
    observed = add_bh(frame, p_column="p", q_column="q")
    assert observed["q"].to_list() == [0.02, None, 0.04]
    np.testing.assert_allclose(bh_adjust(np.array([0.01, 0.04])), [0.02, 0.04])
