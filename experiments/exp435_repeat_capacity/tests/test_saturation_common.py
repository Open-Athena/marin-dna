from __future__ import annotations

import polars as pl

from extract_common import FOCAL_INDEX, ORIENTATIONS, WINDOW_BP
from saturation_common import (
    MUTATIONS_PER_CONTEXT,
    STATES_PER_CONTEXT,
    VIEW_KEYS,
    build_state_table,
    enumerate_context_states,
    kmer_occurrence_counts,
    qualifying_kmer_sets,
    select_contexts,
)


def test_select_contexts_uses_frozen_order_per_view() -> None:
    rows: list[dict[str, object]] = []
    for block, feature_id, orientation in VIEW_KEYS:
        for index in range(66):
            rows.append(
                {
                    "context_id": 1000 * feature_id + index,
                    "chrom": "1",
                    "pos0": index,
                    "activation": float(index // 2),
                    "role": "top",
                    "block": block,
                    "arm": f"block{block:02d}-25m",
                    "feature_id": feature_id,
                    "orientation": orientation,
                    "model_sequence": "A" * WINDOW_BP,
                }
            )
    observed = select_contexts(pl.DataFrame(rows))
    assert observed.height == len(VIEW_KEYS) * 64
    first = observed.filter(
        (pl.col("block") == VIEW_KEYS[0][0])
        & (pl.col("feature_id") == VIEW_KEYS[0][1])
        & (pl.col("orientation") == VIEW_KEYS[0][2])
    )
    assert first["pos0"].head(4).to_list() == [64, 65, 62, 63]


def test_qualifying_kmers_requires_every_frozen_view() -> None:
    rows = [
        {
            "block": block,
            "feature_id": feature_id,
            "orientation": orientation,
            "kmer": "ACG",
            "q_value": 0.01,
            "log2_odds": 1.0,
        }
        for block, feature_id, orientation in VIEW_KEYS
    ]
    rows.append(
        {
            "block": VIEW_KEYS[0][0],
            "feature_id": VIEW_KEYS[0][1],
            "orientation": VIEW_KEYS[0][2],
            "kmer": "TTT",
            "q_value": 0.2,
            "log2_odds": 5.0,
        }
    )
    observed = qualifying_kmer_sets(pl.DataFrame(rows))
    assert set(observed) == set(VIEW_KEYS)
    assert all(value == frozenset({"ACG"}) for value in observed.values())


def test_occurrence_counts_and_state_enumeration_detect_repeat_loss() -> None:
    motif = "AC" * 31 + "A"
    sequence = "G" * (FOCAL_INDEX - 31) + motif + "G" * (WINDOW_BP - FOCAL_INDEX - 32)
    assert len(sequence) == WINDOW_BP
    dictionary = frozenset({"ACA", "CAC", "ACAC", "CACA"})
    counts = kmer_occurrence_counts(motif, dictionary)
    assert counts["ACA"] > 1 and counts["CAC"] > 1
    row = {
        "saturation_context_id": 0,
        "block": 10,
        "arm": "block10-25m",
        "feature_id": 6903,
        "orientation": "forward",
        "model_sequence": sequence,
    }
    states = enumerate_context_states(row, dictionary)
    assert len(states) == STATES_PER_CONTEXT
    assert sum(state["motif_loss"] for state in states) > 0
    assert sum(not state["is_baseline"] for state in states) == MUTATIONS_PER_CONTEXT
    center = [
        state
        for state in states
        if state["model_offset"] == 0 and state["model_alt"] != state["model_ref"]
    ]
    assert len(center) == 3
    assert all(state["reference_offset"] == 0 for state in center)


def test_reverse_complement_offsets_and_state_indices() -> None:
    sequence = "A" * WINDOW_BP
    contexts = pl.DataFrame(
        {
            "saturation_context_id": [0],
            "block": [1],
            "arm": ["block01-25m"],
            "feature_id": [10488],
            "orientation": [ORIENTATIONS[1]],
            "model_sequence": [sequence],
        }
    )
    states = build_state_table(
        contexts, {(1, 10488, ORIENTATIONS[1]): frozenset({"AAA"})}
    )
    assert states.height == STATES_PER_CONTEXT
    mutation = states.filter(
        (pl.col("model_offset") == 7) & (pl.col("model_alt") == "C")
    )
    assert mutation["reference_offset"].item() == -7
    assert mutation["baseline_state_index"].item() == 0
