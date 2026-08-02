from __future__ import annotations

import polars as pl

from build_panel import FOCAL_INDEX, WINDOW_BP
from interpret_feature1829 import (
    FEATURE_ID,
    NUCLEOTIDES,
    counterfactual_sequences,
    dense_feature,
    mutation_frame,
    select_contexts,
)


def _sequence(focal: str) -> str:
    return "A" * FOCAL_INDEX + focal + "C" * FOCAL_INDEX


def _panel() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "panel_row": [0, 1, 2],
            "chrom": ["1", "2", "3"],
            "pos": [101, 202, 303],
            "ref": ["A", "C", "G"],
            "alt": ["T", "G", "A"],
            "effect": [0.1, -0.2, 0.3],
            "official_split": ["train", "test", "train"],
            "ref_sequence": [_sequence("A"), _sequence("C"), _sequence("G")],
            "alt_sequence": [_sequence("T"), _sequence("G"), _sequence("A")],
        }
    )


def _sparse(ref_activation: list[float], alt_activation: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "panel_row": [0, 1],
            "feature_id": [FEATURE_ID, FEATURE_ID],
            "ref_activation": ref_activation,
            "alt_activation": alt_activation,
        }
    ).with_columns((pl.col("alt_activation") - pl.col("ref_activation")).alias("delta"))


def test_dense_feature_fills_sparse_absences_with_zero() -> None:
    dense = dense_feature(
        _panel(), _sparse([1.0, 4.0], [5.0, 2.0]), feature_id=FEATURE_ID
    )
    assert dense["ref_activation"].to_list() == [1.0, 4.0, 0.0]
    assert dense["alt_activation"].to_list() == [5.0, 2.0, 0.0]
    assert dense["delta"].to_list() == [4.0, -2.0, 0.0]


def test_select_contexts_uses_top_allele_once_per_variant() -> None:
    contexts = select_contexts(
        _panel(),
        {
            "forward": _sparse([1.0, 4.0], [5.0, 2.0]),
            "reverse_complement": _sparse([7.0, 3.0], [1.0, 9.0]),
        },
        feature_id=FEATURE_ID,
        contexts_per_orientation=2,
    )
    forward = contexts.filter(pl.col("orientation") == "forward")
    reverse = contexts.filter(pl.col("orientation") == "reverse_complement")
    assert forward["panel_row"].to_list() == [0, 1]
    assert forward["allele"].to_list() == ["alt", "ref"]
    assert forward["recorded_activation"].to_list() == [5.0, 4.0]
    assert reverse["panel_row"].to_list() == [1, 0]
    assert reverse["allele"].to_list() == ["alt", "ref"]
    assert reverse["recorded_activation"].to_list() == [9.0, 7.0]
    assert contexts["context_id"].n_unique() == contexts.height
    assert contexts.filter(
        pl.col("input_sequence").str.len_chars() != WINDOW_BP
    ).is_empty()


def test_counterfactual_sequences_include_one_noop_per_position() -> None:
    sequence = _sequence("G")
    rows = counterfactual_sequences(sequence, radius=2)
    assert len(rows) == 3 * len(NUCLEOTIDES)
    assert sum(not row["changed"] for row in rows) == 3
    assert {row["relative_position_input"] for row in rows} == {-2, -1, 0}
    assert {row["target_base"] for row in rows} == set(NUCLEOTIDES)
    assert all(len(row["input_sequence"]) == WINDOW_BP for row in rows)


def test_mutation_frame_preserves_context_metadata() -> None:
    contexts = select_contexts(
        _panel(),
        {
            "forward": _sparse([1.0, 4.0], [5.0, 2.0]),
            "reverse_complement": _sparse([7.0, 3.0], [1.0, 9.0]),
        },
        feature_id=FEATURE_ID,
        contexts_per_orientation=1,
    )
    mutations = mutation_frame(contexts, radius=1)
    assert mutations.height == contexts.height * 2 * len(NUCLEOTIDES)
    assert mutations["context_id"].n_unique() == contexts.height
    assert mutations.group_by("context_id").len()["len"].to_list() == [8, 8]
    assert mutations.filter(~pl.col("changed")).height == contexts.height * 2
