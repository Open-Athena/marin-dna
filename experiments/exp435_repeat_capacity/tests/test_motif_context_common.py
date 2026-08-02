from __future__ import annotations

import numpy as np
import polars as pl

from motif_context_common import (
    bh_adjust,
    kmer_enrichment,
    match_controls,
    positional_enrichment,
    reverse_complement,
    select_top_contexts,
    sequence_consensus,
)


def test_reverse_complement() -> None:
    assert reverse_complement("ACGTTT") == "AAACGT"


def test_bh_adjust_preserves_order_and_monotonicity() -> None:
    observed = bh_adjust(np.asarray([0.04, 0.001, 0.03, 0.9]))
    np.testing.assert_allclose(observed, [0.0533333333, 0.004, 0.0533333333, 0.9])


def test_select_top_contexts_uses_activation_then_context_id() -> None:
    frame = pl.DataFrame(
        {
            "context_id": [9, 3, 2, 1],
            "feature_id": [7, 7, 7, 8],
            "activation": [2.0, 2.0, 1.0, 10.0],
        }
    )
    observed = select_top_contexts(frame, feature_id=7, limit=2)
    assert observed["context_id"].to_list() == [3, 9]


def test_match_controls_is_unique_and_prefers_exact_cell() -> None:
    contexts = pl.DataFrame(
        {
            "context_id": [0, 1, 2, 3, 4, 5],
            "chrom": ["1", "1", "1", "2", "1", "1"],
            "is_repeat": [True, True, True, True, False, False],
            "repeat_class": ["SINE", "SINE", "LINE", "SINE", None, None],
            "gc_bin": [3, 3, 3, 3, 3, 3],
        }
    )
    observed = match_controls(contexts, [0, 2], namespace="test")
    assert observed.height == 2
    assert observed["control_context_id"].n_unique() == 2
    first = observed.filter(pl.col("top_context_id") == 0)
    assert first["control_context_id"].item() == 1
    assert first["match_level"].item() == "chrom_class_gc"


def test_position_and_kmer_enrichment_recover_a_rich_signal() -> None:
    top = ["CAAAAAAGC", "GAAAAAATC", "TAAAAAACC", "CAAAAAATG"] * 8
    control = ["CCCCCCCGC", "GCCCCCCAC", "TCCCCCCCC", "CCCCCCCTG"] * 8
    position = positional_enrichment(top, control, radius=3)
    focal_a = position.filter((pl.col("offset") == 0) & (pl.col("base") == "A"))
    assert focal_a["log2_odds"].item() > 5
    assert focal_a["q_value"].item() < 0.05
    consensus = sequence_consensus(position)
    assert consensus[3] == "A"

    kmers = kmer_enrichment(
        [sequence[1:8] for sequence in top],
        [sequence[1:8] for sequence in control],
    )
    aaa = kmers.filter(pl.col("kmer") == "AAA")
    assert aaa.height == 1
    assert aaa["log2_odds"].item() > 5
    assert aaa["q_value"].item() < 0.05
