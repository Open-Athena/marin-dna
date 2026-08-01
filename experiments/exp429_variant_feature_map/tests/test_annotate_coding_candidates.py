from __future__ import annotations

import polars as pl

from annotate_coding_candidates import prepare_contexts, summarize_annotations


def test_prepare_contexts_maps_class_to_consequence_schema() -> None:
    frame = pl.DataFrame(
        {
            "class": ["stop_gained", "synonymous_variant", "missense_variant"],
            "panel_row": [0, 1, 2],
        }
    )

    prepared = prepare_contexts(frame)

    assert prepared["panel_row"].to_list() == [0, 1]
    assert prepared["consequence_cre"].to_list() == [
        "stop_gained",
        "synonymous_variant",
    ]


def test_summarize_annotations_preserves_top_and_codon_phase() -> None:
    frame = pl.DataFrame(
        {
            "class": ["stop_gained", "stop_gained", "stop_gained"],
            "is_top": [True, False, False],
            "matching_transcript_count": [1, 1, 0],
            "consensus_strand": ["+", "-", None],
            "consensus_codon_position": [1, 3, None],
            "consensus_ref_codon": ["CAA", "TGG", None],
            "consensus_alt_codon": ["TAA", "TAG", None],
            "transcript_substitution": ["C>T", "G>A", None],
        }
    )

    summary, positions, codons = summarize_annotations(frame)

    top = summary.filter(pl.col("subset") == "top").row(0, named=True)
    remainder = summary.filter(pl.col("subset") == "remainder").row(0, named=True)
    assert top["matching_consequence_fraction"] == 1
    assert remainder["matching_consequence_fraction"] == 0.5
    assert set(positions["codon_position"]) == {1, 3}
    assert set(codons["alt_codon"]) == {"TAA", "TAG"}
