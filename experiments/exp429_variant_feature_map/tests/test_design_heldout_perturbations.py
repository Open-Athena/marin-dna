import polars as pl

import design_heldout_perturbations as module
from design_heldout_perturbations import (
    Intron,
    deterministic_sources,
    one_edit_codon_rows,
    parse_gtf_attributes,
    splice_class_positions,
    splice_strand_index,
)


def test_parse_gtf_attributes() -> None:
    assert parse_gtf_attributes(
        'gene_id "g1"; transcript_id "t1"; transcript_biotype "protein_coding";'
    ) == {
        "gene_id": "g1",
        "transcript_id": "t1",
        "transcript_biotype": "protein_coding",
    }


def test_splice_positions_follow_transcript_orientation() -> None:
    plus = Intron("21", 100, 200, "+")
    minus = Intron("21", 300, 400, "-")
    assert splice_class_positions(plus, "splice_acceptor_variant") == (198, 199)
    assert splice_class_positions(minus, "splice_acceptor_variant") == (300, 301)
    assert splice_class_positions(plus, "splice_donor_5th_base_variant") == (104,)
    assert splice_class_positions(minus, "splice_donor_5th_base_variant") == (395,)
    index = splice_strand_index([plus, minus], class_name="splice_acceptor_variant")
    assert index == {198: {"+"}, 199: {"+"}, 300: {"-"}, 301: {"-"}}


def test_deterministic_sources_uses_hash_not_input_order() -> None:
    classes = (
        "splice_acceptor_variant",
        "splice_donor_5th_base_variant",
        "stop_gained",
        "synonymous_variant",
    )
    rows = []
    panel_row = 0
    for class_name in classes:
        for sample_hash in (30, 10, 20):
            rows.append(
                {
                    "panel_row": panel_row,
                    "consequence_cre": class_name,
                    "sample_hash": sample_hash,
                    "chrom": "21",
                    "pos": panel_row + 1,
                    "ref": "A",
                    "alt": "C",
                }
            )
            panel_row += 1
    frame = pl.DataFrame(rows)
    expected = deterministic_sources(frame, contexts_per_class=2)
    reversed_result = deterministic_sources(frame.reverse(), contexts_per_class=2)
    assert expected.equals(reversed_result)
    assert expected.group_by("consequence_cre").agg(pl.col("sample_hash").sort())[
        "sample_hash"
    ].to_list() == [[10, 20]] * len(classes)


def test_one_edit_codon_rows_are_nine_single_edits() -> None:
    sequence = "A" * 126 + "GCT" + "A" * 126
    assert len(sequence) == 255
    source = {
        "consequence_cre": "synonymous_variant",
        "consensus_strand": "+",
        "consensus_codon_position": 2,
        "consensus_ref_codon": "GCT",
        "panel_row": 7,
        "source_rank": 1,
        "chrom": "21",
        "pos": 128,
    }
    rows = one_edit_codon_rows(
        source,
        sequence,
        window_start0=0,
        window_end0=255,
    )
    assert len(rows) == 9
    assert {row["edit_distance"] for row in rows} == {1}
    assert {row["reference_codon"] for row in rows} == {"GCT"}
    assert len({row["alternate_codon"] for row in rows}) == 9
    assert all(
        sum(a != b for a, b in zip("GCT", row["alternate_codon"], strict=True)) == 1
        for row in rows
    )


def test_reference_codon_filter_rejects_mismatched_annotation(monkeypatch) -> None:
    frame = pl.DataFrame(
        {
            "panel_row": [1, 2],
            "consensus_strand": ["+", "+"],
            "consensus_codon_position": [2, 2],
            "consensus_ref_codon": ["GCT", "AAA"],
        }
    )
    sequence = "A" * 126 + "GCT" + "A" * 126
    monkeypatch.setattr(
        module,
        "reference_window",
        lambda genome, source: (sequence, 0, 255),
    )
    filtered, rejected = module.filter_coding_reference_matches(frame, genome=object())
    assert rejected == 1
    assert filtered["panel_row"].to_list() == [1]


def test_intron_ordering_is_deterministic() -> None:
    assert sorted([Intron("21", 20, 30, "-"), Intron("21", 10, 20, "+")]) == [
        Intron("21", 10, 20, "+"),
        Intron("21", 20, 30, "-"),
    ]
