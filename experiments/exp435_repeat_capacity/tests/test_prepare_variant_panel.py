from __future__ import annotations

import polars as pl

from prepare_reference_panel import add_hierarchy_labels, merge_intervals
from prepare_variant_panel import annotate_variant_panel, summarize_categories


def synthetic_annotations() -> pl.DataFrame:
    return add_hierarchy_labels(
        pl.DataFrame(
            {
                "annotation_id": pl.Series([0, 1], dtype=pl.UInt32),
                "chrom": ["1", "1"],
                "start0": [200, 205],
                "end0": [230, 220],
                "sw_score": [100, 200],
                "milli_div": [10, 20],
                "strand": ["+", "-"],
                "repeat_name": ["L1A", "AluA"],
                "repeat_class": ["LINE", "SINE"],
                "repeat_family": ["L1", "Alu"],
                "record_id": [10, 11],
            }
        )
    )


def test_variant_annotation_distinguishes_focal_near_and_repeat_free() -> None:
    annotations = synthetic_annotations()
    panel = pl.DataFrame(
        {
            "chrom": ["1", "1", "1"],
            "pos": [211, 261, 501],
            "ref": ["A", "C", "G"],
            "alt": ["T", "G", "A"],
            "subset": ["distal", "distal", "distal"],
            "match_group": [1, 2, 3],
            "split": ["test", "test", "test"],
        }
    )
    result = annotate_variant_panel(
        panel,
        annotations,
        merge_intervals(annotations),
        expected_rows=3,
    )

    assert result["position_status"].to_list() == [
        "focal_repeat",
        "near_repeat",
        "repeat_free_window",
    ]
    focal = result.row(0, named=True)
    assert focal["repeat_class"] == "SINE"
    assert focal["subfamily_label"] == "SINE|Alu|AluA"
    assert focal["overlap_count"] == 2
    assert focal["overlap_annotation_ids"] == [0, 1]
    assert focal["repeat_covered_bp"] == 30
    assert not focal["unique_repeat_overlap"]
    assert not focal["repeat_interior_32"]
    assert result.row(1, named=True)["annotation_id"] is None
    assert result.row(1, named=True)["repeat_covered_bp"] == 30
    assert result.row(2, named=True)["repeat_covered_bp"] == 0
    assert "label" not in result.columns


def test_category_summary_uses_frozen_32_variant_threshold() -> None:
    rows = []
    for index in range(33):
        rows.append(
            {
                "position_status": "focal_repeat",
                "repeat_class": "LINE",
                "family_label": "LINE|L1",
                "subfamily_label": "LINE|L1|L1A",
                "subset": "distal",
                "chrom": "1",
                "repeat_interior_32": index < 10,
                "unique_repeat_overlap": index < 20,
            }
        )
    rows.append(
        {
            "position_status": "focal_repeat",
            "repeat_class": "SINE",
            "family_label": "SINE|Alu",
            "subfamily_label": "SINE|Alu|AluA",
            "subset": "distal",
            "chrom": "1",
            "repeat_interior_32": False,
            "unique_repeat_overlap": True,
        }
    )

    summary = summarize_categories(pl.DataFrame(rows))
    assert summary.filter(pl.col("category") == "LINE")["eligible_32"].item()
    assert not summary.filter(pl.col("category") == "SINE")["eligible_32"].item()
    assert (
        summary.filter(pl.col("category") == "LINE")["interior_32_variants"].item()
        == 10
    )
