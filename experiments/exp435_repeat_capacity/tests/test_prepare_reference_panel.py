from __future__ import annotations

import numpy as np
import polars as pl

from panel_common import selection_hash, stable_seed
from prepare_reference_panel import (
    annotate_points,
    gc_bin,
    match_rows,
    merge_intervals,
    repeat_free_focal_intervals,
    sample_interval_positions,
    sequence_metrics,
)


def annotations() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "annotation_id": [0, 1, 2],
            "chrom": ["1", "1", "1"],
            "start0": [10, 15, 40],
            "end0": [30, 25, 50],
            "sw_score": [100, 200, 100],
            "milli_div": [10, 20, 10],
            "strand": ["+", "-", "+"],
            "repeat_name": ["A", "B", "C"],
            "repeat_class": ["LINE", "SINE", "LTR"],
            "repeat_family": ["L1", "Alu", "ERV1"],
            "family_label": ["LINE|L1", "SINE|Alu", "LTR|ERV1"],
            "subfamily_label": ["LINE|L1|A", "SINE|Alu|B", "LTR|ERV1|C"],
        }
    )


def test_union_sampling_and_repeat_free_intervals_are_deterministic() -> None:
    merged = merge_intervals(annotations())
    assert merged.rows() == [("1", 10, 30), ("1", 40, 50)]
    first = sample_interval_positions(merged, 12, namespace="test")
    second = sample_interval_positions(merged, 12, namespace="test")
    assert first == second and len(set(first)) == 12
    free = repeat_free_focal_intervals(
        merged, {"1": 400, **{str(i): 1 for i in range(2, 23)}, "X": 1, "Y": 1}
    )
    chromosome_one = free.filter(pl.col("chrom") == "1")
    assert chromosome_one.rows() == [("1", 177, 273)]


def test_overlap_primary_policy_and_metrics() -> None:
    result = annotate_points([("1", 12), ("1", 20), ("1", 35)], annotations())
    assert result[("1", 12)]["repeat_name"] == "A"
    assert result[("1", 20)]["repeat_name"] == "B"
    assert result[("1", 20)]["overlap_count"] == 2
    assert result[("1", 35)] is None
    metrics = sequence_metrics("A" * 100 + "C" * 55 + "G" * 50 + "T" * 50)
    assert metrics["gc_count"] == 105
    assert metrics["cpg_count"] == 1
    assert metrics["max_homopolymer"] == 100
    assert 0 < metrics["shannon_entropy"] <= 2


def test_matching_uses_chromosome_and_gc_bin() -> None:
    positives = [
        {"chrom": "1", "pos0": 10, "gc_bin": 2, "repeat_class": "LINE"},
        {"chrom": "1", "pos0": 20, "gc_bin": 2, "repeat_class": "LINE"},
    ]
    candidates = [
        {"chrom": "1", "pos0": 30, "gc_bin": 2, "repeat_class": "SINE"},
        {"chrom": "1", "pos0": 40, "gc_bin": 2, "repeat_class": "LTR"},
        {"chrom": "2", "pos0": 50, "gc_bin": 2, "repeat_class": "SINE"},
    ]
    matched = match_rows(
        positives,
        candidates,
        namespace="match",
        different_level="class",
        different_label="LINE",
    )
    assert {(row["chrom"], row["pos0"]) for row in matched} == {("1", 30), ("1", 40)}
    assert stable_seed("x") == stable_seed("x")
    assert selection_hash("x", "1", 10) == selection_hash("x", "1", 10)
    assert gc_bin(50, list(range(0, 110, 10))) == 5
    assert np.isfinite(stable_seed("x"))
