from __future__ import annotations

import math

import polars as pl

from summarize_focal import strand_overlap, target_summary


def synthetic_frame() -> pl.DataFrame:
    rows = []
    for orientation, features in (
        ("forward", [(1, 0.01, 0.01, 0.4), (2, 0.01, 0.2, 0.2), (4, 0.8, 0.9, -0.1)]),
        (
            "reverse_complement",
            [(2, 0.01, 0.01, 0.3), (3, 0.2, 0.01, -0.2), (4, 0.7, 0.8, -0.2)],
        ),
    ):
        for feature_id, welch_q, mann_q, effect in features:
            rows.append(
                {
                    "arm": "block19-25m",
                    "block": 19,
                    "budget": 25_000_200,
                    "orientation": orientation,
                    "response": "delta",
                    "target_kind": "overall",
                    "target": "overall",
                    "feature_id": feature_id,
                    "welch_q": welch_q,
                    "mann_whitney_q": mann_q,
                    "minimum_q": min(welch_q, mann_q),
                    "best_auprc": 0.2 + feature_id / 100,
                    "prevalence": 0.1,
                    "rank_biserial": effect,
                }
            )
    return pl.DataFrame(rows)


def test_target_summary_counts_union_and_both() -> None:
    summary = target_summary(synthetic_frame()).sort("orientation")
    forward = summary.row(0, named=True)
    assert forward["eligible_features"] == 3
    assert forward["welch_discoveries"] == 2
    assert forward["mann_whitney_discoveries"] == 1
    assert forward["both_test_discoveries"] == 1
    assert forward["union_discoveries"] == 2
    assert math.isclose(forward["best_auprc"], 0.24)


def test_strand_overlap_uses_same_feature_ids_and_effects() -> None:
    overlap = strand_overlap(synthetic_frame()).row(0, named=True)
    assert overlap["left_significant"] == 2
    assert overlap["right_significant"] == 2
    assert overlap["significant_overlap"] == 1
    assert overlap["significant_union"] == 3
    assert overlap["significant_jaccard"] == 1 / 3
    assert overlap["shared_eligible_features"] == 2
    assert overlap["effect_sign_concordance_overlap"] == 1
    assert math.isclose(overlap["effect_spearman_shared_eligible"], 1.0)
