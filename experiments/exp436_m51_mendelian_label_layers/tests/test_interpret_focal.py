from __future__ import annotations

import math

import numpy as np
import polars as pl

from interpret_focal import (
    densify_candidates,
    reverse_complement,
    score_correlations,
    substitution_summary,
)


def test_densify_candidates_fills_inactive_slots_with_zero() -> None:
    sparse = pl.DataFrame(
        {
            "panel_row": np.asarray([0, 2], dtype=np.uint32),
            "feature_id": np.asarray([7, 8], dtype=np.uint32),
            "ref_activation": np.asarray([1.0, 0.0], dtype=np.float32),
            "alt_activation": np.asarray([3.0, 4.0], dtype=np.float32),
            "delta": np.asarray([2.0, 4.0], dtype=np.float32),
        }
    )
    dense = densify_candidates(sparse, rows=3, candidates=(7, 8))
    assert dense.height == 6
    inactive = dense.filter(
        (pl.col("panel_row") == 1) & (pl.col("feature_id") == 7)
    ).row(0, named=True)
    assert inactive["ref_activation"] == 0
    assert inactive["alt_activation"] == 0
    assert inactive["delta"] == 0
    assert inactive["abs_delta"] == 0


def synthetic_responses() -> pl.DataFrame:
    rows = []
    for feature_id in (7, 8):
        for panel_row in range(20):
            label = panel_row % 2
            delta = float(panel_row + feature_id)
            if feature_id == 8:
                delta = -delta
            rows.append(
                {
                    "arm": "block19-25m",
                    "block": 19,
                    "budget": 25_000_200,
                    "orientation": "forward",
                    "feature_id": feature_id,
                    "panel_row": panel_row,
                    "label": label,
                    "subset": "splicing" if panel_row < 10 else "distal",
                    "substitution": "A>G",
                    "ref_activation": 0.0,
                    "alt_activation": delta,
                    "delta": delta,
                    "abs_delta": abs(delta),
                    "minus_llr_avg": 2 * delta,
                    "probe_score": -delta,
                }
            )
    return pl.DataFrame(rows)


def test_score_correlations_reports_overall_and_subsets_with_bh() -> None:
    result = score_correlations(synthetic_responses())
    assert set(result["target"].unique().to_list()) == {
        "overall",
        "splicing",
        "distal",
    }
    expected_rows = 2 * 3 * 2 * 2
    assert result.height == expected_rows
    positive = result.filter(
        (pl.col("feature_id") == 7)
        & (pl.col("target") == "overall")
        & (pl.col("response") == "delta")
        & (pl.col("outcome") == "minus_llr_avg")
    ).row(0, named=True)
    assert math.isclose(positive["pearson_r"], 1)
    assert math.isclose(positive["spearman_rho"], 1)
    assert 0 <= positive["pearson_q"] <= 1
    assert 0 <= positive["spearman_q"] <= 1


def test_substitution_summary_preserves_label_contrast() -> None:
    summary = substitution_summary(synthetic_responses())
    overall = summary.filter(
        (pl.col("target") == "overall") & (pl.col("feature_id") == 7)
    ).row(0, named=True)
    assert overall["n"] == 20
    assert overall["positives"] == 10
    assert math.isclose(overall["label_delta_difference"], 1.0)
    assert reverse_complement("ACGT") == "ACGT"
