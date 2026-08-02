from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from analyze_repeat_saturation import (
    build_context_effects,
    build_feature_summary,
    build_planned_tests,
    build_view_summary,
    one_sided_p_values,
    plot_response_heatmaps,
)
from analyze_repeat_saturation_causal_window import select_causal_window
from saturation_common import VIEW_KEYS


def test_context_effects_match_neutral_edits_by_substitution() -> None:
    common = {
        "block": 10,
        "feature_id": 6903,
        "orientation": "forward",
        "saturation_context_id": 0,
        "relative_delta": -0.1,
        "thresholded_to_zero": False,
        "net_kmers_lost": 1,
    }
    responses = pl.DataFrame(
        [
            {
                **common,
                "motif_loss": True,
                "neutral": False,
                "model_ref": "A",
                "model_alt": "C",
                "delta": -2.0,
            },
            {
                **common,
                "motif_loss": True,
                "neutral": False,
                "model_ref": "A",
                "model_alt": "G",
                "delta": -1.0,
            },
            {
                **common,
                "motif_loss": False,
                "neutral": True,
                "model_ref": "A",
                "model_alt": "C",
                "delta": -0.5,
            },
            {
                **common,
                "motif_loss": False,
                "neutral": True,
                "model_ref": "A",
                "model_alt": "G",
                "delta": -0.2,
            },
        ]
    )
    observed = build_context_effects(responses)
    assert observed.height == 1
    assert observed["mean_motif_delta"].item() == -1.5
    np.testing.assert_allclose(observed["specificity_contrast"].item(), -1.15)
    assert observed["matched_motif_edits"].item() == 2


def test_one_sided_tests_and_layerwise_success_tables() -> None:
    t_p, rank_p = one_sided_p_values(np.linspace(-2.0, -0.1, 40))
    assert t_p < 0.05 and rank_p < 0.05
    rows: list[dict[str, object]] = []
    for block, feature_id, orientation in VIEW_KEYS:
        for context in range(40):
            rows.append(
                {
                    "block": block,
                    "feature_id": feature_id,
                    "orientation": orientation,
                    "saturation_context_id": context,
                    "motif_edits": 3,
                    "mean_motif_delta": -1.0 - context / 100,
                    "median_motif_delta": -1.0,
                    "mean_motif_relative_delta": -0.2,
                    "median_motif_relative_delta": -0.2,
                    "motif_zero_fraction": 0.1,
                    "mean_net_kmers_lost": 2.0,
                    "matched_motif_edits": 3,
                    "specificity_contrast": -0.5 - context / 100,
                    "relative_specificity_contrast": -0.1,
                    "available_neutral_edits": 9,
                }
            )
    effects = pl.DataFrame(rows)
    tests = build_planned_tests(effects)
    assert tests.height == len(VIEW_KEYS) * 2
    assert tests["t_q"].max() < 0.05
    assert tests["rank_q"].max() < 0.05
    views = build_view_summary(effects, tests)
    assert views["motif_loss_supported"].all()
    assert views["motif_specificity_supported"].all()
    features = build_feature_summary(views)
    assert features.height == len(VIEW_KEYS) // 2
    assert features["strand_stable_motif_loss"].all()
    assert features["strand_stable_motif_specificity"].all()


def test_heatmaps_emit_png_and_svg_for_every_feature(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for block, feature_id, orientation in VIEW_KEYS:
        rows.append(
            {
                "block": block,
                "feature_id": feature_id,
                "orientation": orientation,
                "model_offset": 0,
                "substitution": "A>C",
                "mean_relative_delta": -0.2,
            }
        )
    paths = plot_response_heatmaps(pl.DataFrame(rows), tmp_path)
    assert len(paths) == len(VIEW_KEYS)
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)


def test_causal_window_requires_exact_downstream_invariance() -> None:
    frame = pl.DataFrame(
        {
            "model_offset": [-1, 0, 1],
            "abs_delta": [2.0, 1.0, 0.0],
            "orientation": ["forward"] * 3,
            "feature_id": [7] * 3,
            "block": [10] * 3,
        }
    )
    observed = select_causal_window(frame)
    assert observed["model_offset"].to_list() == [-1, 0]
    invalid = frame.with_columns(
        pl.when(pl.col("model_offset") == 1)
        .then(0.1)
        .otherwise(pl.col("abs_delta"))
        .alias("abs_delta")
    )
    with pytest.raises(AssertionError):
        select_causal_window(invalid)
