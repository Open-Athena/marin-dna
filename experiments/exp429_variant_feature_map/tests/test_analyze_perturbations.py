from __future__ import annotations

import numpy as np
import polars as pl

from analyze_perturbations import (
    candidate_response,
    codon_context_selectivity,
    summarize_selectivity,
)


def test_candidate_response_aligns_rc_and_uses_local_max() -> None:
    forward = np.array([[0.0, 2.0, 0.0]])
    reverse = np.array([[4.0, 0.0, 0.0]])

    score, peak = candidate_response(
        forward,
        reverse,
        orientation="max_absolute",
        transform="signed",
        direction=1,
        spatial_metric="local_max",
    )

    assert score.tolist() == [4.0]
    assert peak.tolist() == [1]


def test_codon_selectivity_is_within_context() -> None:
    frame = pl.DataFrame(
        {
            "perturbation_type": ["codon_sweep"] * 8,
            "analysis_feature_id": [3312] * 8,
            "response_role": ["primary"] * 8,
            "class": ["stop_gained"] * 8,
            "context_group": ["top"] * 8,
            "source_panel_row": [1] * 4 + [2] * 4,
            "expected_consequence": [
                "stop_gained",
                "stop_gained",
                "missense_variant",
                "synonymous_variant",
            ]
            * 2,
            "response_score": [4.0, 6.0, 1.0, 3.0, 8.0, 10.0, 2.0, 4.0],
            "edit_distance": [1, 1, 1, 2, 1, 1, 1, 2],
        }
    )

    contexts = codon_context_selectivity(frame)
    summary = summarize_selectivity(contexts, bootstrap_samples=100)

    assert contexts["target_minus_other"].to_list() == [3.0, 6.0]
    assert summary["mean_target_minus_other"].item() == 4.5
    assert summary["contexts"].item() == 2

    one_edit = codon_context_selectivity(frame, edit_distance=1)
    assert one_edit["target_minus_other"].to_list() == [4.0, 7.0]
