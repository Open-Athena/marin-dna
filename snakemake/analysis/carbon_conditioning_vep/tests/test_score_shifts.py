import pandas as pd
import pytest
from marin_dna_carbon_conditioning_vep.score_shifts import (
    assemble_score_shifts,
    bootstrap_matched_score_shifts,
    summarize_score_shifts,
)


def _score_frame(scores: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "variant_id": [f"v{index}" for index in range(6)],
            "subset": ["subset_a"] * 6,
            "match_group": [1, 1, 1, 2, 2, 2],
            "label": [True, False, False, True, False, False],
            "score": scores,
        }
    )


def test_score_shift_summaries_preserve_alignment_and_matching() -> None:
    untagged = _score_frame([0.0] * 6)
    correct = _score_frame([1.0, 0.0, 2.0, 3.0, 1.0, 3.0])
    correct = correct.iloc[[4, 1, 5, 0, 3, 2]].reset_index(drop=True)

    shifts = assemble_score_shifts(untagged, {"correct": correct})
    summary = summarize_score_shifts(shifts)
    matched = bootstrap_matched_score_shifts(
        shifts,
        n_bootstrap=100,
        bootstrap_seed=486,
        min_groups=3,
    )

    assert len(shifts) == 6
    assert summary.set_index("label")["n_variants"].to_dict() == {
        False: 4,
        True: 2,
    }
    subset = matched.loc[matched["subset"].eq("subset_a")].iloc[0]
    assert subset["n_groups"] == 2
    assert subset["n_negative"] == 4
    assert subset["mean_delta_positive"] == pytest.approx(2.0)
    assert subset["mean_delta_negative"] == pytest.approx(1.5)
    assert subset["label_separation_shift"] == pytest.approx(0.5)
    assert bool(subset["low_sample"])


def test_score_shift_alignment_rejects_label_mismatch() -> None:
    untagged = _score_frame([0.0] * 6)
    wrong = _score_frame([0.0] * 6)
    wrong.loc[0, "label"] = False

    with pytest.raises(AssertionError, match="alignment"):
        assemble_score_shifts(untagged, {"wrong": wrong})
