import numpy as np
import pandas as pd
import pytest
from marin_dna_carbon_conditioning_vep.metrics import (
    MACRO_SUBSET,
    compute_absolute_auprc,
    compute_paired_auprc_deltas,
)


def _score_frame(condition: str) -> pd.DataFrame:
    rows = []
    group = 0
    for subset, n_groups in (("large", 3), ("small", 1)):
        for _ in range(n_groups):
            group += 1
            for member in range(10):
                label = member == 0
                correct_score = 1.0 if label else member / 100.0
                baseline_score = member / 100.0
                rows.append(
                    {
                        "variant_id": f"1:{group * 20 + member}:A>C",
                        "label": label,
                        "subset": subset,
                        "match_group": group,
                        "score": correct_score
                        if condition == "correct"
                        else baseline_score,
                    }
                )
    return pd.DataFrame(rows)


def test_absolute_metrics_report_low_sample_and_exclude_it_from_macro() -> None:
    metrics = compute_absolute_auprc(
        _score_frame("correct"),
        condition="correct",
        n_bootstrap=20,
        bootstrap_seed=486,
        min_groups_for_macro=2,
    )
    small = metrics.loc[metrics["subset"] == "small"].iloc[0]
    macro = metrics.loc[metrics["subset"] == MACRO_SUBSET].iloc[0]
    large = metrics.loc[metrics["subset"] == "large"].iloc[0]
    assert bool(small["low_sample"])
    assert not bool(small["macro_eligible"])
    assert macro["auprc"] == pytest.approx(large["auprc"])
    assert int(macro["n_subsets"]) == 1


def test_paired_bootstrap_uses_identical_rows_and_macro_draws() -> None:
    correct = _score_frame("correct")
    baseline = _score_frame("baseline")
    deltas = compute_paired_auprc_deltas(
        correct,
        baseline,
        comparison="correct_minus_untagged",
        condition_a="correct",
        condition_b="untagged",
        n_bootstrap=30,
        bootstrap_seed=486,
        min_groups_for_macro=2,
    )
    large = deltas.loc[deltas["subset"] == "large"].iloc[0]
    macro = deltas.loc[deltas["subset"] == MACRO_SUBSET].iloc[0]
    assert large["delta"] > 0
    assert macro["delta"] == pytest.approx(large["delta"])
    assert np.isfinite([macro["ci_low"], macro["ci_high"]]).all()


def test_identical_scores_have_zero_paired_distribution() -> None:
    frame = _score_frame("correct")
    deltas = compute_paired_auprc_deltas(
        frame,
        frame.copy(),
        comparison="same",
        condition_a="correct",
        condition_b="correct",
        n_bootstrap=20,
        bootstrap_seed=486,
        min_groups_for_macro=2,
    )
    assert (deltas["delta"] == 0).all()
    assert (deltas["ci_low"] == 0).all()
    assert (deltas["ci_high"] == 0).all()
