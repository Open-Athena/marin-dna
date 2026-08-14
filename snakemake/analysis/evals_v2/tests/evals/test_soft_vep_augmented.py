"""Offline tests for the exp351-centered replacement assessment."""

import pandas as pd
import pytest

from marin_dna_evals.soft_vep_analysis import (
    compare_metric_detection_timing,
    persistent_specialist_detectability,
    specialist_detectability_summary,
)
from marin_dna_evals.soft_vep_augmented import (
    AUGMENTED_ARMS,
    AUGMENTED_SPECIALIST_ARM,
    AUGMENTED_STEPS,
    AUGMENTED_SUBSETS,
    DISTAL_ARM,
    augmented_manifest,
    compute_closed_form_cohen_d_specialist_detectability,
    compute_closed_form_cohen_d_win_table,
    plot_augmented_distal_trajectories,
    plot_augmented_specialist_closed_form_win_percentage,
    plot_augmented_specialist_trajectories,
)
from marin_dna_evals.soft_vep_metrics import AUPRC, GROUP_SMD, VARIANT_POOLED_SMD


def test_augmented_manifest_is_six_by_eight():
    manifest = augmented_manifest()

    assert len(manifest) == 48
    assert AUGMENTED_STEPS == (500, 1500, 2000, 3000, 3500, 4000, 4500, 4999)
    assert set(manifest["arm"]) == set(AUGMENTED_ARMS)
    assert set(manifest["step"]) == set(AUGMENTED_STEPS)
    assert manifest["uri"].is_unique
    distal = manifest[manifest["arm"] == DISTAL_ARM]
    assert distal["role"].eq("replacement_home").all()
    assert distal["model"].str.startswith("exp351-centered-step-").all()
    assert not distal["model"].str.endswith("-1000").any()


def test_augmented_specialist_detectability_uses_all_six_arms():
    point_values = {
        arm: 0.50 + index / 100 for index, arm in enumerate(AUGMENTED_ARMS)
    }
    point_values[DISTAL_ARM] = 0.80
    point = pd.DataFrame(
        [
            {"score_type": arm, "metric": AUPRC, "value": value}
            for arm, value in point_values.items()
        ]
    )
    samples = pd.DataFrame(
        [
            {
                "draw": draw,
                "score_type": arm,
                "metric": AUPRC,
                "value": value + draw / 1000,
            }
            for draw in range(10)
            for arm, value in point_values.items()
        ]
    )

    result = specialist_detectability_summary(
        point,
        samples,
        subset="distal",
        metrics=(AUPRC,),
        arms=AUGMENTED_ARMS,
        specialist_by_subset=AUGMENTED_SPECIALIST_ARM,
    ).iloc[0]

    assert result["home_arm"] == DISTAL_ARM
    assert result["home_rank"] == 1
    assert result["strongest_competitor"] == "tss_region_and_utr5"
    assert result["home_minus_best_oriented"] == pytest.approx(0.26)
    assert result["bootstrap_home_rank1_frequency"] == pytest.approx(1.0)


def test_augmented_timing_and_distal_plot(tmp_path):
    detectability = pd.DataFrame(
        [
            {
                "subset": subset,
                "home_arm": AUGMENTED_SPECIALIST_ARM[subset],
                "metric": metric,
                "step": step,
                "confidence_supported": step >= threshold,
                "bootstrap_home_rank1_frequency": (
                    0.99 if step >= threshold else 0.5
                ),
                "n_bootstrap": 100,
            }
            for subset in AUGMENTED_SUBSETS
            for metric, threshold in ((AUPRC, 2000), (GROUP_SMD, 1500))
            for step in AUGMENTED_STEPS
        ]
    )
    timing = persistent_specialist_detectability(
        detectability,
        synchronized_steps=AUGMENTED_STEPS,
    )
    comparison = compare_metric_detection_timing(
        timing,
        metrics=(AUPRC, GROUP_SMD),
        subsets=AUGMENTED_SUBSETS,
    ).set_index("metric")
    assert comparison.loc[GROUP_SMD, "candidate_earlier"] == 8

    point_rows = []
    cohen_d_rows = []
    for step in AUGMENTED_STEPS:
        for arm_index, arm in enumerate(AUGMENTED_ARMS):
            value = 0.2 + arm_index / 100 + step / 100_000
            point_rows.append(
                {
                    "arm": arm,
                    "step": step,
                    "subset": "distal",
                    "metric": AUPRC,
                    "value": value,
                    "ci_low": value - 0.01,
                    "ci_high": value + 0.01,
                }
            )
            value = 0.7 + arm_index / 100 + step / 100_000
            cohen_d_rows.append(
                {
                    "arm": arm,
                    "step": step,
                    "subset": "distal",
                    "metric": VARIANT_POOLED_SMD,
                    "value": value,
                    "ci_low": value - 0.01,
                    "ci_high": value + 0.01,
                }
            )

    outputs = plot_augmented_distal_trajectories(
        pd.DataFrame(point_rows),
        pd.DataFrame(cohen_d_rows),
        tmp_path,
    )

    assert set(outputs) == {
        "plot_augmented_distal_metric_trajectories_svg",
        "plot_augmented_distal_metric_trajectories_png",
    }
    assert all(path.stat().st_size > 0 for path in outputs.values())


def test_full_specialist_and_closed_form_win_plots(tmp_path):
    point_rows = []
    cohen_d_rows = []
    for subset in AUGMENTED_SUBSETS:
        home_arm = AUGMENTED_SPECIALIST_ARM[subset]
        for step in AUGMENTED_STEPS:
            for arm_index, arm in enumerate(AUGMENTED_ARMS):
                point_rows.append(
                    {
                        "arm": arm,
                        "step": step,
                        "subset": subset,
                        "metric": AUPRC,
                        "value": 0.2 + arm_index / 100 + step / 100_000,
                        "ci_low": 0.18 + arm_index / 100 + step / 100_000,
                        "ci_high": 0.22 + arm_index / 100 + step / 100_000,
                    }
                )
                value = 1.0 if arm == home_arm else 0.0
                cohen_d_rows.append(
                    {
                        "arm": arm,
                        "step": step,
                        "subset": subset,
                        "metric": VARIANT_POOLED_SMD,
                        "value": value,
                        "se": 2**-0.5,
                        "ci_low": value - 0.1,
                        "ci_high": value + 0.1,
                    }
                )

    point_metrics = pd.DataFrame(point_rows)
    cohen_d_metrics = pd.DataFrame(cohen_d_rows)
    win_probabilities = compute_closed_form_cohen_d_win_table(cohen_d_metrics)

    assert len(win_probabilities) == (
        len(AUGMENTED_SUBSETS)
        * len(AUGMENTED_STEPS)
        * (len(AUGMENTED_ARMS) - 1)
    )
    assert win_probabilities["win_probability"].to_numpy() == pytest.approx(
        0.8413447460685429
    )
    assert win_probabilities["uncertainty_method"].eq(
        "independent_normal_closed_form_cohen_d"
    ).all()
    detectability = compute_closed_form_cohen_d_specialist_detectability(
        cohen_d_metrics
    )
    assert len(detectability) == len(AUGMENTED_SUBSETS) * len(AUGMENTED_STEPS)
    assert detectability["home_minus_best_oriented"].to_numpy() == pytest.approx(1)
    assert detectability["margin_se"].to_numpy() == pytest.approx(1)
    assert detectability["margin_ci_low"].to_numpy() == pytest.approx(
        1 - 1.959963984540054
    )
    assert not detectability["confidence_supported"].any()
    assert detectability["uncertainty_method"].eq(
        "independent_normal_closed_form_cohen_d"
    ).all()

    trajectory_outputs = plot_augmented_specialist_trajectories(
        point_metrics,
        cohen_d_metrics,
        tmp_path,
    )
    win_outputs = plot_augmented_specialist_closed_form_win_percentage(
        win_probabilities,
        tmp_path,
    )

    assert set(trajectory_outputs) == {
        "plot_augmented_specialist_auprc_vs_cohen_d_svg",
        "plot_augmented_specialist_auprc_vs_cohen_d_png",
    }
    assert set(win_outputs) == {
        "plot_augmented_specialist_closed_form_win_percentage_svg",
        "plot_augmented_specialist_closed_form_win_percentage_png",
    }
    assert all(
        path.stat().st_size > 0
        for path in (*trajectory_outputs.values(), *win_outputs.values())
    )
