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
    plot_augmented_distal_trajectories,
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
