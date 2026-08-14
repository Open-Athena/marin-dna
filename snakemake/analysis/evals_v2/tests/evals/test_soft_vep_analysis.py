"""Pure analysis-harness tests; no S3 access."""

import numpy as np
import pandas as pd
import pytest
from marin_dna_evals.soft_vep_analysis import (
    ARMS,
    DETECTABILITY_METRICS,
    NON_DISTAL_SUBSETS,
    SPECIALIST_ARM,
    SYNCHRONIZED_STEPS,
    UNGROUPED_DETECTABILITY_METRICS,
    add_llr_scores,
    compare_metric_detection_timing,
    compute_rank_agreement,
    confidence_filtered_rank_reversals,
    earliest_persistent_specialist_wins,
    exp232_manifest,
    persistent_specialist_detectability,
    permute_labels_within_groups,
    plot_exp232_specialist_auprc_vs_brier,
    plot_metric_detectability_summary,
    plot_specialist_detectability,
    specialist_detectability_summary,
    validate_aligned_bundles,
)
from marin_dna_evals.soft_vep_metrics import (
    AUPRC,
    CALIBRATED_BRIER,
    MEAN_GAP_GLOBAL,
    STUDENT_T,
    VARIANT_POOLED_SMD,
    VARIANT_TOTAL_SD_GAP,
    WELCH_T,
)


def test_exp232_manifest_is_exact_inventory():
    manifest = exp232_manifest()
    assert len(manifest) == 48
    assert SYNCHRONIZED_STEPS == (500, 1000, 1500, 2000, 3000, 3500, 4000, 4500, 4999)
    assert manifest[manifest["step"] == 2500]["arm"].tolist() == ["bg", "cds", "utr3"]
    assert manifest["uri"].is_unique
    assert manifest["split"].eq("train").all()


def test_add_llr_scores_uses_signed_fwd_rc_average():
    frame = pd.DataFrame(
        {
            "label": [True, False],
            "subset": ["x", "x"],
            "match_group": [0, 0],
            "llr_fwd": [-2.0, 1.0],
            "llr_rc": [-4.0, 3.0],
        }
    )
    result = add_llr_scores(frame)
    assert result["minus_llr_avg"].tolist() == pytest.approx([3.0, -2.0])
    assert result["minus_llr_fwd"].tolist() == pytest.approx([2.0, -1.0])


def test_validate_aligned_bundles_fails_on_label_drift():
    base = pd.DataFrame(
        {
            "label": [True, False],
            "subset": ["x", "x"],
            "match_group": [0, 0],
        }
    )
    changed = base.copy()
    changed.loc[0, "label"] = False
    with pytest.raises(AssertionError, match="joint model bootstrap"):
        validate_aligned_bundles({"a": base, "b": changed})


def test_label_permutation_preserves_one_positive_per_group():
    groups = pd.Series([0, 0, 0, 1, 1, 1, 1])
    labels = permute_labels_within_groups(groups, rng=459)
    assert labels.groupby(groups).agg(["sum", "size"]).to_dict("index") == {
        0: {"sum": 1, "size": 3},
        1: {"sum": 1, "size": 4},
    }


def _synthetic_point_metrics() -> pd.DataFrame:
    rows = []
    for subset in NON_DISTAL_SUBSETS:
        specialist = SPECIALIST_ARM[subset]
        for step in SYNCHRONIZED_STEPS:
            for arm_index, arm in enumerate(ARMS):
                specialist_bonus = 0.0
                if arm == specialist:
                    specialist_bonus = 10.0 if step >= 1000 else -10.0
                auprc = arm_index + specialist_bonus
                rows.extend(
                    [
                        {
                            "arm": arm,
                            "step": step,
                            "subset": subset,
                            "metric": AUPRC,
                            "value": auprc,
                        },
                        {
                            "arm": arm,
                            "step": step,
                            "subset": subset,
                            "metric": MEAN_GAP_GLOBAL,
                            "value": auprc,
                        },
                    ]
                )
    return pd.DataFrame(rows)


def test_rank_agreement_exact_match_has_no_reversals():
    summary, pairs = compute_rank_agreement(_synthetic_point_metrics())
    matched = summary[
        (summary["metric"] == MEAN_GAP_GLOBAL)
        & (summary["reference"] == "same_step_auprc")
    ]
    assert np.allclose(matched["spearman"], 1.0)
    assert np.allclose(matched["kendall"], 1.0)
    assert matched["pairwise_reversals"].eq(0).all()
    matched_pairs = pairs[
        (pairs["metric"] == MEAN_GAP_GLOBAL) & (pairs["reference"] == "same_step_auprc")
    ]
    assert not matched_pairs["reversal"].any()


def test_earliest_persistent_specialist_win_uses_two_stored_steps():
    wins = earliest_persistent_specialist_wins(_synthetic_point_metrics())
    assert wins["earliest_persistent_step"].eq(1000).all()
    assert wins["final_step_win"].all()


def test_confidence_filter_drops_unresolved_pairs():
    pairwise = pd.DataFrame(
        [
            {
                "step": 500,
                "subset": "s",
                "metric": AUPRC,
                "arm_a": "a",
                "arm_b": "b",
                "ci_low": 0.1,
                "ci_high": 0.3,
            },
            {
                "step": 500,
                "subset": "s",
                "metric": MEAN_GAP_GLOBAL,
                "arm_a": "a",
                "arm_b": "b",
                "ci_low": -0.2,
                "ci_high": 0.4,
            },
        ]
    )
    result = confidence_filtered_rank_reversals(
        pairwise,
        group_columns=["step", "subset"],
        entity_columns=("arm_a", "arm_b"),
    )
    assert result.loc[0, "n_informative_pairs"] == 0
    assert pd.isna(result.loc[0, "agreement_fraction"])


def test_specialist_comparison_plot_writes_svg_and_png(tmp_path):
    rows = []
    for subset in NON_DISTAL_SUBSETS:
        for arm_index, arm in enumerate(ARMS):
            for step in (500, 1000):
                for metric, value in (
                    (AUPRC, 0.2 + arm_index / 100 + step / 10_000),
                    (CALIBRATED_BRIER, 0.1 - arm_index / 1000 - step / 100_000),
                ):
                    rows.append(
                        {
                            "arm": arm,
                            "step": step,
                            "subset": subset,
                            "metric": metric,
                            "value": value,
                            "ci_low": value - 0.01,
                            "ci_high": value + 0.01,
                        }
                    )

    outputs = plot_exp232_specialist_auprc_vs_brier(
        pd.DataFrame(rows),
        tmp_path,
    )

    assert set(outputs) == {
        "plot_specialist_auprc_vs_brier_svg",
        "plot_specialist_auprc_vs_brier_png",
    }
    assert outputs["plot_specialist_auprc_vs_brier_svg"].stat().st_size > 0
    assert outputs["plot_specialist_auprc_vs_brier_png"].stat().st_size > 0


def test_specialist_detectability_uses_joint_rank_and_brier_orientation():
    point_values = {
        AUPRC: {
            "bg": 0.70,
            "cds": 0.80,
            "utr3": 0.60,
            "ncrna_exon": 0.50,
            "tss_region_and_utr5": 0.40,
        },
        CALIBRATED_BRIER: {
            "bg": 0.20,
            "cds": 0.10,
            "utr3": 0.30,
            "ncrna_exon": 0.40,
            "tss_region_and_utr5": 0.50,
        },
    }
    point = pd.DataFrame(
        [
            {"score_type": arm, "metric": metric, "value": value}
            for metric, arm_values in point_values.items()
            for arm, value in arm_values.items()
        ]
    )
    samples = []
    for draw in range(3):
        for metric, arm_values in point_values.items():
            for arm, value in arm_values.items():
                if arm == "cds" and metric == AUPRC:
                    value = (0.80, 0.65, 0.75)[draw]
                if arm == "cds" and metric == CALIBRATED_BRIER:
                    value = (0.10, 0.12, 0.15)[draw]
                samples.append(
                    {
                        "draw": draw,
                        "score_type": arm,
                        "metric": metric,
                        "value": value,
                    }
                )

    result = specialist_detectability_summary(
        point,
        pd.DataFrame(samples),
        subset="missense_variant",
        metrics=(AUPRC, CALIBRATED_BRIER),
    ).set_index("metric")

    assert result.loc[AUPRC, "home_arm"] == "cds"
    assert result.loc[AUPRC, "strongest_competitor"] == "bg"
    assert result.loc[AUPRC, "home_rank"] == 1
    assert result.loc[AUPRC, "home_minus_best_oriented"] == pytest.approx(0.1)
    assert result.loc[AUPRC, "bootstrap_home_rank1_frequency"] == pytest.approx(2 / 3)
    assert not result.loc[AUPRC, "confidence_supported"]
    assert result.loc[CALIBRATED_BRIER, "home_minus_best_oriented"] == pytest.approx(
        0.1
    )
    assert result.loc[
        CALIBRATED_BRIER, "bootstrap_home_rank1_frequency"
    ] == pytest.approx(1.0)
    assert result.loc[CALIBRATED_BRIER, "confidence_supported"]


def _synthetic_detectability() -> pd.DataFrame:
    rows = []
    for subset in NON_DISTAL_SUBSETS:
        for metric in (AUPRC, CALIBRATED_BRIER):
            threshold = 1000 if metric == AUPRC else 2000
            for step in SYNCHRONIZED_STEPS:
                supported = step >= threshold
                rows.append(
                    {
                        "subset": subset,
                        "home_arm": SPECIALIST_ARM[subset],
                        "metric": metric,
                        "step": step,
                        "confidence_supported": supported,
                        "bootstrap_home_rank1_frequency": (0.99 if supported else 0.55),
                        "n_bootstrap": 1000,
                    }
                )
    return pd.DataFrame(rows)


def test_persistent_detectability_and_plots(tmp_path):
    detectability = _synthetic_detectability()
    timing = persistent_specialist_detectability(detectability)
    timing_wide = timing.pivot(
        index="subset",
        columns="metric",
        values="earliest_persistent_detected_step",
    )
    assert timing_wide[AUPRC].eq(1000).all()
    assert timing_wide[CALIBRATED_BRIER].eq(2000).all()

    outputs = plot_specialist_detectability(detectability, timing, tmp_path)

    assert set(outputs) == {
        "plot_specialist_detectability_svg",
        "plot_specialist_detectability_png",
        "plot_specialist_detection_timing_svg",
        "plot_specialist_detection_timing_png",
    }
    assert all(path.stat().st_size > 0 for path in outputs.values())


def test_all_metric_detection_comparison_and_plot(tmp_path):
    rows = []
    for subset in NON_DISTAL_SUBSETS:
        for metric in DETECTABILITY_METRICS:
            if metric == AUPRC:
                step = 2000
            elif metric == MEAN_GAP_GLOBAL:
                step = 1000
            elif metric == "calibrated_log_loss":
                step = 2000
            elif metric == CALIBRATED_BRIER:
                step = 3000
            else:
                step = np.nan
            rows.append(
                {
                    "subset": subset,
                    "home_arm": SPECIALIST_ARM[subset],
                    "metric": metric,
                    "earliest_persistent_detected_step": step,
                }
            )
    timing = pd.DataFrame(rows)
    comparison = compare_metric_detection_timing(timing).set_index("metric")

    assert comparison.loc[MEAN_GAP_GLOBAL, "candidate_earlier"] == 7
    assert comparison.loc["calibrated_log_loss", "same_step"] == 7
    assert comparison.loc[CALIBRATED_BRIER, "auprc_earlier"] == 7

    outputs = plot_metric_detectability_summary(
        timing,
        comparison.reset_index(),
        tmp_path,
    )
    assert set(outputs) == {
        "plot_specialist_metric_detectability_summary_svg",
        "plot_specialist_metric_detectability_summary_png",
    }
    assert all(path.stat().st_size > 0 for path in outputs.values())


def test_ungrouped_metric_detection_comparison_and_plot(tmp_path):
    detection_steps = {
        AUPRC: 2000,
        VARIANT_POOLED_SMD: 1000,
        VARIANT_TOTAL_SD_GAP: 1500,
        STUDENT_T: 1000,
        WELCH_T: 3000,
    }
    timing = pd.DataFrame(
        [
            {
                "subset": subset,
                "home_arm": SPECIALIST_ARM[subset],
                "metric": metric,
                "earliest_persistent_detected_step": step,
            }
            for subset in NON_DISTAL_SUBSETS
            for metric, step in detection_steps.items()
        ]
    )
    comparison = compare_metric_detection_timing(
        timing,
        metrics=UNGROUPED_DETECTABILITY_METRICS,
    ).set_index("metric")
    assert comparison.loc[VARIANT_POOLED_SMD, "candidate_earlier"] == 7
    assert comparison.loc[VARIANT_TOTAL_SD_GAP, "candidate_earlier"] == 7
    assert comparison.loc[STUDENT_T, "candidate_earlier"] == 7
    assert comparison.loc[WELCH_T, "auprc_earlier"] == 7

    outputs = plot_metric_detectability_summary(
        timing,
        comparison.reset_index(),
        tmp_path,
        metrics=UNGROUPED_DETECTABILITY_METRICS,
        stem="ungrouped",
        bootstrap_unit="class-stratified variant",
        metric_note="No match groups used.",
    )
    assert set(outputs) == {
        "plot_ungrouped_svg",
        "plot_ungrouped_png",
    }
    assert all(path.stat().st_size > 0 for path in outputs.values())
