"""Contracts for the issue #459 Mendelian soft-metric panel."""

import numpy as np
import pandas as pd
import pytest
from marin_dna_evals.soft_vep_metrics import (
    AUPRC,
    CALIBRATED_BRIER,
    CALIBRATED_LOG_LOSS,
    GROUP_MEDIAN_MAD,
    GROUP_SMD,
    MEAN_GAP_GLOBAL,
    MEAN_GAP_GROUP,
    SOFT_WIN,
    STUDENT_T,
    VARIANT_POOLED_SMD,
    VARIANT_TOTAL_SD_GAP,
    WELCH_T,
    cohen_d_closed_form_se,
    cohen_d_closed_form_table,
    compute_mendelian_soft_metric_table,
    compute_mendelian_soft_metrics,
    compute_ungrouped_metric_table,
    group_differences,
    grouped_calibration_scores,
    joint_cluster_bootstrap_soft_metrics,
    joint_stratified_row_bootstrap_ungrouped_metrics,
    reference_soft_win_temperature,
    summarize_joint_bootstrap,
)
from scipy.stats import ttest_ind
from sklearn.metrics import average_precision_score


def _matched_data(
    k: int = 3, n_groups: int = 12
) -> tuple[pd.Series, pd.Series, pd.Series]:
    labels: list[int] = []
    scores: list[float] = []
    groups: list[int] = []
    for group in range(n_groups):
        positive = 0.3 + group * 0.17 + (group % 3) * 0.11
        negatives = [group * 0.05 - j * 0.13 + (group % 2) * 0.04 for j in range(k)]
        labels.extend([1, *([0] * k)])
        scores.extend([positive, *negatives])
        groups.extend([group] * (k + 1))
    return pd.Series(labels), pd.Series(scores), pd.Series(groups)


def test_complete_fixed_ratio_global_and_group_gaps_are_equal():
    label, score, match_group = _matched_data(k=9)
    metrics = compute_mendelian_soft_metrics(
        label,
        score,
        match_group,
        tau=1.0,
        metrics=[MEAN_GAP_GLOBAL, MEAN_GAP_GROUP],
    )
    assert metrics[MEAN_GAP_GLOBAL] == pytest.approx(metrics[MEAN_GAP_GROUP])


def test_variable_control_ratio_can_separate_global_and_group_gaps():
    label = pd.Series([1, 0, 1, 0, 0, 0])
    score = pd.Series([2.0, 0.0, 3.0, 2.0, 2.0, 2.0])
    match_group = pd.Series([0, 0, 1, 1, 1, 1])
    metrics = compute_mendelian_soft_metrics(
        label,
        score,
        match_group,
        tau=1.0,
        metrics=[MEAN_GAP_GLOBAL, MEAN_GAP_GROUP],
    )
    assert metrics[MEAN_GAP_GLOBAL] == pytest.approx(1.0)
    assert metrics[MEAN_GAP_GROUP] == pytest.approx(1.5)


def test_metric_panel_matches_direct_definitions():
    label, score, match_group = _matched_data()
    tau = reference_soft_win_temperature(label, score, match_group)
    metrics = compute_mendelian_soft_metrics(label, score, match_group, tau=tau)
    differences = group_differences(label, score, match_group)

    assert metrics[AUPRC] == pytest.approx(average_precision_score(label, score))
    assert metrics[MEAN_GAP_GLOBAL] == pytest.approx(
        score[label == 1].mean() - score[label == 0].mean()
    )
    assert metrics[MEAN_GAP_GROUP] == pytest.approx(differences.mean())
    assert metrics[GROUP_SMD] == pytest.approx(
        differences.mean() / differences.std(ddof=1)
    )
    median = differences.median()
    mad = np.median(np.abs(differences - median))
    assert metrics[GROUP_MEDIAN_MAD] == pytest.approx(median / (1.4826 * mad))
    assert 0.5 < metrics[SOFT_WIN] < 1.0
    assert metrics[CALIBRATED_LOG_LOSS] > 0
    assert 0 < metrics[CALIBRATED_BRIER] < 1


def test_positive_rescaling_contracts():
    label, score, match_group = _matched_data()
    base = compute_mendelian_soft_metrics(label, score, match_group, tau=0.5)
    scaled = compute_mendelian_soft_metrics(label, 7.0 * score, match_group, tau=0.5)

    assert scaled[AUPRC] == pytest.approx(base[AUPRC])
    assert scaled[MEAN_GAP_GLOBAL] == pytest.approx(7.0 * base[MEAN_GAP_GLOBAL])
    assert scaled[MEAN_GAP_GROUP] == pytest.approx(7.0 * base[MEAN_GAP_GROUP])
    assert scaled[GROUP_SMD] == pytest.approx(base[GROUP_SMD])
    assert scaled[GROUP_MEDIAN_MAD] == pytest.approx(base[GROUP_MEDIAN_MAD])
    assert scaled[SOFT_WIN] != pytest.approx(base[SOFT_WIN])
    assert scaled[CALIBRATED_LOG_LOSS] == pytest.approx(
        base[CALIBRATED_LOG_LOSS], abs=1e-8
    )
    assert scaled[CALIBRATED_BRIER] == pytest.approx(base[CALIBRATED_BRIER], abs=1e-8)


def test_sign_reversal_reverses_directional_soft_metrics():
    label, score, match_group = _matched_data()
    forward = compute_mendelian_soft_metrics(
        label,
        score,
        match_group,
        tau=0.5,
        metrics=[
            MEAN_GAP_GLOBAL,
            MEAN_GAP_GROUP,
            GROUP_SMD,
            GROUP_MEDIAN_MAD,
            SOFT_WIN,
        ],
    )
    reverse = compute_mendelian_soft_metrics(
        label,
        -score,
        match_group,
        tau=0.5,
        metrics=[
            MEAN_GAP_GLOBAL,
            MEAN_GAP_GROUP,
            GROUP_SMD,
            GROUP_MEDIAN_MAD,
            SOFT_WIN,
        ],
    )
    for metric in [MEAN_GAP_GLOBAL, MEAN_GAP_GROUP, GROUP_SMD, GROUP_MEDIAN_MAD]:
        assert reverse[metric] == pytest.approx(-forward[metric])
    assert reverse[SOFT_WIN] == pytest.approx(1.0 - forward[SOFT_WIN])


def test_grouped_calibration_is_positive_affine_invariant():
    label, score, match_group = _matched_data()
    base = grouped_calibration_scores(label, score, match_group)
    transformed = grouped_calibration_scores(label, 12.0 * score + 4.5, match_group)
    assert transformed == pytest.approx(base, abs=1e-8)


def test_invalid_match_group_fails_loud():
    label = pd.Series([1, 1, 0, 0])
    score = pd.Series([1.0, 0.8, 0.2, 0.1])
    match_group = pd.Series([0, 0, 0, 1])
    with pytest.raises(AssertionError, match="exactly one positive"):
        compute_mendelian_soft_metrics(label, score, match_group, tau=1.0)


def test_joint_bootstrap_uses_identical_draws_for_identical_scores():
    label, score, match_group = _matched_data()
    scores = pd.DataFrame({"a": score, "b": score.copy()})
    point, samples = joint_cluster_bootstrap_soft_metrics(
        label,
        scores,
        match_group,
        tau=0.5,
        metrics=[AUPRC, MEAN_GAP_GLOBAL, MEAN_GAP_GROUP, SOFT_WIN],
        n_bootstrap=30,
        rng=17,
    )
    a = samples[samples["score_type"] == "a"].sort_values(["draw", "metric"])
    b = samples[samples["score_type"] == "b"].sort_values(["draw", "metric"])
    assert (
        a[["draw", "metric"]]
        .reset_index(drop=True)
        .equals(b[["draw", "metric"]].reset_index(drop=True))
    )
    assert a["value"].to_numpy() == pytest.approx(b["value"].to_numpy())

    summary = summarize_joint_bootstrap(point, samples)
    assert set(["se", "ci_low", "ci_high"]).issubset(summary.columns)
    assert summary["se"].notna().all()


def test_joint_bootstrap_includes_fixed_oof_calibration_uncertainty():
    label, score, match_group = _matched_data()
    point, samples = joint_cluster_bootstrap_soft_metrics(
        label,
        pd.DataFrame({"score": score}),
        match_group,
        tau=0.5,
        metrics=[CALIBRATED_LOG_LOSS, CALIBRATED_BRIER],
        n_bootstrap=12,
        rng=3,
    )
    assert set(point["metric"]) == {CALIBRATED_LOG_LOSS, CALIBRATED_BRIER}
    assert set(samples["metric"]) == {CALIBRATED_LOG_LOSS, CALIBRATED_BRIER}
    assert samples["draw"].nunique() == 12
    assert samples["value"].notna().all()


def test_weighted_joint_bootstrap_auprc_matches_explicit_duplication():
    label, score, match_group = _matched_data(k=3, n_groups=9)
    n_bootstrap = 20
    seed = 23
    _, samples = joint_cluster_bootstrap_soft_metrics(
        label,
        pd.DataFrame({"score": score}),
        match_group,
        tau=0.5,
        metrics=[AUPRC],
        n_bootstrap=n_bootstrap,
        rng=seed,
    )

    group_rows = list(match_group.groupby(match_group, sort=False).indices.values())
    generator = np.random.default_rng(seed)
    expected = []
    for _ in range(n_bootstrap):
        sampled = generator.integers(0, len(group_rows), size=len(group_rows))
        idx = np.concatenate([group_rows[group] for group in sampled])
        expected.append(average_precision_score(label.iloc[idx], score.iloc[idx]))
    assert samples.sort_values("draw")["value"].to_numpy() == pytest.approx(expected)


def test_ungrouped_metrics_match_standardized_gap_and_t_definitions():
    label = pd.Series([1, 1, 1, 0, 0, 0, 0])
    score = pd.Series([4.0, 5.0, 8.0, 0.0, 1.0, 2.0, 4.0])
    table = compute_ungrouped_metric_table(
        label,
        pd.DataFrame({"score": score}),
    ).set_index("metric")
    positive = score[label == 1].to_numpy()
    negative = score[label == 0].to_numpy()
    gap = positive.mean() - negative.mean()
    pooled_variance = (
        (len(positive) - 1) * positive.var(ddof=1)
        + (len(negative) - 1) * negative.var(ddof=1)
    ) / (len(score) - 2)

    assert table.loc[VARIANT_POOLED_SMD, "value"] == pytest.approx(
        gap / np.sqrt(pooled_variance)
    )
    assert table.loc[VARIANT_TOTAL_SD_GAP, "value"] == pytest.approx(
        gap / score.std(ddof=1)
    )
    assert table.loc[STUDENT_T, "value"] == pytest.approx(
        ttest_ind(positive, negative, equal_var=True).statistic
    )
    assert table.loc[WELCH_T, "value"] == pytest.approx(
        ttest_ind(positive, negative, equal_var=False).statistic
    )


def test_cohen_d_closed_form_uncertainty_uses_class_counts():
    point = pd.DataFrame(
        {
            "score_type": ["model", "model"],
            "metric": [AUPRC, VARIANT_POOLED_SMD],
            "value": [0.8, 0.6],
            "n_rows": [50, 50],
            "n_pos": [20, 20],
        }
    )

    result = cohen_d_closed_form_table(point).iloc[0]
    expected_se = np.sqrt(50 / (20 * 30) + 0.6**2 / (2 * 48))

    assert result["metric"] == VARIANT_POOLED_SMD
    assert result["n_neg"] == 30
    assert result["se"] == pytest.approx(expected_se)
    assert result["ci_low"] == pytest.approx(0.6 - 1.959963984540054 * expected_se)
    assert result["ci_high"] == pytest.approx(0.6 + 1.959963984540054 * expected_se)
    assert result["uncertainty_method"] == "conventional_iid_closed_form"
    assert cohen_d_closed_form_se(0.6, 20, 30) == pytest.approx(expected_se)


def test_ungrouped_metrics_are_positive_affine_invariant_and_directional():
    label, score, _ = _matched_data(k=3, n_groups=8)
    base = compute_ungrouped_metric_table(
        label,
        pd.DataFrame({"score": score}),
    ).set_index("metric")["value"]
    transformed = compute_ungrouped_metric_table(
        label,
        pd.DataFrame({"score": 7.0 * score + 11.0}),
    ).set_index("metric")["value"]
    reversed_values = compute_ungrouped_metric_table(
        label,
        pd.DataFrame({"score": -score}),
    ).set_index("metric")["value"]

    assert transformed.to_dict() == pytest.approx(base.to_dict())
    assert reversed_values[AUPRC] != pytest.approx(base[AUPRC])
    for metric in [VARIANT_POOLED_SMD, VARIANT_TOTAL_SD_GAP, STUDENT_T, WELCH_T]:
        assert reversed_values[metric] == pytest.approx(-base[metric])


def test_stratified_row_bootstrap_is_joint_and_matches_explicit_auprc():
    label, score, _ = _matched_data(k=2, n_groups=5)
    scores = pd.DataFrame({"a": score, "b": score.copy()})
    n_bootstrap = 12
    seed = 31
    point, samples = joint_stratified_row_bootstrap_ungrouped_metrics(
        label,
        scores,
        n_bootstrap=n_bootstrap,
        rng=seed,
    )
    assert set(point["metric"]) == {
        AUPRC,
        VARIANT_POOLED_SMD,
        VARIANT_TOTAL_SD_GAP,
        STUDENT_T,
        WELCH_T,
    }
    a = samples[samples["score_type"] == "a"].sort_values(["draw", "metric"])
    b = samples[samples["score_type"] == "b"].sort_values(["draw", "metric"])
    assert a["value"].to_numpy() == pytest.approx(b["value"].to_numpy())

    positive_rows = np.flatnonzero(label.to_numpy() == 1)
    negative_rows = np.flatnonzero(label.to_numpy() == 0)
    generator = np.random.default_rng(seed)
    sampled_positive = positive_rows[
        generator.integers(0, len(positive_rows), size=(n_bootstrap, len(positive_rows)))
    ]
    sampled_negative = negative_rows[
        generator.integers(0, len(negative_rows), size=(n_bootstrap, len(negative_rows)))
    ]
    expected = [
        average_precision_score(
            label.iloc[np.r_[sampled_positive[draw], sampled_negative[draw]]],
            score.iloc[np.r_[sampled_positive[draw], sampled_negative[draw]]],
        )
        for draw in range(n_bootstrap)
    ]
    observed = samples[
        (samples["score_type"] == "a") & (samples["metric"] == AUPRC)
    ].sort_values("draw")["value"]
    assert observed.to_numpy() == pytest.approx(expected)


def test_metric_table_records_direction_and_counts():
    label, score, match_group = _matched_data(k=2, n_groups=8)
    table = compute_mendelian_soft_metric_table(
        label,
        pd.DataFrame({"score": score}),
        match_group,
        tau=1.0,
    )
    assert set(table["metric"]) == {
        AUPRC,
        MEAN_GAP_GLOBAL,
        MEAN_GAP_GROUP,
        GROUP_SMD,
        GROUP_MEDIAN_MAD,
        SOFT_WIN,
        CALIBRATED_LOG_LOSS,
        CALIBRATED_BRIER,
    }
    assert table["n_groups"].eq(8).all()
    assert table["n_rows"].eq(24).all()
    assert table["n_pos"].eq(8).all()
    assert (
        table.loc[table["metric"] == CALIBRATED_LOG_LOSS, "higher_is_better"]
        .eq(False)
        .all()
    )
