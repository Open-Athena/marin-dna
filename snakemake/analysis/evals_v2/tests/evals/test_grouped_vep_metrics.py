"""Contracts for maintained grouped VEP AUPRC and Group SMD reporting."""

import numpy as np
import pandas as pd
import pytest
from marin_dna_evals.grouped_vep_metrics import (
    AUPRC,
    GROUP_SMD,
    compute_grouped_vep_metrics,
    group_smd,
    matched_group_gaps,
)
from marin_dna_evals.metrics import MACRO_AVG_SUBSET, compute_auprc_metrics
from sklearn.metrics import average_precision_score


def _grouped_data(
    *,
    subsets: tuple[str, ...] = ("coding",),
    n_groups_per_subset: int = 6,
    n_negatives: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_rows: list[dict[str, int | str]] = []
    scores: list[float] = []
    group = 0
    for subset_index, subset in enumerate(subsets):
        for within_subset in range(n_groups_per_subset):
            positive = 1.0 + 0.31 * within_subset + 0.17 * (within_subset % 2)
            negatives = [
                0.13 * within_subset - 0.19 * negative
                for negative in range(n_negatives)
            ]
            dataset_rows.append({"label": 1, "subset": subset, "match_group": group})
            scores.append(positive + 0.05 * subset_index)
            for negative in negatives:
                dataset_rows.append(
                    {"label": 0, "subset": subset, "match_group": group}
                )
                scores.append(negative)
            group += 1
    score = np.asarray(scores)
    return pd.DataFrame(dataset_rows), pd.DataFrame(
        {"score": score, "identical_score": score.copy()}
    )


def test_group_smd_matches_positive_minus_mean_negative_formula():
    label = pd.Series([1, 0, 0, 1, 0, 0, 1, 0, 0])
    score = pd.Series([2.0, 1.0, 1.0, 5.0, 3.0, 3.0, 9.0, 5.0, 5.0])
    match_group = pd.Series([0, 0, 0, 1, 1, 1, 2, 2, 2])
    expected_gaps = pd.Series([1.0, 2.0, 4.0], name="group_gap")

    pd.testing.assert_series_equal(
        matched_group_gaps(label, score, match_group).reset_index(drop=True),
        expected_gaps,
    )
    assert group_smd(label, score, match_group) == pytest.approx(
        expected_gaps.mean() / expected_gaps.std(ddof=1)
    )


def test_group_smd_is_positive_affine_invariant():
    dataset, scores = _grouped_data()
    base = group_smd(dataset["label"], scores["score"], dataset["match_group"])
    transformed = group_smd(
        dataset["label"],
        7.5 * scores["score"] + 13.0,
        dataset["match_group"],
    )
    assert transformed == pytest.approx(base)


@pytest.mark.parametrize(
    ("dataset_mutation", "message"),
    [
        (lambda dataset: dataset.drop(columns="match_group"), "missing required"),
        (lambda dataset: dataset.drop(columns="subset"), "missing required"),
        (
            lambda dataset: dataset.assign(match_group=0),
            "exactly one positive",
        ),
        (
            lambda dataset: dataset.assign(
                match_group=dataset["match_group"].mask(dataset.index == 0, 1)
            ),
            "exactly one positive",
        ),
    ],
)
def test_group_contract_validation_fails_loud(dataset_mutation, message: str):
    dataset, scores = _grouped_data(n_groups_per_subset=3)
    with pytest.raises(ValueError, match=message):
        compute_grouped_vep_metrics(
            dataset_mutation(dataset),
            scores,
            n_bootstrap=5,
            n_min=1,
        )


def test_match_group_may_not_span_subsets():
    dataset, scores = _grouped_data(subsets=("coding", "splicing"))
    dataset.loc[dataset.index[0], "subset"] = "splicing"
    with pytest.raises(ValueError, match="span multiple subsets"):
        compute_grouped_vep_metrics(
            dataset,
            scores,
            n_bootstrap=5,
            n_min=1,
        )


def test_single_group_scope_reports_group_smd_unavailable():
    small_dataset, small_scores = _grouped_data(
        subsets=("single",), n_groups_per_subset=1
    )
    large_dataset, large_scores = _grouped_data(
        subsets=("multiple",), n_groups_per_subset=4
    )
    large_dataset["match_group"] += 1
    dataset = pd.concat([small_dataset, large_dataset], ignore_index=True)
    scores = pd.concat([small_scores, large_scores], ignore_index=True)

    summary, _ = compute_grouped_vep_metrics(
        dataset,
        scores,
        score_columns=["score"],
        n_bootstrap=20,
        rng=4,
        n_min=1,
    )
    row = summary.loc[
        (summary["metric"] == GROUP_SMD) & (summary["subset"] == "single")
    ].iloc[0]
    assert not row["available"]
    assert row["unavailable_reason"] == "requires_at_least_two_match_groups"
    assert np.isnan(row["value"])
    assert np.isnan(row["ci_low"])


def test_standalone_single_group_input_raises_explicit_macro_validation():
    dataset, scores = _grouped_data(n_groups_per_subset=1)
    with pytest.raises(ValueError, match="no subsets meet n_min=30"):
        compute_grouped_vep_metrics(
            dataset,
            scores,
            score_columns=["score"],
            n_bootstrap=5,
        )


def test_zero_group_gap_sd_reports_group_smd_unavailable():
    dataset = pd.DataFrame(
        {
            "label": [1, 0, 1, 0, 1, 0],
            "subset": ["coding"] * 6,
            "match_group": [0, 0, 1, 1, 2, 2],
        }
    )
    scores = pd.DataFrame({"score": [1.0, 0.0, 2.0, 1.0, 3.0, 2.0]})
    summary, _ = compute_grouped_vep_metrics(
        dataset,
        scores,
        n_bootstrap=10,
        rng=7,
        n_min=1,
    )
    row = summary.loc[
        (summary["metric"] == GROUP_SMD) & (summary["subset"] == "coding")
    ].iloc[0]
    assert not row["available"]
    assert row["unavailable_reason"] == "zero_or_non_finite_group_gap_sd"
    assert row["n_bootstrap_valid"] == 0


def test_joint_bootstrap_aligns_scores_and_metrics_on_one_group_draw():
    dataset, scores = _grouped_data(n_groups_per_subset=5)
    n_bootstrap = 12
    seed = 23
    summary, samples = compute_grouped_vep_metrics(
        dataset,
        scores,
        n_bootstrap=n_bootstrap,
        rng=seed,
        n_min=1,
    )

    for metric in (AUPRC, GROUP_SMD):
        score_samples = samples.loc[
            (samples["subset"] == "coding")
            & (samples["score_type"] == "score")
            & (samples["metric"] == metric)
        ].sort_values("draw")
        identical_samples = samples.loc[
            (samples["subset"] == "coding")
            & (samples["score_type"] == "identical_score")
            & (samples["metric"] == metric)
        ].sort_values("draw")
        assert score_samples["value"].to_numpy() == pytest.approx(
            identical_samples["value"].to_numpy(), nan_ok=True
        )

    group_rows = list(
        dataset["match_group"]
        .groupby(dataset["match_group"], sort=True)
        .indices.values()
    )
    generator = np.random.default_rng(seed)
    sampled_groups = generator.integers(0, len(group_rows), size=len(group_rows))
    sampled_rows = np.concatenate([group_rows[group] for group in sampled_groups])
    sampled_dataset = dataset.iloc[sampled_rows]
    sampled_score = scores["score"].iloc[sampled_rows]
    observed = samples.loc[
        (samples["subset"] == "coding")
        & (samples["score_type"] == "score")
        & (samples["draw"] == 0)
    ].set_index("metric")["value"]
    assert observed[AUPRC] == pytest.approx(
        average_precision_score(sampled_dataset["label"], sampled_score)
    )
    gaps = matched_group_gaps(
        dataset["label"], scores["score"], dataset["match_group"]
    ).to_numpy()
    sampled_gaps = gaps[sampled_groups]
    assert observed[GROUP_SMD] == pytest.approx(
        sampled_gaps.mean() / sampled_gaps.std(ddof=1)
    )

    group_rows = summary.loc[summary["metric"] == GROUP_SMD]
    direct_rows = group_rows.loc[group_rows["subset"] != MACRO_AVG_SUBSET]
    assert direct_rows["higher_is_better"].all()
    assert direct_rows["available"].all()
    assert direct_rows["ci_low"].notna().all()
    assert direct_rows["ci_high"].notna().all()


def test_integer_seed_aligns_bootstrap_draws_across_model_outputs():
    dataset, scores = _grouped_data(n_groups_per_subset=7)
    _, first_samples = compute_grouped_vep_metrics(
        dataset,
        scores[["score"]],
        n_bootstrap=15,
        rng=31,
        n_min=1,
    )
    _, second_samples = compute_grouped_vep_metrics(
        dataset,
        pd.DataFrame({"score": 4.0 * scores["score"] + 9.0}),
        n_bootstrap=15,
        rng=31,
        n_min=1,
    )

    sort_columns = ["subset", "metric", "draw"]
    first_values = first_samples.sort_values(sort_columns)["value"].to_numpy()
    second_values = second_samples.sort_values(sort_columns)["value"].to_numpy()
    assert first_values == pytest.approx(second_values, nan_ok=True)


def test_grouped_report_preserves_existing_auprc_output():
    dataset, scores = _grouped_data(
        subsets=("coding", "splicing"), n_groups_per_subset=4
    )
    legacy = compute_auprc_metrics(
        dataset,
        scores,
        score_columns=["score", "identical_score"],
        n_bootstrap=25,
        rng=9,
        n_min=1,
    )
    summary, _ = compute_grouped_vep_metrics(
        dataset,
        scores,
        score_columns=["score", "identical_score"],
        n_bootstrap=25,
        rng=9,
        n_min=1,
    )
    observed = summary.loc[summary["metric"] == AUPRC, legacy.columns]
    pd.testing.assert_frame_equal(observed.reset_index(drop=True), legacy)
    macro_auprc = summary.loc[
        (summary["metric"] == AUPRC) & (summary["subset"] == MACRO_AVG_SUBSET)
    ]
    assert macro_auprc["n_bootstrap_valid"].eq(0).all()
    assert (
        macro_auprc["uncertainty_method"]
        .eq("independent_subset_bootstrap_se_of_mean")
        .all()
    )

    macro_smd = summary.loc[
        (summary["metric"] == GROUP_SMD) & (summary["subset"] == MACRO_AVG_SUBSET)
    ]
    assert macro_smd["available"].eq(False).all()
    assert (
        macro_smd["unavailable_reason"]
        .eq("group_smd_not_defined_for_macro_average")
        .all()
    )
