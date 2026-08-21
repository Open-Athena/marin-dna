"""Contracts for additive Group SMD columns on matched-pair AUPRC rows."""

import numpy as np
import pandas as pd
import pytest
from marin_dna_evals.grouped_vep_metrics import (
    GROUP_SMD_COLUMNS,
    _joint_group_smd_bootstrap,
    compute_grouped_vep_metrics,
    group_smd,
    matched_group_gaps,
)
from marin_dna_evals.metrics import MACRO_AVG_SUBSET, compute_auprc_metrics


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


def test_group_smd_matches_positive_minus_mean_negative_formula() -> None:
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


def test_group_smd_is_positive_affine_invariant() -> None:
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
        (lambda dataset: dataset.assign(match_group=0), "exactly one positive"),
        (
            lambda dataset: dataset.assign(
                match_group=dataset["match_group"].mask(dataset.index == 0, 1)
            ),
            "exactly one positive",
        ),
    ],
)
def test_group_contract_validation_fails_loud(dataset_mutation, message: str) -> None:
    dataset, scores = _grouped_data(n_groups_per_subset=3)
    with pytest.raises(ValueError, match=message):
        compute_grouped_vep_metrics(
            dataset_mutation(dataset),
            scores,
            n_bootstrap=5,
            n_min=1,
        )


def test_match_group_may_not_span_subsets() -> None:
    dataset, scores = _grouped_data(subsets=("coding", "splicing"))
    dataset.loc[dataset.index[0], "subset"] = "splicing"
    with pytest.raises(ValueError, match="span multiple subsets"):
        compute_grouped_vep_metrics(dataset, scores, n_bootstrap=5, n_min=1)


def test_single_group_scope_reports_group_smd_unavailable() -> None:
    small_dataset, small_scores = _grouped_data(
        subsets=("single",), n_groups_per_subset=1
    )
    large_dataset, large_scores = _grouped_data(
        subsets=("multiple",), n_groups_per_subset=4
    )
    large_dataset["match_group"] += 1
    dataset = pd.concat([small_dataset, large_dataset], ignore_index=True)
    scores = pd.concat([small_scores, large_scores], ignore_index=True)

    metrics = compute_grouped_vep_metrics(
        dataset,
        scores,
        score_columns=["score"],
        n_bootstrap=20,
        rng=4,
        n_min=1,
    )
    row = metrics.loc[metrics["subset"] == "single"].iloc[0]
    assert not row["group_smd_available"]
    assert row["group_smd_unavailable_reason"] == "requires_at_least_two_match_groups"
    assert np.isnan(row["group_smd_value"])
    assert np.isnan(row["group_smd_ci_low"])


def test_standalone_single_group_input_raises_explicit_macro_validation() -> None:
    dataset, scores = _grouped_data(n_groups_per_subset=1)
    with pytest.raises(ValueError, match="no subsets meet n_min=30"):
        compute_grouped_vep_metrics(
            dataset,
            scores,
            score_columns=["score"],
            n_bootstrap=5,
        )


def test_zero_group_gap_sd_reports_group_smd_unavailable() -> None:
    dataset = pd.DataFrame(
        {
            "label": [1, 0, 1, 0, 1, 0],
            "subset": ["coding"] * 6,
            "match_group": [0, 0, 1, 1, 2, 2],
        }
    )
    scores = pd.DataFrame({"score": [1.0, 0.0, 2.0, 1.0, 3.0, 2.0]})
    metrics = compute_grouped_vep_metrics(
        dataset,
        scores,
        n_bootstrap=10,
        rng=7,
        n_min=1,
    )
    row = metrics.loc[metrics["subset"] == "coding"].iloc[0]
    assert not row["group_smd_available"]
    assert row["group_smd_unavailable_reason"] == "zero_or_non_finite_group_gap_sd"
    assert row["group_smd_n_bootstrap_valid"] == 0


def test_joint_bootstrap_reuses_one_group_draw_across_scores() -> None:
    dataset, scores = _grouped_data(n_groups_per_subset=5)
    frame = pd.concat([dataset, scores], axis=1)
    seed = 23
    samples = _joint_group_smd_bootstrap(
        frame,
        ["score", "identical_score"],
        n_bootstrap=12,
        generator=np.random.default_rng(seed),
    )
    assert samples["score"] == pytest.approx(samples["identical_score"], nan_ok=True)

    gaps = matched_group_gaps(
        dataset["label"], scores["score"], dataset["match_group"]
    ).to_numpy()
    generator = np.random.default_rng(seed)
    sampled_groups = generator.integers(0, len(gaps), size=len(gaps))
    sampled_gaps = gaps[sampled_groups]
    assert samples["score"][0] == pytest.approx(
        sampled_gaps.mean() / sampled_gaps.std(ddof=1)
    )


def test_enrichment_preserves_existing_auprc_rows_and_columns() -> None:
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
    enriched = compute_grouped_vep_metrics(
        dataset,
        scores,
        score_columns=["score", "identical_score"],
        n_bootstrap=25,
        rng=9,
        n_min=1,
    )

    pd.testing.assert_frame_equal(enriched[legacy.columns], legacy)
    assert list(enriched.columns) == [*legacy.columns, *GROUP_SMD_COLUMNS]
    assert "metric" not in enriched.columns

    direct = enriched.loc[enriched["subset"] != MACRO_AVG_SUBSET]
    assert direct["group_smd_available"].all()
    assert direct["group_smd_ci_low"].notna().all()
    assert direct["group_smd_ci_high"].notna().all()
    assert direct["group_smd_n_bootstrap_valid"].gt(0).all()

    macro = enriched.loc[enriched["subset"] == MACRO_AVG_SUBSET]
    assert macro["group_smd_available"].eq(False).all()
    assert (
        macro["group_smd_unavailable_reason"]
        .eq("group_smd_not_defined_for_macro_average")
        .all()
    )


def test_group_smd_uncertainty_is_seed_reproducible() -> None:
    dataset, scores = _grouped_data(n_groups_per_subset=7)
    first = compute_grouped_vep_metrics(
        dataset,
        scores[["score"]],
        n_bootstrap=15,
        rng=31,
        n_min=1,
    )
    second = compute_grouped_vep_metrics(
        dataset,
        scores[["score"]],
        n_bootstrap=15,
        rng=31,
        n_min=1,
    )
    pd.testing.assert_frame_equal(first, second)


def test_zero_bootstrap_keeps_point_estimate_without_uncertainty() -> None:
    dataset, scores = _grouped_data(n_groups_per_subset=4)
    metrics = compute_grouped_vep_metrics(
        dataset,
        scores[["score"]],
        n_bootstrap=0,
        n_min=1,
    )
    direct = metrics.loc[metrics["subset"] != MACRO_AVG_SUBSET]
    assert direct["group_smd_available"].all()
    assert direct["group_smd_value"].notna().all()
    assert direct["group_smd_se"].isna().all()
    assert direct["group_smd_n_bootstrap_valid"].eq(0).all()
