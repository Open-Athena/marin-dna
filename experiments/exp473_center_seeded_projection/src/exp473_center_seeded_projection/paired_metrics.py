"""Paired development metrics for issue #473 projection-policy comparisons.

This is an experiment-local copy of the Group SMD contract selected in issue
#459.  It intentionally does not modify the maintained ``evals_v2`` metrics
rules.  One joint match-group bootstrap draw is applied to both projection
policies, which makes the reported policy delta paired at the registered
sampling unit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

AUPRC = "auprc"
GROUP_SMD = "group_smd"
METRICS = (AUPRC, GROUP_SMD)


@dataclass(frozen=True)
class BootstrapResult:
    """Point estimates and aligned bootstrap draws for one subset."""

    point: pd.DataFrame
    samples: pd.DataFrame
    deltas: pd.DataFrame


def validate_matched_frame(
    label: pd.Series | np.ndarray,
    scores: pd.DataFrame,
    match_group: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Normalize one matched 1:k frame and fail on incompatible inputs."""
    assert len(scores.columns) >= 2, "paired analysis requires at least two policies"
    assert len(label) == len(scores) == len(match_group), (
        f"length mismatch: label={len(label)} scores={len(scores)} "
        f"match_group={len(match_group)}"
    )
    frame = pd.DataFrame(
        {
            "label": np.asarray(label).astype(int),
            "match_group": np.asarray(match_group),
        }
    )
    assert len(frame) > 0, "paired metrics require at least one row"
    assert frame["match_group"].notna().all(), "match_group contains null values"
    assert set(frame["label"].unique()) == {0, 1}, "labels must contain 0 and 1"
    for column in scores:
        values = np.asarray(scores[column], dtype=float)
        assert np.isfinite(values).all(), f"{column}: score contains non-finite values"
        frame[column] = values

    counts = frame.groupby("match_group", sort=False)["label"].agg(
        n_rows="size", n_pos="sum"
    )
    bad = counts[(counts["n_pos"] != 1) | (counts["n_rows"] < 2)]
    assert bad.empty, (
        "each match_group must contain exactly one positive and at least one "
        f"negative; got {len(bad)} bad groups; first={bad.head().to_dict('index')}"
    )
    assert len(counts) >= 2, "Group SMD requires at least two match groups"
    return frame


def group_differences(frame: pd.DataFrame, score_column: str) -> np.ndarray:
    """Return positive minus mean-negative score for every match group."""
    positive = (
        frame.loc[frame["label"] == 1]
        .set_index("match_group")[score_column]
        .rename("positive")
    )
    negative = (
        frame.loc[frame["label"] == 0]
        .groupby("match_group", sort=False)[score_column]
        .mean()
        .rename("negative_mean")
    )
    joined = positive.to_frame().join(negative)
    assert joined.notna().all(axis=None), "positive/negative match-group sets differ"
    return (joined["positive"] - joined["negative_mean"]).to_numpy(dtype=float)


def group_smd(frame: pd.DataFrame, score_column: str) -> float:
    """Mean matched-group gap divided by its across-group sample SD."""
    differences = group_differences(frame, score_column)
    scale = float(np.std(differences, ddof=1))
    return float(np.mean(differences) / scale) if scale > 0 else float("nan")


def average_precision(label: np.ndarray, score: np.ndarray) -> float:
    """Tie-aware binary average precision, matching sklearn semantics."""
    label = np.asarray(label).astype(int)
    score = np.asarray(score, dtype=float)
    assert len(label) == len(score)
    assert set(np.unique(label)) == {0, 1}
    assert np.isfinite(score).all()
    order = np.argsort(-score, kind="mergesort")
    sorted_score = score[order]
    sorted_label = label[order]
    threshold_ends = np.r_[
        np.flatnonzero(sorted_score[:-1] != sorted_score[1:]),
        len(sorted_score) - 1,
    ]
    true_positive = np.cumsum(sorted_label)[threshold_ends]
    predicted = threshold_ends + 1
    precision = true_positive / predicted
    recall = true_positive / true_positive[-1]
    recall_increment = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_increment * precision))


def _weighted_bootstrap_auprc(
    label: np.ndarray,
    score: np.ndarray,
    row_group_index: np.ndarray,
    group_multiplicity: np.ndarray,
    *,
    batch_size: int = 100,
) -> np.ndarray:
    """Exact AP for cluster-weight draws, sorting the score only once."""
    order = np.argsort(-score, kind="mergesort")
    sorted_score = score[order]
    sorted_label = label[order].astype(float)
    sorted_group = row_group_index[order]
    threshold_ends = np.r_[
        np.flatnonzero(sorted_score[:-1] != sorted_score[1:]),
        len(sorted_score) - 1,
    ]

    result = np.empty(len(group_multiplicity), dtype=float)
    for start in range(0, len(group_multiplicity), batch_size):
        stop = min(start + batch_size, len(group_multiplicity))
        weights = group_multiplicity[start:stop, sorted_group].astype(float)
        true_positive = np.cumsum(weights * sorted_label, axis=1)[:, threshold_ends]
        predicted = np.cumsum(weights, axis=1)[:, threshold_ends]
        precision = np.divide(
            true_positive,
            predicted,
            out=np.ones_like(true_positive),
            where=predicted > 0,
        )
        total_positive = true_positive[:, -1:]
        recall = true_positive / total_positive
        recall_increment = np.diff(
            np.concatenate([np.zeros((len(recall), 1)), recall], axis=1), axis=1
        )
        result[start:stop] = np.sum(recall_increment * precision, axis=1)
    return result


def _interval(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    assert len(finite) > 0, "no finite bootstrap draws"
    low, high = np.percentile(finite, [2.5, 97.5])
    return {
        "se": float(np.std(finite, ddof=1)),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def paired_policy_bootstrap(
    label: pd.Series | np.ndarray,
    scores: pd.DataFrame,
    match_group: pd.Series | np.ndarray,
    *,
    center_column: str = "center_1",
    full_column: str = "full_window",
    n_bootstrap: int = 1_000,
    seed: int = 473,
) -> BootstrapResult:
    """Compute AUPRC, Group SMD, and center-minus-full paired intervals.

    The random seed should remain fixed for a consequence subset across every
    checkpoint.  Because evaluation rows are identical, that makes trajectory
    uncertainty aligned as well as the within-checkpoint policy delta.
    """
    assert n_bootstrap > 1, "bootstrap uncertainty requires at least two draws"
    assert center_column in scores and full_column in scores
    columns = [full_column, center_column]
    frame = validate_matched_frame(label, scores[columns], match_group)
    labels = frame["label"].to_numpy(dtype=int)
    original_groups = frame["match_group"].to_numpy()
    group_rows = list(
        pd.Series(original_groups).groupby(original_groups, sort=False).indices.values()
    )
    n_groups = len(group_rows)
    row_group_index = np.empty(len(frame), dtype=int)
    for index, rows in enumerate(group_rows):
        row_group_index[rows] = index

    generator = np.random.default_rng(seed)
    sampled = generator.integers(0, n_groups, size=(n_bootstrap, n_groups))
    multiplicity = np.zeros((n_bootstrap, n_groups), dtype=np.int32)
    np.add.at(
        multiplicity,
        (np.repeat(np.arange(n_bootstrap), n_groups), sampled.ravel()),
        1,
    )

    point_rows: list[dict[str, str | float | int]] = []
    sample_rows: list[dict[str, str | float | int]] = []
    sample_arrays: dict[tuple[str, str], np.ndarray] = {}
    for column in columns:
        values = frame[column].to_numpy(dtype=float)
        differences = group_differences(frame, column)
        group_mean = multiplicity @ differences / n_groups
        sum_squares = multiplicity @ np.square(differences)
        variance = np.maximum(
            (sum_squares - n_groups * np.square(group_mean)) / (n_groups - 1),
            0.0,
        )
        scale = np.sqrt(variance)
        group_smd_draws = np.divide(
            group_mean,
            scale,
            out=np.full(n_bootstrap, np.nan),
            where=scale > 0,
        )
        auprc_draws = _weighted_bootstrap_auprc(
            labels, values, row_group_index, multiplicity
        )
        metric_values = {
            AUPRC: average_precision(labels, values),
            GROUP_SMD: group_smd(frame, column),
        }
        metric_draws = {AUPRC: auprc_draws, GROUP_SMD: group_smd_draws}
        for metric in METRICS:
            draws = metric_draws[metric]
            sample_arrays[(column, metric)] = draws
            point_rows.append(
                {
                    "policy": column,
                    "metric": metric,
                    "value": metric_values[metric],
                    "n_groups": n_groups,
                    "n_rows": len(frame),
                    **_interval(draws),
                }
            )
            sample_rows.extend(
                {
                    "draw": draw,
                    "policy": column,
                    "metric": metric,
                    "value": float(value),
                }
                for draw, value in enumerate(draws)
            )

    point = pd.DataFrame(point_rows)
    samples = pd.DataFrame(sample_rows)
    delta_rows: list[dict[str, str | float | int]] = []
    point_index = point.set_index(["policy", "metric"])["value"]
    for metric in METRICS:
        delta = (
            sample_arrays[(center_column, metric)]
            - sample_arrays[(full_column, metric)]
        )
        finite = delta[np.isfinite(delta)]
        delta_rows.append(
            {
                "metric": metric,
                "delta_center_minus_full": float(
                    point_index.loc[(center_column, metric)]
                    - point_index.loc[(full_column, metric)]
                ),
                **_interval(delta),
                "probability_center_better": float(np.mean(finite > 0)),
                "n_bootstrap": len(finite),
                "bootstrap_unit": "match_group",
            }
        )
    return BootstrapResult(
        point=point,
        samples=samples,
        deltas=pd.DataFrame(delta_rows),
    )
