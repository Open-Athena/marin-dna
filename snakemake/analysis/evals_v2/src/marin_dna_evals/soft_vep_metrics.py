"""Soft Mendelian VEP metrics for issue #459.

The existing leaderboard endpoint is AUPRC over the FWD/RC-averaged ``-LLR``
score. AUPRC only observes ordering. This module keeps that endpoint beside
candidate summaries that also observe score magnitude, while retaining the
matched 1:k sampling unit for uncertainty.

All functions operate on one consequence subset at a time. Every
``match_group`` must contain exactly one positive and at least one negative.
The point metrics use ordinary row- or group-level formulas; the joint
bootstrap resamples whole groups and applies the same draw to every supplied
score column.
"""

from collections.abc import Collection

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

AUPRC = "auprc"
MEAN_GAP_GLOBAL = "mean_gap_global"
MEAN_GAP_GROUP = "mean_gap_group"
GROUP_SMD = "group_smd"
GROUP_MEDIAN_MAD = "group_median_mad"
SOFT_WIN = "soft_win"
CALIBRATED_LOG_LOSS = "calibrated_log_loss"
CALIBRATED_BRIER = "calibrated_brier"
VARIANT_POOLED_SMD = "variant_pooled_smd"
VARIANT_TOTAL_SD_GAP = "variant_total_sd_gap"
STUDENT_T = "student_t"
WELCH_T = "welch_t"

CALIBRATED_METRICS = frozenset({CALIBRATED_LOG_LOSS, CALIBRATED_BRIER})
DEFAULT_METRICS = (
    AUPRC,
    MEAN_GAP_GLOBAL,
    MEAN_GAP_GROUP,
    GROUP_SMD,
    GROUP_MEDIAN_MAD,
    SOFT_WIN,
    CALIBRATED_LOG_LOSS,
    CALIBRATED_BRIER,
)
UNGROUPED_METRICS = (
    AUPRC,
    VARIANT_POOLED_SMD,
    VARIANT_TOTAL_SD_GAP,
    STUDENT_T,
    WELCH_T,
)
HIGHER_IS_BETTER = {
    AUPRC: True,
    MEAN_GAP_GLOBAL: True,
    MEAN_GAP_GROUP: True,
    GROUP_SMD: True,
    GROUP_MEDIAN_MAD: True,
    SOFT_WIN: True,
    CALIBRATED_LOG_LOSS: False,
    CALIBRATED_BRIER: False,
    VARIANT_POOLED_SMD: True,
    VARIANT_TOTAL_SD_GAP: True,
    STUDENT_T: True,
    WELCH_T: True,
}


def _validated_matched_frame(
    label: pd.Series | np.ndarray,
    score: pd.Series | np.ndarray,
    match_group: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Return a normalized matched frame and fail on ambiguous inputs."""
    assert len(label) == len(score) == len(match_group), (
        f"length mismatch: label={len(label)} score={len(score)} "
        f"match_group={len(match_group)}"
    )
    frame = pd.DataFrame(
        {
            "label": np.asarray(label).astype(int),
            "score": np.asarray(score, dtype=float),
            "match_group": np.asarray(match_group),
        }
    )
    assert len(frame) > 0, "soft VEP metrics require at least one row"
    assert frame["match_group"].notna().all(), "match_group contains null values"
    assert np.isfinite(frame["score"]).all(), "score contains non-finite values"
    label_values = set(frame["label"].unique())
    assert label_values == {0, 1}, (
        f"soft VEP metrics require both binary classes, got {sorted(label_values)}"
    )

    group_counts = frame.groupby("match_group", sort=False)["label"].agg(
        n_rows="size", n_pos="sum"
    )
    bad = group_counts[(group_counts["n_pos"] != 1) | (group_counts["n_rows"] < 2)]
    assert bad.empty, (
        "each match_group must contain exactly one positive and at least one "
        f"negative; got {len(bad)} bad groups; first: {bad.head().to_dict('index')}"
    )
    return frame


def group_differences(
    label: pd.Series | np.ndarray,
    score: pd.Series | np.ndarray,
    match_group: pd.Series | np.ndarray,
) -> pd.Series:
    """Positive score minus the within-group mean negative score."""
    frame = _validated_matched_frame(label, score, match_group)
    positives = (
        frame.loc[frame["label"] == 1]
        .set_index("match_group")["score"]
        .rename("positive")
    )
    negatives = (
        frame.loc[frame["label"] == 0]
        .groupby("match_group", sort=False)["score"]
        .mean()
        .rename("negative_mean")
    )
    differences = positives.to_frame().join(negatives)
    assert differences.notna().all(axis=None), "positive/negative group sets differ"
    return (differences["positive"] - differences["negative_mean"]).rename(
        "group_difference"
    )


def within_group_pairwise_margins(
    label: pd.Series | np.ndarray,
    score: pd.Series | np.ndarray,
    match_group: pd.Series | np.ndarray,
) -> np.ndarray:
    """Return every positive-minus-negative margin within its match group."""
    frame = _validated_matched_frame(label, score, match_group)
    positive_by_group = (
        frame.loc[frame["label"] == 1].set_index("match_group")["score"].to_dict()
    )
    negative_rows = frame.loc[frame["label"] == 0, ["match_group", "score"]]
    positive_scores = negative_rows["match_group"].map(positive_by_group).to_numpy()
    return positive_scores - negative_rows["score"].to_numpy()


def reference_soft_win_temperature(
    label: pd.Series | np.ndarray,
    score: pd.Series | np.ndarray,
    match_group: pd.Series | np.ndarray,
) -> float:
    """Choose one robust ``SoftWin`` temperature from a named reference cell.

    The temperature is the median absolute within-group pairwise margin. It is
    intentionally computed outside the metric table so an analysis must name
    one reference distribution and reuse the resulting scalar for every model.
    """
    margins = within_group_pairwise_margins(label, score, match_group)
    tau = float(np.median(np.abs(margins)))
    assert np.isfinite(tau) and tau > 0, (
        "reference SoftWin temperature is zero or non-finite; choose a "
        "non-degenerate named reference distribution"
    )
    return tau


def grouped_calibration_scores(
    label: pd.Series | np.ndarray,
    score: pd.Series | np.ndarray,
    match_group: pd.Series | np.ndarray,
    *,
    n_splits: int = 5,
) -> dict[str, float]:
    """Cross-validated one-dimensional logistic calibration proper scores.

    All rows from a ``match_group`` stay in the same fold. The calibrator is a
    ``StandardScaler`` followed by unweighted L2 logistic regression, so it
    retains the observed matched-set prevalence and is invariant to positive
    affine rescaling of the input up to numerical precision.
    """
    frame, probabilities = grouped_calibration_probabilities(
        label, score, match_group, n_splits=n_splits
    )
    y = frame["label"].to_numpy()
    return {
        CALIBRATED_LOG_LOSS: float(log_loss(y, probabilities, labels=[0, 1])),
        CALIBRATED_BRIER: float(np.mean(np.square(probabilities - y))),
    }


def grouped_calibration_probabilities(
    label: pd.Series | np.ndarray,
    score: pd.Series | np.ndarray,
    match_group: pd.Series | np.ndarray,
    *,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return the validated rows and out-of-fold calibrated probabilities."""
    frame = _validated_matched_frame(label, score, match_group)
    unique_groups = frame["match_group"].nunique()
    folds = min(n_splits, unique_groups)
    assert folds >= 2, "grouped calibration requires at least two match groups"

    x = frame[["score"]].to_numpy()
    y = frame["label"].to_numpy()
    groups = frame["match_group"].to_numpy()
    probabilities = np.empty(len(frame), dtype=float)
    splitter = GroupKFold(n_splits=folds)
    for train_idx, test_idx in splitter.split(x, y, groups):
        calibrator = make_pipeline(
            StandardScaler(),
            LogisticRegression(l1_ratio=0.0, max_iter=1000),
        )
        calibrator.fit(x[train_idx], y[train_idx])
        probabilities[test_idx] = calibrator.predict_proba(x[test_idx])[:, 1]

    return frame, probabilities


def compute_mendelian_soft_metrics(
    label: pd.Series | np.ndarray,
    score: pd.Series | np.ndarray,
    match_group: pd.Series | np.ndarray,
    *,
    tau: float,
    metrics: Collection[str] = DEFAULT_METRICS,
    calibration_folds: int = 5,
) -> dict[str, float]:
    """Compute the issue #459 metric panel for one score vector and subset."""
    assert np.isfinite(tau) and tau > 0, f"tau must be positive and finite, got {tau}"
    requested = tuple(metrics)
    unknown = set(requested) - set(DEFAULT_METRICS)
    assert not unknown, f"unknown soft VEP metrics: {sorted(unknown)}"

    frame = _validated_matched_frame(label, score, match_group)
    y = frame["label"].to_numpy()
    values = frame["score"].to_numpy()
    differences = group_differences(y, values, frame["match_group"])
    margins = within_group_pairwise_margins(y, values, frame["match_group"])

    result: dict[str, float] = {}
    if AUPRC in requested:
        result[AUPRC] = float(average_precision_score(y, values))
    if MEAN_GAP_GLOBAL in requested:
        result[MEAN_GAP_GLOBAL] = float(values[y == 1].mean() - values[y == 0].mean())
    if MEAN_GAP_GROUP in requested:
        result[MEAN_GAP_GROUP] = float(differences.mean())
    if GROUP_SMD in requested:
        sd = float(differences.std(ddof=1))
        result[GROUP_SMD] = float(differences.mean() / sd) if sd > 0 else float("nan")
    if GROUP_MEDIAN_MAD in requested:
        median = float(differences.median())
        mad = float(np.median(np.abs(differences.to_numpy() - median)))
        robust_scale = 1.4826 * mad
        result[GROUP_MEDIAN_MAD] = (
            float(median / robust_scale) if robust_scale > 0 else float("nan")
        )
    if SOFT_WIN in requested:
        result[SOFT_WIN] = float(expit(margins / tau).mean())

    if CALIBRATED_METRICS.intersection(requested):
        calibrated = grouped_calibration_scores(
            y,
            values,
            frame["match_group"],
            n_splits=calibration_folds,
        )
        for metric in CALIBRATED_METRICS.intersection(requested):
            result[metric] = calibrated[metric]
    return {metric: result[metric] for metric in requested}


def compute_mendelian_soft_metric_table(
    label: pd.Series | np.ndarray,
    scores: pd.DataFrame,
    match_group: pd.Series | np.ndarray,
    *,
    tau: float,
    score_columns: list[str] | None = None,
    metrics: Collection[str] = DEFAULT_METRICS,
    calibration_folds: int = 5,
) -> pd.DataFrame:
    """Long-form point metric table for multiple row-aligned score columns."""
    columns = list(scores.columns) if score_columns is None else score_columns
    assert columns, "at least one score column is required"
    missing = set(columns) - set(scores.columns)
    assert not missing, f"score columns missing from frame: {sorted(missing)}"

    n_groups = int(pd.Series(match_group).nunique())
    n_rows = len(label)
    n_pos = int(np.asarray(label).astype(int).sum())
    rows: list[dict[str, str | bool | float | int]] = []
    for score_column in columns:
        values = compute_mendelian_soft_metrics(
            label,
            scores[score_column],
            match_group,
            tau=tau,
            metrics=metrics,
            calibration_folds=calibration_folds,
        )
        rows.extend(
            {
                "score_type": score_column,
                "metric": metric,
                "value": value,
                "higher_is_better": HIGHER_IS_BETTER[metric],
                "n_groups": n_groups,
                "n_rows": n_rows,
                "n_pos": n_pos,
            }
            for metric, value in values.items()
        )
    return pd.DataFrame(rows)


def _validated_binary_frame(
    label: pd.Series | np.ndarray,
    score: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Return finite binary rows without requiring a match-group column."""
    assert len(label) == len(score), (
        f"length mismatch: label={len(label)} score={len(score)}"
    )
    frame = pd.DataFrame(
        {
            "label": np.asarray(label).astype(int),
            "score": np.asarray(score, dtype=float),
        }
    )
    assert len(frame) > 0, "ungrouped VEP metrics require at least one row"
    assert np.isfinite(frame["score"]).all(), "score contains non-finite values"
    label_values = set(frame["label"].unique())
    assert label_values == {0, 1}, (
        f"ungrouped VEP metrics require both binary classes, got {sorted(label_values)}"
    )
    counts = frame["label"].value_counts()
    assert (counts >= 2).all(), (
        "ungrouped variance metrics require at least two rows in each class"
    )
    return frame


def _ungrouped_metric_values(
    label: np.ndarray,
    score: np.ndarray,
    metrics: Collection[str],
) -> dict[str, float]:
    """Compute no-match-group statistics from two variant-level score samples."""
    requested = tuple(metrics)
    positive = score[label == 1]
    negative = score[label == 0]
    n_positive = len(positive)
    n_negative = len(negative)
    positive_variance = float(positive.var(ddof=1))
    negative_variance = float(negative.var(ddof=1))
    gap = float(positive.mean() - negative.mean())
    pooled_variance = (
        (n_positive - 1) * positive_variance
        + (n_negative - 1) * negative_variance
    ) / (n_positive + n_negative - 2)
    total_sd = float(score.std(ddof=1))
    pooled_se = float(
        np.sqrt(pooled_variance * (1.0 / n_positive + 1.0 / n_negative))
    )
    welch_se = float(
        np.sqrt(positive_variance / n_positive + negative_variance / n_negative)
    )

    values: dict[str, float] = {}
    if AUPRC in requested:
        values[AUPRC] = float(average_precision_score(label, score))
    if VARIANT_POOLED_SMD in requested:
        pooled_sd = float(np.sqrt(pooled_variance))
        values[VARIANT_POOLED_SMD] = gap / pooled_sd if pooled_sd > 0 else float("nan")
    if VARIANT_TOTAL_SD_GAP in requested:
        values[VARIANT_TOTAL_SD_GAP] = gap / total_sd if total_sd > 0 else float("nan")
    if STUDENT_T in requested:
        values[STUDENT_T] = gap / pooled_se if pooled_se > 0 else float("nan")
    if WELCH_T in requested:
        values[WELCH_T] = gap / welch_se if welch_se > 0 else float("nan")
    return {metric: values[metric] for metric in requested}


def compute_ungrouped_metric_table(
    label: pd.Series | np.ndarray,
    scores: pd.DataFrame,
    *,
    score_columns: list[str] | None = None,
    metrics: Collection[str] = UNGROUPED_METRICS,
) -> pd.DataFrame:
    """Long-form no-match-group point metrics for row-aligned score columns."""
    requested = tuple(metrics)
    unknown = set(requested) - set(UNGROUPED_METRICS)
    assert not unknown, f"unknown ungrouped VEP metrics: {sorted(unknown)}"
    columns = list(scores.columns) if score_columns is None else score_columns
    assert columns, "at least one score column is required"
    missing = set(columns) - set(scores.columns)
    assert not missing, f"score columns missing from frame: {sorted(missing)}"

    first = _validated_binary_frame(label, scores[columns[0]])
    y = first["label"].to_numpy()
    rows: list[dict[str, str | bool | float | int]] = []
    for column in columns:
        frame = _validated_binary_frame(label, scores[column])
        values = _ungrouped_metric_values(
            y,
            frame["score"].to_numpy(),
            requested,
        )
        rows.extend(
            {
                "score_type": column,
                "metric": metric,
                "value": value,
                "higher_is_better": HIGHER_IS_BETTER[metric],
                "n_rows": len(frame),
                "n_pos": int(y.sum()),
            }
            for metric, value in values.items()
        )
    return pd.DataFrame(rows)


def joint_stratified_row_bootstrap_ungrouped_metrics(
    label: pd.Series | np.ndarray,
    scores: pd.DataFrame,
    *,
    score_columns: list[str] | None = None,
    metrics: Collection[str] = UNGROUPED_METRICS,
    n_bootstrap: int = 1000,
    rng: np.random.Generator | int | None = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Point estimates and joint class-stratified variant bootstrap samples.

    Positive and negative variants are sampled separately with replacement, so
    every draw preserves the observed class counts. The same row multiplicities
    are applied to every score column. No match-group information is accepted or
    used.
    """
    assert n_bootstrap >= 0, f"n_bootstrap must be non-negative, got {n_bootstrap}"
    requested = tuple(metrics)
    columns = list(scores.columns) if score_columns is None else score_columns
    point = compute_ungrouped_metric_table(
        label,
        scores,
        score_columns=columns,
        metrics=requested,
    )
    if n_bootstrap == 0:
        return point, pd.DataFrame(columns=["draw", "score_type", "metric", "value"])

    first = _validated_binary_frame(label, scores[columns[0]])
    y = first["label"].to_numpy()
    score_arrays = {
        column: _validated_binary_frame(label, scores[column])["score"].to_numpy()
        for column in columns
    }
    positive_rows = np.flatnonzero(y == 1)
    negative_rows = np.flatnonzero(y == 0)
    n_positive = len(positive_rows)
    n_negative = len(negative_rows)
    n_rows = len(y)

    generator = np.random.default_rng(rng)
    row_multiplicity = np.zeros((n_bootstrap, n_rows), dtype=np.int32)
    sampled_positive = positive_rows[
        generator.integers(0, n_positive, size=(n_bootstrap, n_positive))
    ]
    np.add.at(
        row_multiplicity,
        (np.repeat(np.arange(n_bootstrap), n_positive), sampled_positive.ravel()),
        1,
    )
    del sampled_positive
    sampled_negative = negative_rows[
        generator.integers(0, n_negative, size=(n_bootstrap, n_negative))
    ]
    np.add.at(
        row_multiplicity,
        (np.repeat(np.arange(n_bootstrap), n_negative), sampled_negative.ravel()),
        1,
    )
    del sampled_negative

    rows: list[dict[str, str | float | int]] = []
    positive_indicator = (y == 1).astype(float)
    negative_indicator = (y == 0).astype(float)
    for column in columns:
        score = score_arrays[column]
        positive_sum = row_multiplicity @ (score * positive_indicator)
        negative_sum = row_multiplicity @ (score * negative_indicator)
        positive_squares = row_multiplicity @ (np.square(score) * positive_indicator)
        negative_squares = row_multiplicity @ (np.square(score) * negative_indicator)
        positive_mean = positive_sum / n_positive
        negative_mean = negative_sum / n_negative
        gap = positive_mean - negative_mean
        positive_variance = np.maximum(
            (positive_squares - np.square(positive_sum) / n_positive)
            / (n_positive - 1),
            0.0,
        )
        negative_variance = np.maximum(
            (negative_squares - np.square(negative_sum) / n_negative)
            / (n_negative - 1),
            0.0,
        )
        pooled_variance = (
            (n_positive - 1) * positive_variance
            + (n_negative - 1) * negative_variance
        ) / (n_rows - 2)
        total_sum = positive_sum + negative_sum
        total_squares = positive_squares + negative_squares
        total_variance = np.maximum(
            (total_squares - np.square(total_sum) / n_rows) / (n_rows - 1),
            0.0,
        )
        pooled_se = np.sqrt(
            pooled_variance * (1.0 / n_positive + 1.0 / n_negative)
        )
        welch_se = np.sqrt(
            positive_variance / n_positive + negative_variance / n_negative
        )

        values: dict[str, np.ndarray] = {}
        if AUPRC in requested:
            values[AUPRC] = _weighted_bootstrap_average_precision(
                y,
                score,
                np.arange(n_rows),
                row_multiplicity,
            )
        denominators = {
            VARIANT_POOLED_SMD: np.sqrt(pooled_variance),
            VARIANT_TOTAL_SD_GAP: np.sqrt(total_variance),
            STUDENT_T: pooled_se,
            WELCH_T: welch_se,
        }
        for metric, denominator in denominators.items():
            if metric in requested:
                values[metric] = np.divide(
                    gap,
                    denominator,
                    out=np.full(n_bootstrap, np.nan),
                    where=denominator > 0,
                )
        rows.extend(
            {
                "draw": draw,
                "score_type": column,
                "metric": metric,
                "value": float(values[metric][draw]),
            }
            for metric in requested
            for draw in range(n_bootstrap)
        )
    return point, pd.DataFrame(rows)


def _weighted_bootstrap_average_precision(
    label: np.ndarray,
    score: np.ndarray,
    row_group_index: np.ndarray,
    group_multiplicity: np.ndarray,
    *,
    batch_size: int = 100,
) -> np.ndarray:
    """Exact sklearn-compatible AP for many cluster-weight bootstrap draws.

    Scores are sorted once. Each bootstrap is represented by an integer weight
    per source group, which is equivalent to physically duplicating every row
    from a sampled group. Tied score thresholds are collapsed before the
    precision-recall integral, matching ``average_precision_score`` semantics.
    Draw batches bound the largest temporary array.
    """
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
        true_positives = np.cumsum(weights * sorted_label, axis=1)[:, threshold_ends]
        predicted = np.cumsum(weights, axis=1)[:, threshold_ends]
        precision = np.divide(
            true_positives,
            predicted,
            out=np.ones_like(true_positives),
            where=predicted > 0,
        )
        total_positives = true_positives[:, -1:]
        recall = true_positives / total_positives
        recall_increment = np.diff(
            np.concatenate([np.zeros((len(recall), 1)), recall], axis=1),
            axis=1,
        )
        result[start:stop] = np.sum(recall_increment * precision, axis=1)
    return result


def joint_cluster_bootstrap_soft_metrics(
    label: pd.Series | np.ndarray,
    scores: pd.DataFrame,
    match_group: pd.Series | np.ndarray,
    *,
    tau: float,
    score_columns: list[str] | None = None,
    metrics: Collection[str] = DEFAULT_METRICS,
    n_bootstrap: int = 1000,
    rng: np.random.Generator | int | None = 0,
    calibration_folds: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Point estimates and joint match-group bootstrap samples.

    Each bootstrap draw samples group positions once. That one index vector is
    applied to every score column, preserving paired model and metric
    comparisons. Repeated copies receive distinct point-metric group IDs so
    group-balanced summaries preserve bootstrap multiplicity. The original
    IDs remain the calibration groups, preventing duplicate copies of one
    source group from leaking across calibration folds. Calibration models are
    cross-fitted once on the original sample; bootstrap draws resample their
    out-of-fold row losses without refitting. The resulting uncertainty is
    conditional on the fitted one-dimensional calibrator, which keeps the
    secondary proper-score analysis tractable and avoids duplicate-group
    leakage.
    """
    assert n_bootstrap >= 0, f"n_bootstrap must be non-negative, got {n_bootstrap}"
    requested = tuple(metrics)
    columns = list(scores.columns) if score_columns is None else score_columns
    point = compute_mendelian_soft_metric_table(
        label,
        scores,
        match_group,
        tau=tau,
        score_columns=columns,
        metrics=requested,
        calibration_folds=calibration_folds,
    )
    if n_bootstrap == 0:
        return point, pd.DataFrame(columns=["draw", "score_type", "metric", "value"])

    first = _validated_matched_frame(label, scores[columns[0]], match_group)
    y = first["label"].to_numpy()
    original_group = first["match_group"].to_numpy()
    score_arrays: dict[str, np.ndarray] = {}
    calibration_probabilities: dict[str, np.ndarray] = {}
    for column in columns:
        candidate = _validated_matched_frame(label, scores[column], match_group)
        score_arrays[column] = candidate["score"].to_numpy()
        if CALIBRATED_METRICS.intersection(requested):
            _, probabilities = grouped_calibration_probabilities(
                label,
                scores[column],
                match_group,
                n_splits=calibration_folds,
            )
            calibration_probabilities[column] = probabilities

    group_to_rows = list(
        pd.Series(original_group).groupby(original_group, sort=False).indices.values()
    )
    n_groups = len(group_to_rows)
    row_group_index = np.empty(len(y), dtype=int)
    for group_index, group_rows in enumerate(group_to_rows):
        row_group_index[group_rows] = group_index

    group_atoms: dict[str, dict[str, np.ndarray]] = {}
    for column in columns:
        column_scores = score_arrays[column]
        differences = np.empty(n_groups, dtype=float)
        positive_scores = np.empty(n_groups, dtype=float)
        negative_sums = np.empty(n_groups, dtype=float)
        negative_counts = np.empty(n_groups, dtype=int)
        soft_sums = np.empty(n_groups, dtype=float)
        soft_counts = np.empty(n_groups, dtype=int)
        calibration_log_loss_sums = np.empty(n_groups, dtype=float)
        calibration_brier_sums = np.empty(n_groups, dtype=float)
        calibration_counts = np.empty(n_groups, dtype=int)
        probabilities = calibration_probabilities.get(column)
        if probabilities is not None:
            clipped = np.clip(
                probabilities,
                np.finfo(float).eps,
                1.0 - np.finfo(float).eps,
            )
            row_log_loss = -(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))
            row_brier = np.square(probabilities - y)
        for group_index, group_rows in enumerate(group_to_rows):
            group_labels = y[group_rows]
            group_scores = column_scores[group_rows]
            positive = float(group_scores[group_labels == 1][0])
            negative = group_scores[group_labels == 0]
            positive_scores[group_index] = positive
            negative_sums[group_index] = float(negative.sum())
            negative_counts[group_index] = len(negative)
            differences[group_index] = positive - float(negative.mean())
            soft_values = expit((positive - negative) / tau)
            soft_sums[group_index] = float(soft_values.sum())
            soft_counts[group_index] = len(soft_values)
            if probabilities is not None:
                calibration_log_loss_sums[group_index] = float(
                    row_log_loss[group_rows].sum()
                )
                calibration_brier_sums[group_index] = float(row_brier[group_rows].sum())
                calibration_counts[group_index] = len(group_rows)
        group_atoms[column] = {
            "difference": differences,
            "positive_score": positive_scores,
            "negative_sum": negative_sums,
            "negative_count": negative_counts,
            "soft_sum": soft_sums,
            "soft_count": soft_counts,
            "calibration_log_loss_sum": calibration_log_loss_sums,
            "calibration_brier_sum": calibration_brier_sums,
            "calibration_count": calibration_counts,
        }

    generator = np.random.default_rng(rng)
    sampled_groups = generator.integers(
        0,
        n_groups,
        size=(n_bootstrap, n_groups),
    )
    group_multiplicity = np.zeros((n_bootstrap, n_groups), dtype=np.int32)
    np.add.at(
        group_multiplicity,
        (
            np.repeat(np.arange(n_bootstrap), n_groups),
            sampled_groups.ravel(),
        ),
        1,
    )

    rows: list[dict[str, str | float | int]] = []
    for column in columns:
        atoms = group_atoms[column]
        values: dict[str, np.ndarray] = {}
        if AUPRC in requested:
            values[AUPRC] = _weighted_bootstrap_average_precision(
                y,
                score_arrays[column],
                row_group_index,
                group_multiplicity,
            )
        if MEAN_GAP_GLOBAL in requested:
            positive_mean = group_multiplicity @ atoms["positive_score"] / n_groups
            negative_mean = np.divide(
                group_multiplicity @ atoms["negative_sum"],
                group_multiplicity @ atoms["negative_count"],
            )
            values[MEAN_GAP_GLOBAL] = positive_mean - negative_mean
        if MEAN_GAP_GROUP in requested or GROUP_SMD in requested:
            group_mean = group_multiplicity @ atoms["difference"] / n_groups
            if MEAN_GAP_GROUP in requested:
                values[MEAN_GAP_GROUP] = group_mean
            if GROUP_SMD in requested:
                sum_squares = group_multiplicity @ np.square(atoms["difference"])
                variance = np.maximum(
                    (sum_squares - n_groups * np.square(group_mean)) / (n_groups - 1),
                    0.0,
                )
                sd = np.sqrt(variance)
                values[GROUP_SMD] = np.divide(
                    group_mean,
                    sd,
                    out=np.full(n_bootstrap, np.nan),
                    where=sd > 0,
                )
        if GROUP_MEDIAN_MAD in requested:
            sampled_differences = atoms["difference"][sampled_groups]
            median = np.median(sampled_differences, axis=1)
            mad = np.median(
                np.abs(sampled_differences - median[:, np.newaxis]),
                axis=1,
            )
            robust_scale = 1.4826 * mad
            values[GROUP_MEDIAN_MAD] = np.divide(
                median,
                robust_scale,
                out=np.full(n_bootstrap, np.nan),
                where=robust_scale > 0,
            )
        if SOFT_WIN in requested:
            values[SOFT_WIN] = np.divide(
                group_multiplicity @ atoms["soft_sum"],
                group_multiplicity @ atoms["soft_count"],
            )
        if CALIBRATED_LOG_LOSS in requested:
            values[CALIBRATED_LOG_LOSS] = np.divide(
                group_multiplicity @ atoms["calibration_log_loss_sum"],
                group_multiplicity @ atoms["calibration_count"],
            )
        if CALIBRATED_BRIER in requested:
            values[CALIBRATED_BRIER] = np.divide(
                group_multiplicity @ atoms["calibration_brier_sum"],
                group_multiplicity @ atoms["calibration_count"],
            )
        rows.extend(
            {
                "draw": draw,
                "score_type": column,
                "metric": metric,
                "value": float(values[metric][draw]),
            }
            for metric in requested
            for draw in range(n_bootstrap)
        )
    return point, pd.DataFrame(rows)


def summarize_joint_bootstrap(
    point: pd.DataFrame,
    bootstrap_samples: pd.DataFrame,
) -> pd.DataFrame:
    """Attach bootstrap SE and percentile intervals to a point metric table."""
    required_point = {"score_type", "metric", "value"}
    required_samples = {"score_type", "metric", "value"}
    assert required_point.issubset(point.columns), (
        f"point table missing columns: {sorted(required_point - set(point.columns))}"
    )
    assert required_samples.issubset(bootstrap_samples.columns), (
        "bootstrap table missing columns: "
        f"{sorted(required_samples - set(bootstrap_samples.columns))}"
    )
    if bootstrap_samples.empty:
        result = point.copy()
        result[["se", "ci_low", "ci_high"]] = np.nan
        return result

    uncertainty = (
        bootstrap_samples.groupby(["score_type", "metric"], sort=False)["value"]
        .agg(
            se=lambda values: float(np.nanstd(values, ddof=1)),
            ci_low=lambda values: float(np.nanpercentile(values, 2.5)),
            ci_high=lambda values: float(np.nanpercentile(values, 97.5)),
        )
        .reset_index()
    )
    return point.merge(
        uncertainty,
        on=["score_type", "metric"],
        how="left",
        validate="one_to_one",
    )
