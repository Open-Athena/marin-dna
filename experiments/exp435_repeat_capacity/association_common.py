"""Sparse exact association helpers for the issue-435 repeat panel."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
from scipy import stats
from scipy.sparse import csr_matrix

from extract_common import D_SAE


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    assert values.ndim == 1 and np.isfinite(values).all()
    assert np.all((values >= 0) & (values <= 1))
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    scaled = ranked * ranked.size / np.arange(1, ranked.size + 1)
    monotone = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted = np.empty_like(values)
    adjusted[order] = np.clip(monotone, 0.0, 1.0)
    return adjusted


def _group_counts(
    values: np.ndarray, is_positive: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert values.ndim == is_positive.ndim == 1 and values.size == is_positive.size
    assert values.size > 0 and np.all(values > 0) and np.isfinite(values).all()
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    ordered_positive = is_positive[order]
    starts = np.concatenate(
        ([0], np.flatnonzero(ordered_values[1:] != ordered_values[:-1]) + 1)
    )
    counts = np.diff(np.concatenate((starts, [values.size])))
    positive_counts = np.add.reduceat(ordered_positive.astype(np.int64), starts)
    negative_counts = counts - positive_counts
    assert int(positive_counts.sum() + negative_counts.sum()) == values.size
    return counts, positive_counts, negative_counts


def _rank_and_ap_metrics(
    values: np.ndarray,
    is_positive: np.ndarray,
    *,
    n_positive: int,
    n_negative: int,
) -> tuple[float, float, float, float]:
    positive_nonzero = int(is_positive.sum())
    negative_nonzero = int(values.size - positive_nonzero)
    positive_zero = n_positive - positive_nonzero
    negative_zero = n_negative - negative_nonzero
    assert min(positive_zero, negative_zero) >= 0
    counts, positive_counts, negative_counts = _group_counts(values, is_positive)

    negative_before = negative_zero + np.concatenate(
        ([0], np.cumsum(negative_counts[:-1], dtype=np.int64))
    )
    u_statistic = 0.5 * positive_zero * negative_zero + float(
        np.sum(positive_counts * (negative_before + 0.5 * negative_counts))
    )

    total = n_positive + n_negative
    zero_count = positive_zero + negative_zero
    tie_term = zero_count**3 - zero_count + int(np.sum(counts**3 - counts))
    variance = (n_positive * n_negative / 12.0) * (
        (total + 1) - tie_term / (total * (total - 1))
    )
    assert variance >= -1e-9
    variance = max(0.0, variance)

    descending_positive = positive_counts[::-1]
    descending_negative = negative_counts[::-1]
    cumulative_positive = np.cumsum(descending_positive, dtype=np.int64)
    cumulative_negative = np.cumsum(descending_negative, dtype=np.int64)
    auprc = float(
        np.sum(
            (descending_positive / n_positive)
            * cumulative_positive
            / (cumulative_positive + cumulative_negative)
        )
    )
    if positive_zero:
        auprc += (positive_zero / n_positive) * (n_positive / total)

    cumulative_positive = positive_zero + np.cumsum(positive_counts, dtype=np.int64)
    cumulative_negative = negative_zero + np.cumsum(negative_counts, dtype=np.int64)
    auprc_negated = 0.0
    if positive_zero:
        auprc_negated += (positive_zero / n_positive) * (positive_zero / zero_count)
    auprc_negated += float(
        np.sum(
            (positive_counts / n_positive)
            * cumulative_positive
            / (cumulative_positive + cumulative_negative)
        )
    )
    assert -1e-12 <= auprc <= 1 + 1e-12
    assert -1e-12 <= auprc_negated <= 1 + 1e-12
    return u_statistic, variance, auprc, auprc_negated


def comparison_metrics(
    matrix: csr_matrix,
    positive_ids: np.ndarray,
    negative_ids: np.ndarray,
    *,
    minimum_nonzero_support: int,
) -> pl.DataFrame:
    positive_ids = np.asarray(positive_ids, dtype=np.int64)
    negative_ids = np.asarray(negative_ids, dtype=np.int64)
    assert matrix.shape[1] == D_SAE and minimum_nonzero_support > 0
    assert positive_ids.ndim == negative_ids.ndim == 1
    assert positive_ids.size > 1 and negative_ids.size > 1
    assert len(set(positive_ids) & set(negative_ids)) == 0
    ids = np.concatenate((positive_ids, negative_ids))
    assert ids.min() >= 0 and ids.max() < matrix.shape[0]
    submatrix = matrix[ids, :].tocsc()
    assert submatrix.has_sorted_indices
    support = np.diff(submatrix.indptr)
    eligible = np.flatnonzero(support >= minimum_nonzero_support)
    assert eligible.size > 0

    n_positive = positive_ids.size
    n_negative = negative_ids.size
    rows: list[dict[str, float | int | str]] = []
    welch_df: list[float] = []
    mann_variance: list[float] = []
    for feature_id in eligible:
        start, end = submatrix.indptr[feature_id : feature_id + 2]
        local_rows = submatrix.indices[start:end]
        values = submatrix.data[start:end].astype(np.float64, copy=False)
        assert values.size == support[feature_id]
        assert np.all(values > 0) and np.isfinite(values).all()
        is_positive = local_rows < n_positive
        positive_values = values[is_positive]
        negative_values = values[~is_positive]
        positive_sum = float(positive_values.sum())
        negative_sum = float(negative_values.sum())
        positive_sum_squares = float(np.square(positive_values).sum())
        negative_sum_squares = float(np.square(negative_values).sum())
        mean_positive = positive_sum / n_positive
        mean_negative = negative_sum / n_negative
        variance_positive = max(
            0.0,
            (positive_sum_squares - positive_sum**2 / n_positive) / (n_positive - 1),
        )
        variance_negative = max(
            0.0,
            (negative_sum_squares - negative_sum**2 / n_negative) / (n_negative - 1),
        )
        mean_difference = mean_positive - mean_negative
        standard_error_squared = (
            variance_positive / n_positive + variance_negative / n_negative
        )
        if standard_error_squared == 0:
            welch_statistic = (
                0.0
                if mean_difference == 0
                else math.copysign(math.inf, mean_difference)
            )
            degrees_freedom = math.inf
        else:
            welch_statistic = mean_difference / math.sqrt(standard_error_squared)
            denominator = (variance_positive / n_positive) ** 2 / (n_positive - 1) + (
                variance_negative / n_negative
            ) ** 2 / (n_negative - 1)
            degrees_freedom = (
                standard_error_squared**2 / denominator if denominator > 0 else math.inf
            )
        pooled_sd = math.sqrt((variance_positive + variance_negative) / 2)
        standardized = (
            mean_difference / pooled_sd
            if pooled_sd > 0
            else (
                0.0
                if mean_difference == 0
                else math.copysign(math.inf, mean_difference)
            )
        )
        u_statistic, u_variance, auprc, auprc_negated = _rank_and_ap_metrics(
            values,
            is_positive,
            n_positive=n_positive,
            n_negative=n_negative,
        )
        rows.append(
            {
                "feature_id": int(feature_id),
                "n": int(n_positive + n_negative),
                "n_positive": int(n_positive),
                "n_negative": int(n_negative),
                "prevalence": n_positive / (n_positive + n_negative),
                "nonzero_support": int(support[feature_id]),
                "positive_nonzero": int(positive_values.size),
                "negative_nonzero": int(negative_values.size),
                "mean_positive": mean_positive,
                "mean_negative": mean_negative,
                "mean_difference": mean_difference,
                "standardized_mean_difference": standardized,
                "welch_statistic": welch_statistic,
                "u_statistic": u_statistic,
                "rank_biserial": 2 * u_statistic / (n_positive * n_negative) - 1,
                "auprc": auprc,
                "auprc_negated": auprc_negated,
                "best_auprc": max(auprc, auprc_negated),
                "best_auprc_direction": (
                    "higher" if auprc >= auprc_negated else "lower"
                ),
            }
        )
        welch_df.append(degrees_freedom)
        mann_variance.append(u_variance)

    result = pl.DataFrame(rows)
    welch_statistics = result["welch_statistic"].to_numpy()
    welch_p = 2 * stats.t.sf(np.abs(welch_statistics), np.asarray(welch_df))
    u_values = result["u_statistic"].to_numpy()
    u_mean = n_positive * n_negative / 2
    u_variance = np.asarray(mann_variance)
    mann_p = np.ones(result.height, dtype=np.float64)
    variable = u_variance > 0
    z = np.zeros(result.height, dtype=np.float64)
    z[variable] = (np.abs(u_values[variable] - u_mean) - 0.5) / np.sqrt(
        u_variance[variable]
    )
    mann_p[variable] = np.minimum(1.0, 2 * stats.norm.sf(z[variable]))
    assert np.isfinite(welch_p).all() and np.isfinite(mann_p).all()
    return result.with_columns(
        pl.Series("welch_p", welch_p),
        pl.Series("mann_whitney_p", mann_p),
    )
