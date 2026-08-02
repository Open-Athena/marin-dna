"""Frozen statistical helpers for issue-435 paired repeat-variant analysis."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
from scipy import stats

from association_common import bh_adjust

PAIRED_RUN_ID = "dna-exp435-repeat-variant-deltas-r1"
VARIANT_PANEL_ARCHIVE_SHA256 = (
    "1617638081e099af2e7b370b221c5cb964e8eb85ff8cbbcc2ede8643764bdf60"
)
REFERENCE_ASSOCIATION_ARCHIVE_SHA256 = (
    "cc72fbb0033290af54d2c6dcb0a7521e9b23f5f84b6906bfd9fee1f69206ece0"
)
PAIRED_ACTIVATION_MANIFEST_SHA256 = (
    "0b2e77abf4967a6c9bf7e07bbebe11b0585e21fc4290fcb571cbb16117da10c5"
)

ARMS = ("block01-25m", "block10-25m", "block19-25m")
BLOCK_BY_ARM = {"block01-25m": 1, "block10-25m": 10, "block19-25m": 19}
ORIENTATIONS = ("forward", "reverse_complement")
RESPONSES = ("abs_delta", "delta")
HIERARCHIES = ("class", "family", "subfamily")
SUBSET_TARGETS = (
    "3_prime_UTR_variant",
    "5_prime_UTR_variant",
    "distal",
    "missense_variant",
    "non_coding_transcript_exon_variant",
    "splicing",
    "tss_proximal",
)

MINIMUM_GLOBAL_SUPPORT = 64
MINIMUM_STRATIFIED_SUPPORT = 16
FDR_THRESHOLD = 0.05


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision with stable, grouped handling of score ties."""

    values = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(labels, dtype=bool)
    assert values.ndim == positive.ndim == 1 and values.size == positive.size > 0
    assert np.isfinite(values).all()
    n_positive = int(positive.sum())
    assert 0 < n_positive < positive.size
    order = np.argsort(-values, kind="stable")
    ordered_values = values[order]
    ordered_positive = positive[order]
    ends = np.concatenate(
        (np.flatnonzero(ordered_values[1:] != ordered_values[:-1]), [values.size - 1])
    )
    cumulative = np.cumsum(ordered_positive, dtype=np.int64)
    cumulative_at_end = cumulative[ends]
    positives_in_group = np.diff(np.concatenate(([0], cumulative_at_end)))
    precision = cumulative_at_end / (ends + 1)
    result = float(np.sum((positives_in_group / n_positive) * precision))
    assert 0 <= result <= 1
    return result


def binary_feature_metrics(
    matrix: np.ndarray,
    feature_ids: np.ndarray,
    positive_rows: np.ndarray,
    negative_rows: np.ndarray,
    *,
    minimum_nonzero_support: int,
) -> pl.DataFrame:
    """Test dense signed or unsigned feature responses for one binary contrast."""

    values = np.asarray(matrix, dtype=np.float32)
    features = np.asarray(feature_ids, dtype=np.int64)
    positive_rows = np.asarray(positive_rows, dtype=np.int64)
    negative_rows = np.asarray(negative_rows, dtype=np.int64)
    assert values.ndim == 2 and values.shape[1] == features.size > 0
    assert np.isfinite(values).all() and minimum_nonzero_support > 0
    assert positive_rows.ndim == negative_rows.ndim == 1
    assert positive_rows.size > 1 and negative_rows.size > 1
    assert len(set(positive_rows) & set(negative_rows)) == 0
    assert min(positive_rows.min(), negative_rows.min()) >= 0
    assert max(positive_rows.max(), negative_rows.max()) < values.shape[0]

    selected_rows = np.concatenate((positive_rows, negative_rows))
    support = np.count_nonzero(values[selected_rows, :], axis=0)
    eligible_columns = np.flatnonzero(support >= minimum_nonzero_support)
    rows: list[dict[str, float | int | str]] = []
    labels = np.concatenate(
        (
            np.ones(positive_rows.size, dtype=bool),
            np.zeros(negative_rows.size, dtype=bool),
        )
    )
    for column in eligible_columns:
        positive = values[positive_rows, column].astype(np.float64, copy=False)
        negative = values[negative_rows, column].astype(np.float64, copy=False)
        combined = np.concatenate((positive, negative))
        mean_positive = float(positive.mean())
        mean_negative = float(negative.mean())
        variance_positive = float(positive.var(ddof=1))
        variance_negative = float(negative.var(ddof=1))
        mean_difference = mean_positive - mean_negative
        standard_error_squared = (
            variance_positive / positive.size + variance_negative / negative.size
        )
        if standard_error_squared == 0:
            welch_statistic = (
                0.0
                if mean_difference == 0
                else math.copysign(math.inf, mean_difference)
            )
            welch_p = 1.0 if mean_difference == 0 else 0.0
        else:
            welch_statistic = mean_difference / math.sqrt(standard_error_squared)
            denominator = (variance_positive / positive.size) ** 2 / (
                positive.size - 1
            ) + (variance_negative / negative.size) ** 2 / (negative.size - 1)
            degrees_freedom = (
                standard_error_squared**2 / denominator if denominator > 0 else math.inf
            )
            welch_p = float(2 * stats.t.sf(abs(welch_statistic), degrees_freedom))
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
        if np.all(combined == combined[0]):
            u_statistic = positive.size * negative.size / 2
            mann_p = 1.0
        else:
            mann = stats.mannwhitneyu(
                positive,
                negative,
                alternative="two-sided",
                method="asymptotic",
            )
            u_statistic = float(mann.statistic)
            mann_p = float(mann.pvalue)
        rank_biserial = 2 * u_statistic / (positive.size * negative.size) - 1
        auprc = average_precision(combined, labels)
        auprc_negated = average_precision(-combined, labels)
        rows.append(
            {
                "feature_id": int(features[column]),
                "n": int(combined.size),
                "n_positive": int(positive.size),
                "n_negative": int(negative.size),
                "prevalence": positive.size / combined.size,
                "nonzero_support": int(support[column]),
                "positive_nonzero": int(np.count_nonzero(positive)),
                "negative_nonzero": int(np.count_nonzero(negative)),
                "mean_positive": mean_positive,
                "mean_negative": mean_negative,
                "mean_difference": mean_difference,
                "standardized_mean_difference": standardized,
                "welch_statistic": welch_statistic,
                "u_statistic": u_statistic,
                "rank_biserial": rank_biserial,
                "auprc": auprc,
                "auprc_negated": auprc_negated,
                "best_auprc": max(auprc, auprc_negated),
                "best_auprc_direction": (
                    "higher" if auprc >= auprc_negated else "lower"
                ),
                "welch_p": welch_p,
                "mann_whitney_p": mann_p,
            }
        )
    if not rows:
        return pl.DataFrame()
    result = pl.DataFrame(rows)
    assert result.filter(
        ~pl.col("welch_p").is_between(0, 1, closed="both")
        | ~pl.col("mann_whitney_p").is_between(0, 1, closed="both")
    ).is_empty()
    return result


def with_fdr(frame: pl.DataFrame, *, response: str) -> pl.DataFrame:
    """Apply the frozen complete-family corrections and call concordant effects."""

    assert response in RESPONSES and frame.height > 0
    welch_q = bh_adjust(frame["welch_p"].to_numpy())
    mann_q = bh_adjust(frame["mann_whitney_p"].to_numpy())
    same_positive_direction = (pl.col("mean_difference") > 0) & (
        pl.col("rank_biserial") > 0
    )
    same_negative_direction = (pl.col("mean_difference") < 0) & (
        pl.col("rank_biserial") < 0
    )
    significant = (pl.col("welch_q") <= FDR_THRESHOLD) & (
        pl.col("mann_whitney_q") <= FDR_THRESHOLD
    )
    return frame.with_columns(
        pl.Series("welch_q", welch_q),
        pl.Series("mann_whitney_q", mann_q),
    ).with_columns(
        pl.max_horizontal("welch_q", "mann_whitney_q").alias("maximum_q"),
        (significant & (same_positive_direction | same_negative_direction)).alias(
            "concordant_association"
        ),
        (pl.lit(response == "abs_delta") & significant & same_positive_direction).alias(
            "positive_mutation_association"
        ),
        pl.when(same_positive_direction)
        .then(pl.lit("higher"))
        .when(same_negative_direction)
        .then(pl.lit("lower"))
        .otherwise(pl.lit("discordant"))
        .alias("effect_direction"),
    )
