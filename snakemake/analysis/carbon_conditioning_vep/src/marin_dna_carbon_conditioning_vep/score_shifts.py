"""Per-variant prompt-conditioning score-shift summaries."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

ALIGNMENT_COLUMNS = (
    "variant_id",
    "subset",
    "match_group",
    "label",
)


def assemble_score_shifts(
    untagged: pd.DataFrame,
    tagged: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Align conditions by variant and return tagged-minus-untagged score shifts."""
    required = {*ALIGNMENT_COLUMNS, "score"}
    missing = sorted(required - set(untagged.columns))
    assert not missing, f"untagged scores missing columns: {missing}"
    assert untagged["variant_id"].is_unique, "untagged variant IDs must be unique"

    rows: list[pd.DataFrame] = []
    baseline = untagged.loc[:, [*ALIGNMENT_COLUMNS, "score"]].reset_index(drop=True)
    baseline_ids = pd.Index(baseline["variant_id"])
    for condition, frame in tagged.items():
        missing = sorted(required - set(frame.columns))
        assert not missing, f"{condition} scores missing columns: {missing}"
        assert frame["variant_id"].is_unique, f"{condition} variant IDs must be unique"
        candidate = frame.loc[:, [*ALIGNMENT_COLUMNS, "score"]].reset_index(drop=True)
        candidate_ids = pd.Index(candidate["variant_id"])
        missing_ids = baseline_ids.difference(candidate_ids)
        extra_ids = candidate_ids.difference(baseline_ids)
        assert missing_ids.empty and extra_ids.empty, (
            f"{condition} variant IDs differ from untagged: "
            f"{len(missing_ids)} missing, {len(extra_ids)} extra"
        )
        candidate = candidate.set_index("variant_id").loc[baseline_ids].reset_index()
        pd.testing.assert_frame_equal(
            candidate.loc[:, list(ALIGNMENT_COLUMNS)],
            baseline.loc[:, list(ALIGNMENT_COLUMNS)],
            check_exact=True,
            obj=f"{condition} alignment",
        )
        shifts = baseline.loc[:, list(ALIGNMENT_COLUMNS)].copy()
        shifts["condition"] = condition
        shifts["delta_score"] = candidate["score"] - baseline["score"]
        rows.append(shifts)

    result = pd.concat(rows, ignore_index=True)
    assert np.isfinite(result["delta_score"]).all(), "score shifts must be finite"
    return result


def summarize_score_shifts(shifts: pd.DataFrame) -> pd.DataFrame:
    """Summarize score-shift location and spread for each subset and label."""
    required = {"condition", "subset", "label", "delta_score"}
    missing = sorted(required - set(shifts.columns))
    assert not missing, f"score shifts missing columns: {missing}"
    rows: list[dict[str, object]] = []
    for (condition, subset, label), group in shifts.groupby(
        ["condition", "subset", "label"], sort=True
    ):
        values = group["delta_score"].to_numpy(dtype=float)
        rows.append(
            {
                "condition": str(condition),
                "subset": str(subset),
                "label": bool(label),
                "n_variants": len(values),
                "mean_delta": float(values.mean()),
                "median_delta": float(np.median(values)),
                "std_delta": float(values.std(ddof=1)),
                "iqr_delta": float(
                    np.quantile(values, 0.75) - np.quantile(values, 0.25)
                ),
                "median_abs_delta": float(np.median(np.abs(values))),
                "p90_abs_delta": float(np.quantile(np.abs(values), 0.90)),
            }
        )
    return pd.DataFrame(rows)


def _sample_std_from_sums(
    sums: np.ndarray,
    sums_of_squares: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    variance = (sums_of_squares - (sums**2 / counts)) / (counts - 1)
    return np.sqrt(np.maximum(variance, 0.0))


def bootstrap_matched_score_shifts(
    shifts: pd.DataFrame,
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
    min_groups: int,
) -> pd.DataFrame:
    """Estimate label-separation shifts and positive/negative spread ratios."""
    required = {"condition", "subset", "match_group", "label", "delta_score"}
    missing = sorted(required - set(shifts.columns))
    assert not missing, f"score shifts missing columns: {missing}"
    assert n_bootstrap > 0
    rng = np.random.default_rng(bootstrap_seed)
    rows: list[dict[str, object]] = []

    for condition in sorted(shifts["condition"].unique()):
        condition_rows = shifts.loc[shifts["condition"].eq(condition)]
        subsets = ["_all_development_", *sorted(condition_rows["subset"].unique())]
        for subset in subsets:
            frame = (
                condition_rows
                if subset == "_all_development_"
                else condition_rows.loc[condition_rows["subset"].eq(subset)]
            )
            positive_values: list[float] = []
            negative_sums: list[float] = []
            negative_sums_of_squares: list[float] = []
            negative_counts: list[int] = []
            for _group_id, group in frame.groupby("match_group", sort=True):
                positives = group.loc[group["label"], "delta_score"].to_numpy(
                    dtype=float
                )
                negatives = group.loc[~group["label"], "delta_score"].to_numpy(
                    dtype=float
                )
                assert len(positives) == 1, "each match group must have one positive"
                assert len(negatives) >= 1, "each match group must have negatives"
                positive_values.append(float(positives[0]))
                negative_sums.append(float(negatives.sum()))
                negative_sums_of_squares.append(float(np.square(negatives).sum()))
                negative_counts.append(len(negatives))

            positive = np.asarray(positive_values)
            negative_sum = np.asarray(negative_sums)
            negative_sum_of_squares = np.asarray(negative_sums_of_squares)
            negative_count = np.asarray(negative_counts)
            n_groups = len(positive)
            assert n_groups >= 2, "at least two match groups are required"
            indices = rng.integers(0, n_groups, size=(n_bootstrap, n_groups))

            bootstrap_positive_sum = positive[indices].sum(axis=1)
            bootstrap_positive_sum_of_squares = np.square(positive[indices]).sum(axis=1)
            bootstrap_negative_sum = negative_sum[indices].sum(axis=1)
            bootstrap_negative_sum_of_squares = negative_sum_of_squares[indices].sum(
                axis=1
            )
            bootstrap_negative_count = negative_count[indices].sum(axis=1)
            bootstrap_positive_count = np.full(n_bootstrap, n_groups)

            bootstrap_separation = (
                bootstrap_positive_sum / bootstrap_positive_count
                - bootstrap_negative_sum / bootstrap_negative_count
            )
            bootstrap_positive_std = _sample_std_from_sums(
                bootstrap_positive_sum,
                bootstrap_positive_sum_of_squares,
                bootstrap_positive_count,
            )
            bootstrap_negative_std = _sample_std_from_sums(
                bootstrap_negative_sum,
                bootstrap_negative_sum_of_squares,
                bootstrap_negative_count,
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                bootstrap_log2_std_ratio = np.log2(
                    bootstrap_positive_std / bootstrap_negative_std
                )
            finite_std_ratios = bootstrap_log2_std_ratio[
                np.isfinite(bootstrap_log2_std_ratio)
            ]
            assert len(finite_std_ratios) >= n_bootstrap // 2, (
                "too few finite bootstrap spread ratios"
            )

            positive_mean = float(positive.mean())
            negative_total_count = int(negative_count.sum())
            negative_mean = float(negative_sum.sum() / negative_total_count)
            positive_std = float(positive.std(ddof=1))
            negative_std = float(
                _sample_std_from_sums(
                    np.asarray([negative_sum.sum()]),
                    np.asarray([negative_sum_of_squares.sum()]),
                    np.asarray([negative_total_count]),
                )[0]
            )
            rows.append(
                {
                    "condition": str(condition),
                    "subset": str(subset),
                    "n_groups": n_groups,
                    "n_positive": n_groups,
                    "n_negative": negative_total_count,
                    "mean_delta_positive": positive_mean,
                    "mean_delta_negative": negative_mean,
                    "label_separation_shift": positive_mean - negative_mean,
                    "label_separation_ci_low": float(
                        np.quantile(bootstrap_separation, 0.025)
                    ),
                    "label_separation_ci_high": float(
                        np.quantile(bootstrap_separation, 0.975)
                    ),
                    "std_delta_positive": positive_std,
                    "std_delta_negative": negative_std,
                    "std_ratio_positive_negative": positive_std / negative_std,
                    "log2_std_ratio_positive_negative": float(
                        np.log2(positive_std / negative_std)
                    ),
                    "log2_std_ratio_ci_low": float(
                        np.quantile(finite_std_ratios, 0.025)
                    ),
                    "log2_std_ratio_ci_high": float(
                        np.quantile(finite_std_ratios, 0.975)
                    ),
                    "low_sample": n_groups < min_groups,
                }
            )

    result = pd.DataFrame(rows)
    numeric = result.select_dtypes(include=[np.number])
    assert np.isfinite(numeric.to_numpy()).all(), "bootstrap summaries must be finite"
    return result
