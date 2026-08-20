"""Maintained AUPRC and Group SMD reporting for grouped VEP datasets.

Group SMD standardizes one positive-minus-mean-negative gap per matched group.
The grouped report keeps the existing AUPRC point estimates and bootstrap
standard errors, then adds Group SMD with a joint match-group bootstrap.
"""

import numpy as np
import pandas as pd

from marin_dna_evals.metrics import (
    GLOBAL_SUBSET,
    MACRO_AVG_SUBSET,
    compute_auprc_metrics,
)

AUPRC = "AUPRC"
GROUP_SMD = "group_smd"
HIGHER_IS_BETTER = {AUPRC: True, GROUP_SMD: True}

SUMMARY_COLUMNS = [
    "metric",
    "higher_is_better",
    "score_type",
    "subset",
    "value",
    "se",
    "ci_low",
    "ci_high",
    "confidence_level",
    "n_groups",
    "n_rows",
    "available",
    "unavailable_reason",
    "uncertainty_method",
    "n_bootstrap",
    "n_bootstrap_valid",
]
BOOTSTRAP_COLUMNS = [
    "draw",
    "metric",
    "score_type",
    "subset",
    "value",
    "n_groups",
]


def _validated_group_frame(
    label: pd.Series | np.ndarray,
    score: pd.Series | np.ndarray,
    match_group: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Return finite scores with a validated one-positive matched-group contract."""
    if len(label) != len(score) or len(label) != len(match_group):
        raise ValueError(
            "length mismatch: "
            f"label={len(label)} score={len(score)} match_group={len(match_group)}"
        )
    if len(label) == 0:
        raise ValueError("grouped VEP metrics require at least one row")

    label_series = pd.Series(label).reset_index(drop=True)
    group_series = pd.Series(match_group).reset_index(drop=True)
    if label_series.isna().any():
        raise ValueError("label contains null values")
    if not label_series.isin((0, 1, False, True)).all():
        values = label_series.loc[~label_series.isin((0, 1, False, True))].unique()
        raise ValueError(f"label must be binary 0/1; got {values[:5].tolist()}")
    if group_series.isna().any():
        raise ValueError("match_group contains null values")

    try:
        score_values = np.asarray(score, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("score must be numeric") from error
    if not np.isfinite(score_values).all():
        raise ValueError("score contains non-finite values")

    frame = pd.DataFrame(
        {
            "label": label_series.astype(int),
            "score": score_values,
            "match_group": group_series,
        }
    )
    group_counts = frame.groupby("match_group", sort=False)["label"].agg(
        n_rows="size", n_pos="sum"
    )
    bad = group_counts[(group_counts["n_pos"] != 1) | (group_counts["n_rows"] < 2)]
    if not bad.empty:
        raise ValueError(
            "each match_group must contain exactly one positive and at least one "
            f"negative; got {len(bad)} incompatible groups; first: "
            f"{bad.head().to_dict('index')}"
        )
    return frame


def matched_group_gaps(
    label: pd.Series | np.ndarray,
    score: pd.Series | np.ndarray,
    match_group: pd.Series | np.ndarray,
) -> pd.Series:
    """Return one positive-minus-mean-negative score gap per match group."""
    frame = _validated_group_frame(label, score, match_group)
    positives = frame.loc[frame["label"] == 1].set_index("match_group")["score"]
    negative_means = (
        frame.loc[frame["label"] == 0]
        .groupby("match_group", sort=False)["score"]
        .mean()
    )
    gaps = positives.to_frame("positive").join(
        negative_means.rename("negative_mean"), validate="one_to_one"
    )
    if gaps.isna().any(axis=None):
        raise ValueError("positive and negative match-group sets differ")
    return (gaps["positive"] - gaps["negative_mean"]).rename("group_gap")


def group_smd(
    label: pd.Series | np.ndarray,
    score: pd.Series | np.ndarray,
    match_group: pd.Series | np.ndarray,
) -> float:
    """Mean matched-group gap divided by the sample SD of group gaps."""
    gaps = matched_group_gaps(label, score, match_group)
    if len(gaps) < 2:
        raise ValueError("Group SMD requires at least two match groups")
    sd = float(gaps.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("Group SMD requires non-zero finite SD across group gaps")
    return float(gaps.mean() / sd)


def _weighted_bootstrap_average_precision(
    label: np.ndarray,
    score: np.ndarray,
    row_group_index: np.ndarray,
    group_multiplicity: np.ndarray,
    *,
    batch_size: int = 100,
) -> np.ndarray:
    """Compute sklearn-compatible AP for cluster-weight bootstrap draws."""
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


def _joint_scope_bootstrap(
    frame: pd.DataFrame,
    score_columns: list[str],
    *,
    subset: str,
    n_bootstrap: int,
    generator: np.random.Generator,
) -> pd.DataFrame:
    """Bootstrap all scores and both metrics with one match-group draw matrix."""
    if n_bootstrap == 0:
        return pd.DataFrame(columns=BOOTSTRAP_COLUMNS)

    labels = frame["label"].to_numpy(dtype=int)
    groups = frame["match_group"].to_numpy()
    group_indices = pd.Series(groups).groupby(groups, sort=True).indices
    group_ids = list(group_indices)
    group_to_rows = list(group_indices.values())
    n_groups = len(group_to_rows)
    row_group_index = np.empty(len(frame), dtype=int)
    for group_index, rows in enumerate(group_to_rows):
        row_group_index[rows] = group_index

    sampled_groups = generator.integers(0, n_groups, size=(n_bootstrap, n_groups))
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
    for score_column in score_columns:
        scores = frame[score_column].to_numpy(dtype=float)
        auprc_samples = _weighted_bootstrap_average_precision(
            labels,
            scores,
            row_group_index,
            group_multiplicity,
        )
        gaps = matched_group_gaps(labels, scores, groups).reindex(group_ids).to_numpy()
        if n_groups < 2:
            smd_samples = np.full(n_bootstrap, np.nan)
        else:
            gap_mean = group_multiplicity @ gaps / n_groups
            gap_squares = group_multiplicity @ np.square(gaps)
            gap_variance = np.maximum(
                (gap_squares - n_groups * np.square(gap_mean)) / (n_groups - 1),
                0.0,
            )
            gap_sd = np.sqrt(gap_variance)
            smd_samples = np.divide(
                gap_mean,
                gap_sd,
                out=np.full(n_bootstrap, np.nan),
                where=gap_sd > 0,
            )

        for metric, values in ((AUPRC, auprc_samples), (GROUP_SMD, smd_samples)):
            rows.extend(
                {
                    "draw": draw,
                    "metric": metric,
                    "score_type": score_column,
                    "subset": subset,
                    "value": float(values[draw]),
                    "n_groups": n_groups,
                }
                for draw in range(n_bootstrap)
            )
    return pd.DataFrame(rows, columns=BOOTSTRAP_COLUMNS)


def _group_smd_summary(
    frame: pd.DataFrame,
    score_column: str,
    samples: pd.DataFrame,
) -> dict[str, float | int | bool | str | None]:
    """Summarize one direct Group SMD scope, including explicit unavailability."""
    gaps = matched_group_gaps(frame["label"], frame[score_column], frame["match_group"])
    n_groups = len(gaps)
    if n_groups < 2:
        return {
            "value": float("nan"),
            "se": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_groups": n_groups,
            "n_rows": len(frame),
            "available": False,
            "unavailable_reason": "requires_at_least_two_match_groups",
            "n_bootstrap_valid": 0,
        }

    gap_sd = float(gaps.std(ddof=1))
    if not np.isfinite(gap_sd) or gap_sd <= 0:
        return {
            "value": float("nan"),
            "se": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_groups": n_groups,
            "n_rows": len(frame),
            "available": False,
            "unavailable_reason": "zero_or_non_finite_group_gap_sd",
            "n_bootstrap_valid": 0,
        }

    bootstrap_values = samples.loc[
        (samples["metric"] == GROUP_SMD) & (samples["score_type"] == score_column),
        "value",
    ].to_numpy(dtype=float)
    finite_bootstrap = bootstrap_values[np.isfinite(bootstrap_values)]
    n_valid = len(finite_bootstrap)
    se = float(np.std(finite_bootstrap, ddof=1)) if n_valid >= 2 else float("nan")
    if n_valid:
        ci_low, ci_high = (
            float(value) for value in np.percentile(finite_bootstrap, [2.5, 97.5])
        )
    else:
        ci_low = ci_high = float("nan")
    return {
        "value": float(gaps.mean() / gap_sd),
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_groups": n_groups,
        "n_rows": len(frame),
        "available": True,
        "unavailable_reason": None,
        "n_bootstrap_valid": n_valid,
    }


def compute_grouped_vep_metrics(
    dataset: pd.DataFrame,
    scores: pd.DataFrame,
    score_columns: list[str] | None = None,
    *,
    n_bootstrap: int = 1000,
    rng: int | None = 0,
    n_min: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Report unchanged AUPRC rows beside Group SMD and aligned bootstrap draws.

    ``dataset`` must contain ``label``, ``subset``, and ``match_group``.
    Every match group must contain exactly one positive and at least one
    negative, and a group may belong to only one subset. Missing or incompatible
    grouping raises ``ValueError``. Direct scopes with one group, or with zero
    variance across group gaps, retain an explicit unavailable Group SMD row.

    The bootstrap resamples match groups once per scope and applies each draw to
    all score columns and both metrics. Integer seeds therefore align ``draw``
    across model outputs built from the same dataset revision and row order.
    Group SMD uses the 2.5th and 97.5th bootstrap percentiles for its interval.
    The returned AUPRC ``value``, ``se``, ``n_groups``, and ``n_rows`` are copied
    from ``compute_auprc_metrics`` without recomputation.
    """
    required = {"label", "subset", "match_group"}
    missing = required - set(dataset.columns)
    if missing:
        raise ValueError(
            f"grouped VEP dataset missing required columns: {sorted(missing)}"
        )
    if len(dataset) != len(scores):
        raise ValueError(
            f"dataset/scores length mismatch: {len(dataset)} != {len(scores)}"
        )
    if n_bootstrap < 0:
        raise ValueError(f"n_bootstrap must be non-negative, got {n_bootstrap}")
    if n_min < 1:
        raise ValueError(f"n_min must be positive, got {n_min}")

    columns = list(scores.columns) if score_columns is None else list(score_columns)
    if not columns:
        raise ValueError("at least one score column is required")
    if len(columns) != len(set(columns)):
        raise ValueError(f"score_columns contains duplicates: {columns}")
    missing_scores = set(columns) - set(scores.columns)
    if missing_scores:
        raise ValueError(f"scores missing columns: {sorted(missing_scores)}")
    collisions = required.intersection(columns)
    if collisions:
        raise ValueError(
            f"score columns collide with grouped VEP columns: {sorted(collisions)}"
        )

    base = dataset[["label", "subset", "match_group"]].reset_index(drop=True).copy()
    if base["subset"].isna().any():
        raise ValueError("subset contains null values")
    reserved_subsets = {GLOBAL_SUBSET, MACRO_AVG_SUBSET}.intersection(
        set(base["subset"].astype(str))
    )
    if reserved_subsets:
        raise ValueError(
            f"subset uses reserved aggregate names: {sorted(reserved_subsets)}"
        )

    for score_column in columns:
        validated = _validated_group_frame(
            base["label"], scores[score_column], base["match_group"]
        )
        base["label"] = validated["label"]
        base[score_column] = validated["score"]

    subset_per_group = base.groupby("match_group", sort=False)["subset"].nunique()
    bad_groups = subset_per_group[subset_per_group > 1]
    if not bad_groups.empty:
        raise ValueError(
            f"{len(bad_groups)} match_group(s) span multiple subsets; first: "
            f"{bad_groups.head().to_dict()}"
        )

    legacy_auprc = compute_auprc_metrics(
        dataset=base[["label", "subset", "match_group"]],
        scores=base[columns],
        score_columns=columns,
        n_bootstrap=n_bootstrap,
        rng=rng,
        n_min=n_min,
    )

    generator = np.random.default_rng(rng)
    scope_frames: dict[str, pd.DataFrame] = {
        str(subset): frame.reset_index(drop=True)
        for subset, frame in base.groupby("subset", sort=False)
    }
    scope_frames[GLOBAL_SUBSET] = base
    sample_tables = [
        _joint_scope_bootstrap(
            frame,
            columns,
            subset=subset,
            n_bootstrap=n_bootstrap,
            generator=generator,
        )
        for subset, frame in scope_frames.items()
    ]
    bootstrap_samples = (
        pd.concat(sample_tables, ignore_index=True)
        if sample_tables
        else pd.DataFrame(columns=BOOTSTRAP_COLUMNS)
    )

    smd_summaries: dict[tuple[str, str], dict[str, object]] = {}
    for subset, frame in scope_frames.items():
        subset_samples = bootstrap_samples.loc[bootstrap_samples["subset"] == subset]
        for score_column in columns:
            smd_summaries[(score_column, subset)] = _group_smd_summary(
                frame, score_column, subset_samples
            )

    summary_rows: list[dict[str, object]] = []
    for auprc_row in legacy_auprc.to_dict("records"):
        score_column = str(auprc_row["score_type"])
        subset = str(auprc_row["subset"])
        summary_rows.append(
            {
                "metric": AUPRC,
                "higher_is_better": HIGHER_IS_BETTER[AUPRC],
                **auprc_row,
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "confidence_level": float("nan"),
                "available": True,
                "unavailable_reason": None,
                "uncertainty_method": "match_group_bootstrap_standard_error",
                "n_bootstrap": n_bootstrap,
                "n_bootstrap_valid": max(n_bootstrap, 0),
            }
        )
        if subset == MACRO_AVG_SUBSET:
            smd = {
                "value": float("nan"),
                "se": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "n_groups": 0,
                "n_rows": 0,
                "available": False,
                "unavailable_reason": "group_smd_not_defined_for_macro_average",
                "n_bootstrap_valid": 0,
            }
            uncertainty_method = None
            confidence_level = float("nan")
        else:
            smd = smd_summaries[(score_column, subset)]
            uncertainty_method = "joint_match_group_bootstrap_percentile"
            confidence_level = 0.95
        summary_rows.append(
            {
                "metric": GROUP_SMD,
                "higher_is_better": HIGHER_IS_BETTER[GROUP_SMD],
                "score_type": score_column,
                "subset": subset,
                **smd,
                "confidence_level": confidence_level,
                "uncertainty_method": uncertainty_method,
                "n_bootstrap": n_bootstrap,
            }
        )

    return (
        pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS),
        bootstrap_samples,
    )
