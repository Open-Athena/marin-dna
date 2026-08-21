"""Add Group SMD columns to the matched-pair AUPRC table."""

import numpy as np
import pandas as pd

from marin_dna_evals.metrics import (
    GLOBAL_SUBSET,
    MACRO_AVG_SUBSET,
    compute_auprc_metrics,
)


GROUP_SMD_COLUMNS = [
    "group_smd_value",
    "group_smd_se",
    "group_smd_ci_low",
    "group_smd_ci_high",
    "group_smd_confidence_level",
    "group_smd_available",
    "group_smd_unavailable_reason",
    "group_smd_uncertainty_method",
    "group_smd_n_bootstrap",
    "group_smd_n_bootstrap_valid",
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
    """Return the mean matched-group gap divided by its sample standard deviation."""
    gaps = matched_group_gaps(label, score, match_group)
    if len(gaps) < 2:
        raise ValueError("Group SMD requires at least two match groups")
    sd = float(gaps.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("Group SMD requires non-zero finite SD across group gaps")
    return float(gaps.mean() / sd)


def _joint_group_smd_bootstrap(
    frame: pd.DataFrame,
    score_columns: list[str],
    *,
    n_bootstrap: int,
    generator: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Use one match-group draw matrix for every score column in one scope."""
    if n_bootstrap == 0:
        return {column: np.empty(0, dtype=float) for column in score_columns}

    group_ids = list(frame.groupby("match_group", sort=True).indices)
    n_groups = len(group_ids)
    sampled_groups = generator.integers(0, n_groups, size=(n_bootstrap, n_groups))
    multiplicity = np.zeros((n_bootstrap, n_groups), dtype=np.int32)
    np.add.at(
        multiplicity,
        (
            np.repeat(np.arange(n_bootstrap), n_groups),
            sampled_groups.ravel(),
        ),
        1,
    )

    samples: dict[str, np.ndarray] = {}
    for score_column in score_columns:
        gaps = matched_group_gaps(
            frame["label"], frame[score_column], frame["match_group"]
        ).reindex(group_ids)
        if gaps.isna().any():
            raise ValueError("match-group gaps do not align with bootstrap groups")
        if n_groups < 2:
            samples[score_column] = np.full(n_bootstrap, np.nan)
            continue
        values = gaps.to_numpy(dtype=float)
        gap_mean = multiplicity @ values / n_groups
        gap_squares = multiplicity @ np.square(values)
        gap_variance = np.maximum(
            (gap_squares - n_groups * np.square(gap_mean)) / (n_groups - 1),
            0.0,
        )
        gap_sd = np.sqrt(gap_variance)
        samples[score_column] = np.divide(
            gap_mean,
            gap_sd,
            out=np.full(n_bootstrap, np.nan),
            where=gap_sd > 0,
        )
    return samples


def _group_smd_summary(
    frame: pd.DataFrame,
    score_column: str,
    samples: np.ndarray,
) -> dict[str, float | int | bool | str | None]:
    """Summarize Group SMD for one direct subset or global scope."""
    gaps = matched_group_gaps(frame["label"], frame[score_column], frame["match_group"])
    if len(gaps) < 2:
        return {
            "value": float("nan"),
            "se": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
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
            "available": False,
            "unavailable_reason": "zero_or_non_finite_group_gap_sd",
            "n_bootstrap_valid": 0,
        }

    finite_samples = samples[np.isfinite(samples)]
    n_valid = len(finite_samples)
    se = float(np.std(finite_samples, ddof=1)) if n_valid >= 2 else float("nan")
    if n_valid:
        ci_low, ci_high = (
            float(value) for value in np.percentile(finite_samples, [2.5, 97.5])
        )
    else:
        ci_low = ci_high = float("nan")
    return {
        "value": float(gaps.mean() / gap_sd),
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
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
    rng: np.random.Generator | int | None = 0,
    n_min: int = 30,
) -> pd.DataFrame:
    """Return the existing AUPRC rows with additive Group SMD columns.

    Direct subset and ``_global_`` rows receive Group SMD estimates and
    match-group bootstrap uncertainty.
    The same bootstrap group draws are reused for every score column within a
    scope and are discarded after the summary columns are computed.
    ``_macro_avg_`` rows mark Group SMD unavailable because averaging
    subset-standardized effects defines a different statistic.
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

    groups_per_subset = base.groupby("subset", sort=False)["match_group"].nunique()
    if not groups_per_subset.ge(n_min).any():
        raise ValueError(
            f"no subsets meet n_min={n_min} for the AUPRC macro average; "
            f"per-subset match-group counts: {groups_per_subset.to_dict()}"
        )

    result = compute_auprc_metrics(
        dataset=base[["label", "subset", "match_group"]],
        scores=base[columns],
        score_columns=columns,
        n_bootstrap=n_bootstrap,
        rng=rng,
        n_min=n_min,
    )

    result["group_smd_value"] = np.nan
    result["group_smd_se"] = np.nan
    result["group_smd_ci_low"] = np.nan
    result["group_smd_ci_high"] = np.nan
    result["group_smd_confidence_level"] = np.nan
    result["group_smd_available"] = False
    result["group_smd_unavailable_reason"] = pd.Series(
        [None] * len(result), dtype="object"
    )
    result["group_smd_uncertainty_method"] = pd.Series(
        [None] * len(result), dtype="object"
    )
    result["group_smd_n_bootstrap"] = n_bootstrap
    result["group_smd_n_bootstrap_valid"] = 0

    generator = np.random.default_rng(rng)
    scope_frames: dict[str, pd.DataFrame] = {
        str(subset): frame.reset_index(drop=True)
        for subset, frame in base.groupby("subset", sort=False)
    }
    scope_frames[GLOBAL_SUBSET] = base
    summaries: dict[tuple[str, str], dict[str, object]] = {}
    for subset, frame in scope_frames.items():
        samples = _joint_group_smd_bootstrap(
            frame,
            columns,
            n_bootstrap=n_bootstrap,
            generator=generator,
        )
        for score_column in columns:
            summaries[(score_column, subset)] = _group_smd_summary(
                frame, score_column, samples[score_column]
            )

    for index, row in result.iterrows():
        score_column = str(row["score_type"])
        subset = str(row["subset"])
        if subset == MACRO_AVG_SUBSET:
            result.at[index, "group_smd_unavailable_reason"] = (
                "group_smd_not_defined_for_macro_average"
            )
            continue

        summary = summaries[(score_column, subset)]
        for field in (
            "value",
            "se",
            "ci_low",
            "ci_high",
            "available",
            "unavailable_reason",
            "n_bootstrap_valid",
        ):
            result.at[index, f"group_smd_{field}"] = summary[field]
        result.at[index, "group_smd_confidence_level"] = 0.95
        result.at[index, "group_smd_uncertainty_method"] = (
            "joint_match_group_bootstrap_percentile"
        )

    return result
