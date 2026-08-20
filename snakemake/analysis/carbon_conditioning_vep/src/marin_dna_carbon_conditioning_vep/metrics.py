"""AUPRC aggregation and paired match-group bootstrap for issue #486.

The contracts are copied and adapted from evals_v2 metrics.py at MarinDNA
commit d40a56ac83ac414bc5c31625bc3996007edbd407.
This package intentionally has no runtime import from marin_dna_evals.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

MACRO_SUBSET = "_macro_avg_"
SCORE_METADATA_COLUMNS = ("variant_id", "label", "subset", "match_group")


def _validate_metric_frame(frame: pd.DataFrame, score_column: str) -> None:
    missing = sorted(set(SCORE_METADATA_COLUMNS + (score_column,)) - set(frame.columns))
    assert not missing, f"score frame missing metric columns: {missing}"
    assert not frame["variant_id"].duplicated().any(), "variant_id must be unique"
    scores = frame[score_column].to_numpy(dtype=float)
    assert np.isfinite(scores).all(), f"{score_column} contains non-finite values"
    labels = frame["label"].astype(int).to_numpy()
    assert set(np.unique(labels)) == {0, 1}, "metric frame requires both binary labels"
    subsets_per_group = frame.groupby("match_group")["subset"].nunique()
    assert subsets_per_group.eq(1).all(), "match_group spans consequence subsets"


def _group_rows(match_group: pd.Series) -> list[np.ndarray]:
    values = np.asarray(match_group)
    return [
        np.asarray(indices, dtype=int)
        for indices in pd.Series(values).groupby(values).indices.values()
    ]


def _bootstrap_auprc(
    label: pd.Series,
    score: pd.Series,
    match_group: pd.Series,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray]:
    y = np.asarray(label, dtype=int)
    values = np.asarray(score, dtype=float)
    groups = _group_rows(match_group)
    assert groups
    point = float(average_precision_score(y, values))
    bootstrap = np.empty(n_bootstrap, dtype=float)
    for draw in range(n_bootstrap):
        sampled = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in sampled])
        sampled_y = y[indices]
        assert 0 < int(sampled_y.sum()) < len(sampled_y), (
            "match-group bootstrap produced one class; the 1:9 group contract is broken"
        )
        bootstrap[draw] = average_precision_score(sampled_y, values[indices])
    return point, bootstrap


def _distribution_summary(distribution: np.ndarray) -> dict[str, float]:
    low, high = np.percentile(distribution, [2.5, 97.5])
    return {
        "se": float(np.std(distribution, ddof=1)),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def compute_absolute_auprc(
    scores: pd.DataFrame,
    *,
    condition: str,
    score_column: str = "score",
    n_bootstrap: int = 1000,
    bootstrap_seed: int = 0,
    min_groups_for_macro: int = 30,
) -> pd.DataFrame:
    """Compute every observed subset plus the qualifying-subset macro average."""
    _validate_metric_frame(scores, score_column)
    rng = np.random.default_rng(bootstrap_seed)
    rows: list[dict[str, Any]] = []
    qualifying_points: list[float] = []
    qualifying_bootstrap: list[np.ndarray] = []
    qualifying_group_count = 0
    qualifying_row_count = 0
    for subset, subset_frame in scores.groupby("subset", sort=True):
        point, bootstrap = _bootstrap_auprc(
            subset_frame["label"],
            subset_frame[score_column],
            subset_frame["match_group"],
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
        n_groups = int(subset_frame["match_group"].nunique())
        eligible = n_groups >= min_groups_for_macro
        row: dict[str, Any] = {
            "condition": condition,
            "score_type": "minus_llr_avg",
            "subset": str(subset),
            "auprc": point,
            **_distribution_summary(bootstrap),
            "n_groups": n_groups,
            "n_rows": len(subset_frame),
            "low_sample": not eligible,
            "macro_eligible": eligible,
            "n_subsets": 1,
        }
        rows.append(row)
        if eligible:
            qualifying_points.append(point)
            qualifying_bootstrap.append(bootstrap)
            qualifying_group_count += n_groups
            qualifying_row_count += len(subset_frame)
    assert qualifying_points, (
        f"no subsets meet min_groups_for_macro={min_groups_for_macro}"
    )
    macro_bootstrap = np.stack(qualifying_bootstrap).mean(axis=0)
    rows.append(
        {
            "condition": condition,
            "score_type": "minus_llr_avg",
            "subset": MACRO_SUBSET,
            "auprc": float(np.mean(qualifying_points)),
            **_distribution_summary(macro_bootstrap),
            "n_groups": qualifying_group_count,
            "n_rows": qualifying_row_count,
            "low_sample": False,
            "macro_eligible": True,
            "n_subsets": len(qualifying_points),
        }
    )
    return pd.DataFrame(rows)


def _align_score_tables(
    score_a: pd.DataFrame,
    score_b: pd.DataFrame,
    *,
    score_column: str,
) -> pd.DataFrame:
    _validate_metric_frame(score_a, score_column)
    _validate_metric_frame(score_b, score_column)
    left = score_a.sort_values("variant_id").reset_index(drop=True)
    right = score_b.sort_values("variant_id").reset_index(drop=True)
    assert left["variant_id"].equals(right["variant_id"]), (
        "paired prompt conditions do not contain identical variant rows"
    )
    for column in ("label", "subset", "match_group"):
        assert left[column].equals(right[column]), (
            f"paired metadata column {column!r} differs between conditions"
        )
    return pd.DataFrame(
        {
            "variant_id": left["variant_id"],
            "label": left["label"].astype(int),
            "subset": left["subset"].astype(str),
            "match_group": left["match_group"],
            "score_a": left[score_column].astype(float),
            "score_b": right[score_column].astype(float),
        }
    )


def _paired_delta_bootstrap(
    frame: pd.DataFrame,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray]:
    y = frame["label"].to_numpy(dtype=int)
    score_a = frame["score_a"].to_numpy(dtype=float)
    score_b = frame["score_b"].to_numpy(dtype=float)
    groups = _group_rows(frame["match_group"])
    point = float(
        average_precision_score(y, score_a) - average_precision_score(y, score_b)
    )
    bootstrap = np.empty(n_bootstrap, dtype=float)
    for draw in range(n_bootstrap):
        sampled = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in sampled])
        sampled_y = y[indices]
        assert 0 < int(sampled_y.sum()) < len(sampled_y)
        bootstrap[draw] = average_precision_score(
            sampled_y, score_a[indices]
        ) - average_precision_score(sampled_y, score_b[indices])
    return point, bootstrap


def compute_paired_auprc_deltas(
    score_a: pd.DataFrame,
    score_b: pd.DataFrame,
    *,
    comparison: str,
    condition_a: str,
    condition_b: str,
    score_column: str = "score",
    n_bootstrap: int = 1000,
    bootstrap_seed: int = 0,
    min_groups_for_macro: int = 30,
) -> pd.DataFrame:
    """Compute AUPRC(a)-AUPRC(b) using one shared group resample per cell."""
    aligned = _align_score_tables(score_a, score_b, score_column=score_column)
    rng = np.random.default_rng(bootstrap_seed)
    rows: list[dict[str, Any]] = []
    macro_points: list[float] = []
    macro_bootstraps: list[np.ndarray] = []
    macro_groups = 0
    macro_rows = 0
    for subset, subset_frame in aligned.groupby("subset", sort=True):
        point, bootstrap = _paired_delta_bootstrap(
            subset_frame,
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
        n_groups = int(subset_frame["match_group"].nunique())
        eligible = n_groups >= min_groups_for_macro
        rows.append(
            {
                "comparison": comparison,
                "condition_a": condition_a,
                "condition_b": condition_b,
                "score_type": "minus_llr_avg",
                "subset": str(subset),
                "delta": point,
                **_distribution_summary(bootstrap),
                "n_groups": n_groups,
                "n_rows": len(subset_frame),
                "low_sample": not eligible,
                "macro_eligible": eligible,
                "n_subsets": 1,
            }
        )
        if eligible:
            macro_points.append(point)
            macro_bootstraps.append(bootstrap)
            macro_groups += n_groups
            macro_rows += len(subset_frame)
    assert macro_points
    macro_distribution = np.stack(macro_bootstraps).mean(axis=0)
    rows.append(
        {
            "comparison": comparison,
            "condition_a": condition_a,
            "condition_b": condition_b,
            "score_type": "minus_llr_avg",
            "subset": MACRO_SUBSET,
            "delta": float(np.mean(macro_points)),
            **_distribution_summary(macro_distribution),
            "n_groups": macro_groups,
            "n_rows": macro_rows,
            "low_sample": False,
            "macro_eligible": True,
            "n_subsets": len(macro_points),
        }
    )
    return pd.DataFrame(rows)
