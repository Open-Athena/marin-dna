"""Lightweight CPU primitives for the exp420 prediction-target analysis."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl

ISSUE = 420
D_SAE = 15_360
ORIENTATIONS = ("forward", "reverse_complement")
TRANSFORMS = ("signed", "absolute")
BOOTSTRAPS = 2_000
PERMUTATIONS = 10_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_panel(frame: pl.DataFrame) -> None:
    required = {
        "chrom",
        "pos",
        "ref",
        "alt",
        "label",
        "subset",
        "match_group",
        "split",
    }
    assert required <= set(frame.columns), required - set(frame.columns)
    assert frame.height > 0
    assert (
        frame.select([pl.col(column).null_count() for column in sorted(required)])
        .sum_horizontal()
        .item()
        == 0
    )
    assert set(frame["label"]) == {0, 1}
    assert set(frame["split"]) == {"discovery", "validation", "test"}
    assert (frame["pos"] > 0).all()
    assert (
        frame.select(pl.struct("chrom", "pos", "ref", "alt").n_unique()).item()
        == frame.height
    )
    groups = frame.group_by("match_group").agg(
        pl.len().alias("rows"),
        pl.col("label").sum().alias("positive"),
        pl.col("subset").n_unique().alias("subsets"),
        pl.col("split").n_unique().alias("splits"),
        pl.col("chrom").n_unique().alias("chromosomes"),
    )
    assert (groups["rows"] == 10).all()
    assert (groups["positive"] == 1).all()
    assert (groups["subsets"] == 1).all()
    assert (groups["splits"] == 1).all()
    assert (groups["chromosomes"] == 1).all()


def matched_contrasts(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    assert scores.ndim in (1, 2)
    assert labels.shape == groups.shape == (scores.shape[0],)
    assert set(np.unique(labels)) == {0, 1}
    group_order = np.asarray(list(dict.fromkeys(groups.tolist())))
    output_shape = (len(group_order),) + scores.shape[1:]
    output = np.empty(output_shape, dtype=np.float32)
    for index, group in enumerate(group_order):
        selected = np.flatnonzero(groups == group)
        assert selected.shape == (10,)
        positive = selected[labels[selected] == 1]
        controls = selected[labels[selected] == 0]
        assert positive.shape == (1,) and controls.shape == (9,)
        output[index] = scores[positive[0]] - scores[controls].mean(axis=0)
    assert np.isfinite(output).all()
    return output, group_order


def standardized_means(contrasts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    assert contrasts.ndim == 2 and contrasts.shape[0] >= 2
    means = contrasts.mean(axis=0, dtype=np.float64)
    standard_errors = contrasts.std(axis=0, ddof=1, dtype=np.float64) / np.sqrt(
        contrasts.shape[0]
    )
    statistics = np.divide(
        means,
        standard_errors,
        out=np.zeros_like(means),
        where=standard_errors > 0,
    )
    assert np.isfinite(means).all() and np.isfinite(statistics).all()
    return means, statistics


def bootstrap_mean_interval(
    values: np.ndarray, *, seed: int, samples: int = BOOTSTRAPS
) -> tuple[float, float]:
    assert values.ndim == 1 and len(values) >= 2 and samples > 0
    rng = np.random.default_rng(seed)
    boot = np.empty(samples, dtype=np.float64)
    for offset in range(0, samples, 250):
        size = min(250, samples - offset)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        boot[offset : offset + size] = values[indices].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return float(low), float(high)


def matched_permutation_pvalue(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    permutations: int = PERMUTATIONS,
) -> float:
    assert scores.ndim == 1
    observed, group_order = matched_contrasts(scores, labels, groups)
    statistic = abs(float(observed.mean()))
    group_values: list[np.ndarray] = []
    for group in group_order:
        values = scores[groups == group]
        assert values.shape == (10,)
        group_values.append(values)
    matrix = np.stack(group_values)
    group_sums = matrix.sum(axis=1)
    rng = np.random.default_rng(seed)
    exceed = 0
    for offset in range(0, permutations, 500):
        size = min(500, permutations - offset)
        chosen = rng.integers(0, 10, size=(size, len(group_order)))
        positives = np.take_along_axis(
            np.broadcast_to(matrix, (size, *matrix.shape)),
            chosen[:, :, None],
            axis=2,
        )[:, :, 0]
        contrasts = positives - (group_sums[None, :] - positives) / 9
        null_statistics = np.abs(contrasts.mean(axis=1))
        exceed += int(np.count_nonzero(null_statistics >= statistic))
    return float((exceed + 1) / (permutations + 1))


@dataclass(frozen=True)
class FeatureSelection:
    feature_id: int
    transform: Literal["signed", "absolute"]
    direction: int
    discovery_effect: float
    validation_effect: float
    validation_direction_consistent: bool
    discovery_positive_support: int
    discovery_negative_support: int
    validation_positive_support: int
    validation_negative_support: int
    scores: np.ndarray

    def metadata(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "transform": self.transform,
            "direction": self.direction,
            "discovery_effect": self.discovery_effect,
            "validation_effect": self.validation_effect,
            "validation_direction_consistent": self.validation_direction_consistent,
            "discovery_positive_support": self.discovery_positive_support,
            "discovery_negative_support": self.discovery_negative_support,
            "validation_positive_support": self.validation_positive_support,
            "validation_negative_support": self.validation_negative_support,
        }


def load_dense_delta(path: Path, *, rows: int, features: int) -> np.ndarray:
    assert path.is_file() and rows > 0 and features > 0
    sparse = pl.read_parquet(path, columns=["row_index", "feature_id", "delta"])
    assert sparse.height > rows
    assert sparse.null_count().sum_horizontal().sum() == 0
    row_index = sparse["row_index"].to_numpy().astype(np.int64, copy=False)
    feature_id = sparse["feature_id"].to_numpy().astype(np.int64, copy=False)
    delta = sparse["delta"].to_numpy().astype(np.float32, copy=False)
    assert row_index.min() == 0 and row_index.max() == rows - 1
    assert feature_id.min() >= 0 and feature_id.max() < features
    assert np.isfinite(delta).all()
    keys = row_index * features + feature_id
    assert np.unique(keys).size == len(keys)
    matrix = np.zeros((rows, features), dtype=np.float32)
    matrix[row_index, feature_id] = delta
    assert np.isfinite(matrix).all()
    return matrix


def transformed(values: np.ndarray, transform: str) -> np.ndarray:
    assert transform in TRANSFORMS
    return values if transform == "signed" else np.abs(values)


def standardized_difference(
    values: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert values.ndim == 2 and target.shape == (values.shape[0],)
    assert target.dtype == np.bool_ and target.any() and (~target).any()
    positive = values[target]
    negative = values[~target]
    positive_support = np.count_nonzero(positive, axis=0)
    negative_support = np.count_nonzero(negative, axis=0)
    numerator = positive.mean(axis=0, dtype=np.float64) - negative.mean(
        axis=0, dtype=np.float64
    )
    pooled = np.sqrt(
        (
            positive.var(axis=0, ddof=1, dtype=np.float64)
            + negative.var(axis=0, ddof=1, dtype=np.float64)
        )
        / 2
    )
    effect = np.divide(
        numerator, pooled, out=np.zeros_like(numerator), where=pooled > 0
    )
    effect[~np.isfinite(effect)] = 0
    return effect, positive_support, negative_support


def select_feature(
    matrix: np.ndarray,
    target: np.ndarray,
    splits: np.ndarray,
    task_rows: np.ndarray,
    *,
    top_k: int,
    min_nonzero: int,
) -> tuple[FeatureSelection, list[dict[str, Any]]]:
    assert matrix.ndim == 2 and target.shape == splits.shape == (matrix.shape[0],)
    discovery_rows = task_rows & (splits == "discovery")
    validation_rows = task_rows & (splits == "validation")
    assert target[discovery_rows].any() and (~target[discovery_rows]).any()
    assert target[validation_rows].any() and (~target[validation_rows]).any()
    candidates: list[dict[str, Any]] = []
    for transform in TRANSFORMS:
        effect, positive_support, negative_support = standardized_difference(
            transformed(matrix[discovery_rows], transform), target[discovery_rows]
        )
        eligible = (
            (positive_support >= min_nonzero)
            & (negative_support >= min_nonzero)
            & np.isfinite(effect)
        )
        score = np.where(eligible, np.abs(effect), -np.inf)
        keep = min(top_k, int(eligible.sum()))
        assert keep > 0
        order = np.argsort(-score, kind="stable")[:keep]
        for rank, feature_id in enumerate(order, start=1):
            candidates.append(
                {
                    "transform": transform,
                    "feature_id": int(feature_id),
                    "discovery_rank_within_transform": rank,
                    "discovery_effect": float(effect[feature_id]),
                    "discovery_positive_support": int(positive_support[feature_id]),
                    "discovery_negative_support": int(negative_support[feature_id]),
                }
            )
    validation_cache = {
        transform: standardized_difference(
            transformed(matrix[validation_rows], transform), target[validation_rows]
        )
        for transform in TRANSFORMS
    }
    for candidate in candidates:
        effect, positive_support, negative_support = validation_cache[
            candidate["transform"]
        ]
        feature_id = candidate["feature_id"]
        validation_effect = float(effect[feature_id])
        consistent = bool(
            np.sign(validation_effect) == np.sign(candidate["discovery_effect"])
            and positive_support[feature_id] >= min_nonzero
            and negative_support[feature_id] >= min_nonzero
        )
        candidate.update(
            {
                "validation_effect": validation_effect,
                "validation_positive_support": int(positive_support[feature_id]),
                "validation_negative_support": int(negative_support[feature_id]),
                "validation_direction_consistent": consistent,
            }
        )
    consistent = [row for row in candidates if row["validation_direction_consistent"]]
    ranked = sorted(
        consistent or candidates,
        key=lambda row: (
            -abs(row["validation_effect"]),
            -abs(row["discovery_effect"]),
            row["transform"],
            row["feature_id"],
        ),
    )
    chosen = ranked[0]
    direction = 1 if chosen["discovery_effect"] > 0 else -1
    scores = transformed(matrix[:, chosen["feature_id"]], chosen["transform"])
    return (
        FeatureSelection(
            feature_id=chosen["feature_id"],
            transform=chosen["transform"],
            direction=direction,
            discovery_effect=chosen["discovery_effect"],
            validation_effect=chosen["validation_effect"],
            validation_direction_consistent=chosen["validation_direction_consistent"],
            discovery_positive_support=chosen["discovery_positive_support"],
            discovery_negative_support=chosen["discovery_negative_support"],
            validation_positive_support=chosen["validation_positive_support"],
            validation_negative_support=chosen["validation_negative_support"],
            scores=np.asarray(scores, dtype=np.float32),
        ),
        candidates,
    )


def standardize_pretest(
    selection: FeatureSelection, train_rows: np.ndarray
) -> tuple[np.ndarray, float, float]:
    assert train_rows.dtype == np.bool_ and train_rows.shape == selection.scores.shape
    oriented = selection.direction * selection.scores.astype(np.float64)
    center = float(oriented[train_rows].mean())
    scale = float(oriented[train_rows].std(ddof=1))
    assert np.isfinite(center) and np.isfinite(scale) and scale > 0
    return (oriented - center) / scale, center, scale
