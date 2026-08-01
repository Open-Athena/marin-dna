"""Analyze Mendelian consequence classes from all-feature SAE ref/alt deltas."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import matplotlib
import numpy as np
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

from analysis import D_SAE, ISSUE, ORIENTATIONS, _sha256, _validate_panel

matplotlib.use("Agg")
import matplotlib.pyplot as plt

POSITIVE_CLASS = "missense_variant"
NEGATIVE_CLASS = "synonymous_variant"
BINARY_CLASSES = (POSITIVE_CLASS, NEGATIVE_CLASS)
DESCRIPTIVE_SUBSETS = frozenset({"mature_miRNA_variant"})
TRANSFORMS = ("signed", "absolute")
TOP_DISCOVERY_CANDIDATES = 32
MIN_NONZERO_PER_CLASS = 8
MIN_MULTICLASS_GROUPS_PER_SPLIT = 20
BOOTSTRAPS = 2_000
SEED = 420_2


def _seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in (SEED, *parts))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


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
    """Reconstruct one orientation's dense delta matrix from the sparse union."""

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


def _transformed(values: np.ndarray, transform: str) -> np.ndarray:
    assert transform in TRANSFORMS
    return values if transform == "signed" else np.abs(values)


def standardized_difference(
    values: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return vectorized target-minus-other standardized effects and support."""

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
        numerator,
        pooled,
        out=np.zeros_like(numerator),
        where=pooled > 0,
    )
    effect[~np.isfinite(effect)] = 0
    return effect, positive_support, negative_support


def select_feature(
    matrix: np.ndarray,
    target: np.ndarray,
    splits: np.ndarray,
    task_rows: np.ndarray,
    *,
    top_k: int = TOP_DISCOVERY_CANDIDATES,
    min_nonzero: int = MIN_NONZERO_PER_CLASS,
) -> tuple[FeatureSelection, list[dict[str, Any]]]:
    """Rank on discovery and choose direction-consistent candidate on validation."""

    assert matrix.ndim == 2 and target.shape == splits.shape == (matrix.shape[0],)
    assert task_rows.dtype == np.bool_ and task_rows.shape == target.shape
    discovery_rows = task_rows & (splits == "discovery")
    validation_rows = task_rows & (splits == "validation")
    assert target[discovery_rows].any() and (~target[discovery_rows]).any()
    assert target[validation_rows].any() and (~target[validation_rows]).any()
    candidates: list[dict[str, Any]] = []
    for transform in TRANSFORMS:
        values = _transformed(matrix[discovery_rows], transform)
        effect, positive_support, negative_support = standardized_difference(
            values, target[discovery_rows]
        )
        eligible = (
            (positive_support >= min_nonzero)
            & (negative_support >= min_nonzero)
            & np.isfinite(effect)
        )
        score = np.where(eligible, np.abs(effect), -np.inf)
        available = int(eligible.sum())
        assert available > 0
        keep = min(top_k, available)
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
    assert candidates
    validation_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for transform in TRANSFORMS:
        validation_cache[transform] = standardized_difference(
            _transformed(matrix[validation_rows], transform),
            target[validation_rows],
        )
    for candidate in candidates:
        effect, positive_support, negative_support = validation_cache[
            candidate["transform"]
        ]
        feature_id = candidate["feature_id"]
        validation_effect = float(effect[feature_id])
        direction_consistent = (
            np.sign(validation_effect) == np.sign(candidate["discovery_effect"])
            and positive_support[feature_id] >= min_nonzero
            and negative_support[feature_id] >= min_nonzero
        )
        candidate.update(
            {
                "validation_effect": validation_effect,
                "validation_positive_support": int(positive_support[feature_id]),
                "validation_negative_support": int(negative_support[feature_id]),
                "validation_direction_consistent": bool(direction_consistent),
            }
        )
    consistent = [
        candidate
        for candidate in candidates
        if candidate["validation_direction_consistent"]
    ]
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
    scores = _transformed(matrix[:, chosen["feature_id"]], chosen["transform"])
    selection = FeatureSelection(
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
    )
    return selection, candidates


def standardize_pretest(
    selection: FeatureSelection, train_rows: np.ndarray
) -> tuple[np.ndarray, float, float]:
    assert train_rows.dtype == np.bool_ and train_rows.shape == selection.scores.shape
    oriented = selection.direction * selection.scores.astype(np.float64)
    center = float(oriented[train_rows].mean())
    scale = float(oriented[train_rows].std(ddof=1))
    assert np.isfinite(center) and np.isfinite(scale) and scale > 0
    return (oriented - center) / scale, center, scale


def cluster_bootstrap_binary(
    target: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    samples: int = BOOTSTRAPS,
) -> dict[str, float]:
    """Bootstrap groups within each class for AUROC and mean difference."""

    assert target.shape == scores.shape == groups.shape
    assert target.dtype == np.bool_ and target.any() and (~target).any()
    group_to_rows = {
        group: np.flatnonzero(groups == group) for group in np.unique(groups)
    }
    positive_groups = np.unique(groups[target])
    negative_groups = np.unique(groups[~target])
    assert all(target[group_to_rows[group]].all() for group in positive_groups)
    assert all((~target[group_to_rows[group]]).all() for group in negative_groups)
    rng = np.random.default_rng(seed)
    aucs = np.empty(samples, dtype=np.float64)
    effects = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        sampled_groups = np.concatenate(
            (
                rng.choice(positive_groups, len(positive_groups), replace=True),
                rng.choice(negative_groups, len(negative_groups), replace=True),
            )
        )
        indices = np.concatenate([group_to_rows[group] for group in sampled_groups])
        sampled_target = target[indices]
        sampled_scores = scores[indices]
        aucs[sample] = roc_auc_score(sampled_target, sampled_scores)
        effects[sample] = (
            sampled_scores[sampled_target].mean()
            - sampled_scores[~sampled_target].mean()
        )
    return {
        "auc": float(roc_auc_score(target, scores)),
        "auc_ci95_low": float(np.quantile(aucs, 0.025)),
        "auc_ci95_high": float(np.quantile(aucs, 0.975)),
        "mean_difference": float(scores[target].mean() - scores[~target].mean()),
        "mean_difference_ci95_low": float(np.quantile(effects, 0.025)),
        "mean_difference_ci95_high": float(np.quantile(effects, 0.975)),
        "positive_rows": int(target.sum()),
        "negative_rows": int((~target).sum()),
        "positive_groups": len(positive_groups),
        "negative_groups": len(negative_groups),
    }


def conditional_auc(
    target: np.ndarray, scores: np.ndarray, strata: np.ndarray
) -> dict[str, float | int]:
    """Compute pair-count-weighted AUROC within strata containing both classes."""

    assert target.shape == scores.shape == strata.shape
    numerator = 0.0
    pairs = 0
    covered = np.zeros(len(target), dtype=np.bool_)
    retained_strata = 0
    for stratum in np.unique(strata):
        rows = strata == stratum
        positives = int(target[rows].sum())
        negatives = int((~target[rows]).sum())
        if positives == 0 or negatives == 0:
            continue
        pair_count = positives * negatives
        numerator += roc_auc_score(target[rows], scores[rows]) * pair_count
        pairs += pair_count
        covered |= rows
        retained_strata += 1
    assert pairs > 0
    return {
        "auc": float(numerator / pairs),
        "pairs": pairs,
        "covered_rows": int(covered.sum()),
        "covered_fraction": float(covered.mean()),
        "retained_strata": retained_strata,
    }


def categorical_rate_scores(
    train_target: np.ndarray,
    train_strata: np.ndarray,
    test_strata: np.ndarray,
) -> np.ndarray:
    """Fit Laplace-smoothed categorical class rates without test labels."""

    assert train_target.shape == train_strata.shape
    overall = float((train_target.sum() + 1) / (len(train_target) + 2))
    rates: dict[Any, float] = {}
    for stratum in np.unique(train_strata):
        values = train_target[train_strata == stratum]
        rates[stratum] = float((values.sum() + 1) / (len(values) + 2))
    return np.asarray([rates.get(value, overall) for value in test_strata])


def _binary_strata(
    panel: pl.DataFrame, contexts: pl.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    substitution = np.asarray(
        [f"{ref}>{alt}" for ref, alt in zip(panel["ref"], panel["alt"], strict=True)]
    )
    gc_bin = contexts["flank_gc_bin"].to_numpy()
    substitution_gc = np.asarray(
        [
            f"{change}|gc{int(bin_value)}"
            for change, bin_value in zip(substitution, gc_bin, strict=True)
        ]
    )
    return substitution, substitution_gc


def _task_rows(
    labels: np.ndarray,
    subsets: np.ndarray,
    *,
    label: int,
    classes: tuple[str, ...],
) -> np.ndarray:
    return (labels == label) & np.isin(subsets, classes)


def _candidate_rows(
    candidates: list[dict[str, Any]],
    *,
    analysis: str,
    label: int,
    orientation: str,
    target_class: str,
) -> list[dict[str, Any]]:
    return [
        {
            "analysis": analysis,
            "label": label,
            "orientation": orientation,
            "target_class": target_class,
            **candidate,
        }
        for candidate in candidates
    ]


def _evaluate_binary_view(
    *,
    view: str,
    scores: np.ndarray,
    target: np.ndarray,
    test_rows: np.ndarray,
    groups: np.ndarray,
    substitution: np.ndarray,
    substitution_gc: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    metrics = cluster_bootstrap_binary(
        target[test_rows],
        scores[test_rows],
        groups[test_rows],
        seed=seed,
    )
    return {
        "view": view,
        **metrics,
        "substitution_conditional": conditional_auc(
            target[test_rows], scores[test_rows], substitution[test_rows]
        ),
        "substitution_gc_conditional": conditional_auc(
            target[test_rows], scores[test_rows], substitution_gc[test_rows]
        ),
    }


def _top_context_rows(
    *,
    panel: pl.DataFrame,
    contexts: pl.DataFrame,
    scores: np.ndarray,
    test_rows: np.ndarray,
    target: np.ndarray,
    label: int,
    view: str,
    count: int = 5,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for class_value, class_name in ((True, POSITIVE_CLASS), (False, NEGATIVE_CLASS)):
        eligible = np.flatnonzero(test_rows & (target == class_value))
        order = eligible[np.argsort(-scores[eligible], kind="stable")[:count]]
        for rank, row_index in enumerate(order, start=1):
            panel_row = panel.row(int(row_index), named=True)
            context_row = contexts.row(int(row_index), named=True)
            output.append(
                {
                    "label": label,
                    "view": view,
                    "class": class_name,
                    "rank": rank,
                    "score": float(scores[row_index]),
                    "row_index": int(row_index),
                    "chrom": panel_row["chrom"],
                    "pos1": panel_row["pos"],
                    "pos0": panel_row["pos"] - 1,
                    "ref": panel_row["ref"],
                    "alt": panel_row["alt"],
                    "match_group": panel_row["match_group"],
                    "ref_context": context_row["ref_context"],
                    "alt_context": context_row["alt_context"],
                    "flank_gc_count": context_row["flank_gc_count"],
                }
            )
    return output


def _multiclass_metrics(
    truth: np.ndarray,
    score_matrix: np.ndarray,
    classes: tuple[str, ...],
) -> dict[str, Any]:
    assert score_matrix.shape == (len(truth), len(classes))
    prediction = np.asarray(classes)[np.argmax(score_matrix, axis=1)]
    per_class_auc = {
        target_class: float(
            roc_auc_score(truth == target_class, score_matrix[:, class_index])
        )
        for class_index, target_class in enumerate(classes)
    }
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1": float(
            f1_score(truth, prediction, labels=list(classes), average="macro")
        ),
        "per_class_auc": per_class_auc,
        "confusion": {
            true_class: {
                predicted_class: int(
                    np.sum((truth == true_class) & (prediction == predicted_class))
                )
                for predicted_class in classes
            }
            for true_class in classes
        },
    }


def _bootstrap_multiclass_macro_f1(
    truth: np.ndarray,
    score_matrix: np.ndarray,
    groups: np.ndarray,
    classes: tuple[str, ...],
    *,
    seed: int,
    samples: int = BOOTSTRAPS,
) -> tuple[float, float]:
    group_to_rows = {
        group: np.flatnonzero(groups == group) for group in np.unique(groups)
    }
    groups_by_class = {
        target_class: np.unique(groups[truth == target_class])
        for target_class in classes
    }
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        sampled_groups = np.concatenate(
            [
                rng.choice(class_groups, len(class_groups), replace=True)
                for class_groups in groups_by_class.values()
            ]
        )
        rows = np.concatenate([group_to_rows[group] for group in sampled_groups])
        predicted = np.asarray(classes)[np.argmax(score_matrix[rows], axis=1)]
        values[sample] = f1_score(
            truth[rows], predicted, labels=list(classes), average="macro"
        )
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _support_table(panel: pl.DataFrame) -> tuple[pl.DataFrame, tuple[str, ...]]:
    support = (
        panel.filter(~pl.col("subset").is_in(sorted(DESCRIPTIVE_SUBSETS)))
        .group_by("subset", "split")
        .agg(pl.col("match_group").n_unique().alias("groups"))
        .sort("subset", "split")
    )
    minima = support.group_by("subset").agg(pl.col("groups").min().alias("minimum"))
    eligible = tuple(
        sorted(
            minima.filter(pl.col("minimum") >= MIN_MULTICLASS_GROUPS_PER_SPLIT)[
                "subset"
            ].to_list()
        )
    )
    assert POSITIVE_CLASS in eligible
    return support, eligible


def _plot_results(
    binary: pl.DataFrame, multiclass: pl.DataFrame, output_dir: Path
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    colors = {0: "#0072B2", 1: "#D55E00"}
    views = ["forward", "reverse_complement", "aggregate"]
    x = np.arange(len(views))
    for label in (0, 1):
        rows = binary.filter(pl.col("label") == label).sort(
            pl.col("view").replace_strict(views, list(range(len(views))))
        )
        assert rows.height == len(views)
        axes[0].errorbar(
            x + (-0.04 if label == 0 else 0.04),
            rows["auc"],
            yerr=np.vstack(
                (
                    rows["auc"] - rows["auc_ci95_low"],
                    rows["auc_ci95_high"] - rows["auc"],
                )
            ),
            marker="o",
            capsize=0,
            color=colors[label],
            label=f"label={label}",
        )
        multi = multiclass.filter(pl.col("label") == label).sort(
            pl.col("view").replace_strict(views, list(range(len(views))))
        )
        assert multi.height == len(views)
        axes[1].errorbar(
            x + (-0.04 if label == 0 else 0.04),
            multi["macro_f1"],
            yerr=np.vstack(
                (
                    multi["macro_f1"] - multi["macro_f1_ci95_low"],
                    multi["macro_f1_ci95_high"] - multi["macro_f1"],
                )
            ),
            marker="o",
            capsize=0,
            color=colors[label],
            label=f"label={label}",
        )
    axes[0].axhline(0.5, color="grey", linewidth=0.8, linestyle="--")
    axes[1].axhline(0.25, color="grey", linewidth=0.8, linestyle="--")
    axes[0].set_title("Missense vs synonymous")
    axes[0].set_ylabel("Held-out AUROC")
    axes[1].set_title("Four-class consequence score")
    axes[1].set_ylabel("Held-out macro-F1")
    for axis in axes:
        axis.set_xticks(x, ["FWD", "RC", "aggregate"])
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend()
    figure.suptitle("exp420: label-stratified SAE consequence discrimination")
    figure.savefig(output_dir / "subset_summary.png", dpi=180)
    figure.savefig(output_dir / "subset_summary.svg")
    plt.close(figure)


def _markdown(
    binary: pl.DataFrame,
    baselines: pl.DataFrame,
    multiclass: pl.DataFrame,
    eligible_classes: tuple[str, ...],
) -> str:
    lines = [
        "# exp420 direct consequence discrimination",
        "",
        "Features and transforms were ranked on discovery chromosomes, direction-gated and selected on validation chromosomes, and evaluated once on chr11/X. FWD and RC were selected separately; the aggregate is their equal-weight pretest-standardized score mean.",
        "",
        "## Missense versus synonymous",
        "",
        "| label | view | feature | transform | gate | AUROC (95% group-bootstrap CI) | substitution-conditional | substitution+GC-conditional |",
        "|---:|---|---:|---|---|---:|---:|---:|",
    ]
    for row in binary.sort("label", "view").to_dicts():
        feature = (
            "two selected IDs" if row["view"] == "aggregate" else row["feature_id"]
        )
        transform = "score mean" if row["view"] == "aggregate" else row["transform"]
        lines.append(
            f"| {row['label']} | {row['view']} | {feature} | {transform} | "
            f"{row['validation_direction_consistent']} | {row['auc']:.3f} "
            f"[{row['auc_ci95_low']:.3f}, {row['auc_ci95_high']:.3f}] | "
            f"{row['substitution_conditional_auc']:.3f} | "
            f"{row['substitution_gc_conditional_auc']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Categorical controls fitted without test labels:",
            "",
            "| label | substitution AUROC | substitution+GC AUROC |",
            "|---:|---:|---:|",
        ]
    )
    for row in baselines.sort("label").to_dicts():
        lines.append(
            f"| {row['label']} | {row['substitution_auc']:.3f} | "
            f"{row['substitution_gc_auc']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Supported multiclass subset",
            "",
            f"Eligible classes (>= {MIN_MULTICLASS_GROUPS_PER_SPLIT} groups in every split): "
            + ", ".join(f"`{value}`" for value in eligible_classes)
            + ".",
            "",
            "| label | view | macro-F1 (95% group-bootstrap CI) | balanced accuracy | accuracy | all validation gates |",
            "|---:|---|---:|---:|---:|---|",
        ]
    )
    for row in multiclass.sort("label", "view").to_dicts():
        lines.append(
            f"| {row['label']} | {row['view']} | {row['macro_f1']:.3f} "
            f"[{row['macro_f1_ci95_low']:.3f}, {row['macro_f1_ci95_high']:.3f}] | "
            f"{row['balanced_accuracy']:.3f} | {row['accuracy']:.3f} | "
            f"{row['all_validation_gates']} |"
        )
    lines.append("")
    return "\n".join(lines)


def analyze_subsets(
    *,
    panel_path: Path,
    extraction_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    assert panel_path.is_file() and extraction_dir.is_dir()
    assert not output_dir.exists()
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert len(experiment_commit) == 40
    assert all(character in "0123456789abcdef" for character in experiment_commit)
    extraction_manifest_path = extraction_dir / "manifest.json"
    extraction_manifest = json.loads(extraction_manifest_path.read_text())
    assert extraction_manifest["experiment_commit"] == experiment_commit
    assert extraction_manifest["panel"]["sha256"] == _sha256(panel_path)
    for name, metadata in extraction_manifest["artifacts"].items():
        path = extraction_dir / name
        assert path.stat().st_size == metadata["bytes"]
        assert _sha256(path) == metadata["sha256"]

    panel = pl.read_parquet(panel_path)
    _validate_panel(panel)
    contexts = pl.read_parquet(extraction_dir / "variant_contexts.parquet").sort(
        "row_index"
    )
    assert contexts.height == panel.height
    assert contexts["row_index"].to_list() == list(range(panel.height))
    output_dir.mkdir(parents=True, exist_ok=False)

    labels = panel["label"].to_numpy().astype(np.int8)
    subsets = panel["subset"].to_numpy()
    splits = panel["split"].to_numpy()
    groups = panel["match_group"].to_numpy()
    substitution, substitution_gc = _binary_strata(panel, contexts)
    support, eligible_classes = _support_table(panel)
    assert len(eligible_classes) == 4, eligible_classes

    binary_selections: dict[tuple[int, str], FeatureSelection] = {}
    multiclass_selections: dict[tuple[int, str, str], FeatureSelection] = {}
    candidate_rows: list[dict[str, Any]] = []
    for orientation in ORIENTATIONS:
        matrix = load_dense_delta(
            extraction_dir / f"sae_activations_{orientation}.parquet",
            rows=panel.height,
            features=D_SAE,
        )
        for label in (0, 1):
            binary_rows = _task_rows(
                labels, subsets, label=label, classes=BINARY_CLASSES
            )
            binary_target = subsets == POSITIVE_CLASS
            selection, candidates = select_feature(
                matrix, binary_target, splits, binary_rows
            )
            binary_selections[(label, orientation)] = selection
            candidate_rows.extend(
                _candidate_rows(
                    candidates,
                    analysis="missense_vs_synonymous",
                    label=label,
                    orientation=orientation,
                    target_class=POSITIVE_CLASS,
                )
            )
            multiclass_rows = _task_rows(
                labels, subsets, label=label, classes=eligible_classes
            )
            for target_class in eligible_classes:
                target = subsets == target_class
                class_selection, class_candidates = select_feature(
                    matrix, target, splits, multiclass_rows
                )
                multiclass_selections[(label, orientation, target_class)] = (
                    class_selection
                )
                candidate_rows.extend(
                    _candidate_rows(
                        class_candidates,
                        analysis="eligible_multiclass_one_vs_rest",
                        label=label,
                        orientation=orientation,
                        target_class=target_class,
                    )
                )
        del matrix

    binary_rows_out: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    for label in (0, 1):
        task_rows = _task_rows(labels, subsets, label=label, classes=BINARY_CLASSES)
        target = subsets == POSITIVE_CLASS
        train_rows = task_rows & (splits != "test")
        test_rows = task_rows & (splits == "test")
        standardized: dict[str, np.ndarray] = {}
        standardizers: dict[str, tuple[float, float]] = {}
        for orientation in ORIENTATIONS:
            selection = binary_selections[(label, orientation)]
            standardized[orientation], center, scale = standardize_pretest(
                selection, train_rows
            )
            standardizers[orientation] = (center, scale)
            result = _evaluate_binary_view(
                view=orientation,
                scores=standardized[orientation],
                target=target,
                test_rows=test_rows,
                groups=groups,
                substitution=substitution,
                substitution_gc=substitution_gc,
                seed=_seed("binary", label, orientation),
            )
            binary_rows_out.append(
                {
                    "label": label,
                    **result,
                    **selection.metadata(),
                    "pretest_center": center,
                    "pretest_scale": scale,
                }
            )
            context_rows.extend(
                _top_context_rows(
                    panel=panel,
                    contexts=contexts,
                    scores=standardized[orientation],
                    test_rows=test_rows,
                    target=target,
                    label=label,
                    view=orientation,
                )
            )
        aggregate = (standardized["forward"] + standardized["reverse_complement"]) / 2
        aggregate_result = _evaluate_binary_view(
            view="aggregate",
            scores=aggregate,
            target=target,
            test_rows=test_rows,
            groups=groups,
            substitution=substitution,
            substitution_gc=substitution_gc,
            seed=_seed("binary", label, "aggregate"),
        )
        binary_rows_out.append(
            {
                "label": label,
                **aggregate_result,
                "feature_id": None,
                "transform": None,
                "direction": None,
                "discovery_effect": None,
                "validation_effect": None,
                "validation_direction_consistent": all(
                    binary_selections[
                        (label, orientation)
                    ].validation_direction_consistent
                    for orientation in ORIENTATIONS
                ),
                "discovery_positive_support": None,
                "discovery_negative_support": None,
                "validation_positive_support": None,
                "validation_negative_support": None,
                "pretest_center": None,
                "pretest_scale": None,
            }
        )
        context_rows.extend(
            _top_context_rows(
                panel=panel,
                contexts=contexts,
                scores=aggregate,
                test_rows=test_rows,
                target=target,
                label=label,
                view="aggregate",
            )
        )
        train_target = target[train_rows]
        test_target = target[test_rows]
        substitution_score = categorical_rate_scores(
            train_target, substitution[train_rows], substitution[test_rows]
        )
        substitution_gc_score = categorical_rate_scores(
            train_target, substitution_gc[train_rows], substitution_gc[test_rows]
        )
        baseline_rows.append(
            {
                "label": label,
                "substitution_auc": float(
                    roc_auc_score(test_target, substitution_score)
                ),
                "substitution_gc_auc": float(
                    roc_auc_score(test_target, substitution_gc_score)
                ),
                "test_positive_rows": int(test_target.sum()),
                "test_negative_rows": int((~test_target).sum()),
            }
        )

    multiclass_rows_out: list[dict[str, Any]] = []
    multiclass_selection_rows: list[dict[str, Any]] = []
    for label in (0, 1):
        task_rows = _task_rows(labels, subsets, label=label, classes=eligible_classes)
        train_rows = task_rows & (splits != "test")
        test_rows = task_rows & (splits == "test")
        truth = subsets[test_rows]
        score_by_view: dict[str, np.ndarray] = {}
        for orientation in ORIENTATIONS:
            columns: list[np.ndarray] = []
            for target_class in eligible_classes:
                selection = multiclass_selections[(label, orientation, target_class)]
                standardized, center, scale = standardize_pretest(selection, train_rows)
                columns.append(standardized[test_rows])
                multiclass_selection_rows.append(
                    {
                        "label": label,
                        "orientation": orientation,
                        "target_class": target_class,
                        **selection.metadata(),
                        "pretest_center": center,
                        "pretest_scale": scale,
                    }
                )
            score_by_view[orientation] = np.column_stack(columns)
        score_by_view["aggregate"] = (
            score_by_view["forward"] + score_by_view["reverse_complement"]
        ) / 2
        for view, score_matrix in score_by_view.items():
            metrics = _multiclass_metrics(truth, score_matrix, eligible_classes)
            low, high = _bootstrap_multiclass_macro_f1(
                truth,
                score_matrix,
                groups[test_rows],
                eligible_classes,
                seed=_seed("multiclass", label, view),
            )
            if view == "aggregate":
                gate = all(
                    multiclass_selections[
                        (label, orientation, target_class)
                    ].validation_direction_consistent
                    for orientation in ORIENTATIONS
                    for target_class in eligible_classes
                )
            else:
                gate = all(
                    multiclass_selections[
                        (label, view, target_class)
                    ].validation_direction_consistent
                    for target_class in eligible_classes
                )
            multiclass_rows_out.append(
                {
                    "label": label,
                    "view": view,
                    **metrics,
                    "macro_f1_ci95_low": low,
                    "macro_f1_ci95_high": high,
                    "all_validation_gates": gate,
                    "test_rows": int(test_rows.sum()),
                    "test_groups": int(np.unique(groups[test_rows]).size),
                }
            )

    binary = pl.DataFrame(binary_rows_out).unnest(
        ["substitution_conditional", "substitution_gc_conditional"],
        separator="_",
    )
    baselines = pl.DataFrame(baseline_rows)
    candidates = pl.DataFrame(candidate_rows)
    binary_contexts = pl.DataFrame(context_rows)
    multiclass = pl.DataFrame(multiclass_rows_out)
    multiclass_selections_frame = pl.DataFrame(multiclass_selection_rows)
    output_tables = {
        "binary_summary": binary,
        "categorical_baselines": baselines,
        "candidates": candidates,
        "binary_contexts": binary_contexts,
        "class_support": support,
        "multiclass_summary": multiclass,
        "multiclass_selections": multiclass_selections_frame,
    }
    for name, table in output_tables.items():
        assert table.height > 0
        table.write_parquet(output_dir / f"{name}.parquet", compression="zstd")
    _plot_results(binary, multiclass, output_dir)
    (output_dir / "RESULTS.md").write_text(
        _markdown(binary, baselines, multiclass, eligible_classes)
    )

    result: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "issue": ISSUE,
        "experiment_commit": experiment_commit,
        "panel": {"sha256": _sha256(panel_path), "rows": panel.height},
        "extraction": {
            "manifest_sha256": _sha256(extraction_manifest_path),
            "manifest": extraction_manifest,
        },
        "protocol": {
            "positive_class": POSITIVE_CLASS,
            "negative_class": NEGATIVE_CLASS,
            "transforms": list(TRANSFORMS),
            "top_discovery_candidates_per_transform": TOP_DISCOVERY_CANDIDATES,
            "minimum_nonzero_per_class": MIN_NONZERO_PER_CLASS,
            "minimum_multiclass_groups_per_split": MIN_MULTICLASS_GROUPS_PER_SPLIT,
            "eligible_multiclass_classes": list(eligible_classes),
            "bootstraps": BOOTSTRAPS,
            "primary_aggregate": "equal mean of independently selected, direction-oriented, pretest-standardized FWD/RC scores",
            "conditional_controls": [
                "ref>alt substitution",
                "ref>alt substitution + focal-excluded 40-bp GC bin",
            ],
        },
    }
    _write_json(output_dir / "results.json", result)
    artifact_files = sorted(path for path in output_dir.iterdir() if path.is_file())
    manifest = {
        **result,
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in artifact_files
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--extraction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = analyze_subsets(
        panel_path=args.panel,
        extraction_dir=args.extraction_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
