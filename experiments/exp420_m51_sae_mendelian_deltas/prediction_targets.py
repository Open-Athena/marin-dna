"""Evaluate the three primary Mendelian SAE prediction targets.

The targets are: pathogenic ``label`` pooled across all subsets, pathogenic
``label`` within each subset (read from the already-frozen analysis), and
consequence ``subset`` pooled across both labels. Feature selection uses only
discovery and validation chromosomes; chr11/X are evaluated once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import matplotlib
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score

from prediction_primitives import (
    D_SAE,
    ISSUE,
    ORIENTATIONS,
    TRANSFORMS,
    FeatureSelection,
    bootstrap_mean_interval,
    load_dense_delta,
    matched_contrasts,
    matched_permutation_pvalue,
    select_feature,
    sha256,
    standardize_pretest,
    standardized_means,
    validate_panel,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MIN_GROUPS_PER_SPLIT = 10
MIN_NONZERO_GROUPS = 8
TOP_DISCOVERY_CANDIDATES = 32
BOOTSTRAPS = 2_000
SEED = 420_4


def _seed(*parts: Any) -> int:
    value = "|".join(str(part) for part in (SEED, *parts))
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "little")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


@dataclass(frozen=True)
class MatchedSelection:
    feature_id: int
    transform: Literal["signed", "absolute"]
    direction: int
    discovery_mean: float
    discovery_t: float
    validation_mean: float
    validation_t: float
    validation_direction_consistent: bool
    discovery_nonzero_groups: int
    validation_nonzero_groups: int
    scores: np.ndarray

    def metadata(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "transform": self.transform,
            "direction": self.direction,
            "discovery_mean": self.discovery_mean,
            "discovery_t": self.discovery_t,
            "validation_mean": self.validation_mean,
            "validation_t": self.validation_t,
            "validation_direction_consistent": self.validation_direction_consistent,
            "discovery_nonzero_groups": self.discovery_nonzero_groups,
            "validation_nonzero_groups": self.validation_nonzero_groups,
        }


def _transformed(matrix: np.ndarray, transform: str) -> np.ndarray:
    assert transform in TRANSFORMS
    return matrix if transform == "signed" else np.abs(matrix)


def select_matched_label_feature(
    matrix: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    splits: np.ndarray,
    *,
    top_k: int = TOP_DISCOVERY_CANDIDATES,
    min_nonzero_groups: int = MIN_NONZERO_GROUPS,
) -> tuple[MatchedSelection, list[dict[str, Any]]]:
    """Select a pooled label feature using matched group contrasts."""

    assert matrix.ndim == 2
    assert labels.shape == groups.shape == splits.shape == (matrix.shape[0],)
    assert set(np.unique(labels)) == {0, 1}
    discovery_rows = splits == "discovery"
    validation_rows = splits == "validation"
    candidates: list[dict[str, Any]] = []
    for transform in TRANSFORMS:
        values = _transformed(matrix, transform)
        discovery, _ = matched_contrasts(
            values[discovery_rows], labels[discovery_rows], groups[discovery_rows]
        )
        validation, _ = matched_contrasts(
            values[validation_rows], labels[validation_rows], groups[validation_rows]
        )
        discovery_mean, discovery_t = standardized_means(discovery)
        discovery_support = np.count_nonzero(discovery, axis=0)
        eligible = discovery_support >= min_nonzero_groups
        score = np.where(eligible, np.abs(discovery_t), -np.inf)
        assert np.isfinite(score).any()
        keep = min(top_k, int(eligible.sum()))
        ranked = np.argsort(-score, kind="stable")[:keep]
        validation_mean, validation_t = standardized_means(validation[:, ranked])
        validation_support = np.count_nonzero(validation[:, ranked], axis=0)
        for rank, feature_id in enumerate(ranked, start=1):
            direction = 1 if discovery_mean[feature_id] >= 0 else -1
            validation_consistent = bool(
                validation_support[rank - 1] >= min_nonzero_groups
                and direction * validation_t[rank - 1] > 0
            )
            candidates.append(
                {
                    "transform": transform,
                    "feature_id": int(feature_id),
                    "discovery_rank_within_transform": rank,
                    "direction": direction,
                    "discovery_mean": float(discovery_mean[feature_id]),
                    "discovery_t": float(discovery_t[feature_id]),
                    "validation_mean": float(validation_mean[rank - 1]),
                    "validation_t": float(validation_t[rank - 1]),
                    "validation_direction_consistent": validation_consistent,
                    "discovery_nonzero_groups": int(discovery_support[feature_id]),
                    "validation_nonzero_groups": int(validation_support[rank - 1]),
                }
            )
    consistent = [row for row in candidates if row["validation_direction_consistent"]]
    ranked_candidates = sorted(
        consistent or candidates,
        key=lambda row: (
            -(row["direction"] * row["validation_t"]),
            -abs(row["discovery_t"]),
            row["transform"],
            row["feature_id"],
        ),
    )
    chosen = ranked_candidates[0]
    scores = _transformed(matrix[:, chosen["feature_id"]], chosen["transform"])
    selection = MatchedSelection(
        feature_id=chosen["feature_id"],
        transform=chosen["transform"],
        direction=chosen["direction"],
        discovery_mean=chosen["discovery_mean"],
        discovery_t=chosen["discovery_t"],
        validation_mean=chosen["validation_mean"],
        validation_t=chosen["validation_t"],
        validation_direction_consistent=chosen["validation_direction_consistent"],
        discovery_nonzero_groups=chosen["discovery_nonzero_groups"],
        validation_nonzero_groups=chosen["validation_nonzero_groups"],
        scores=np.asarray(scores, dtype=np.float32),
    )
    return selection, candidates


def bootstrap_label_auprc(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    samples: int = BOOTSTRAPS,
) -> dict[str, float | int]:
    """Cluster-bootstrap row AUPRC by resampling intact 1:9 match groups."""

    assert labels.shape == scores.shape == groups.shape
    group_order = np.unique(groups)
    group_to_rows = {group: np.flatnonzero(groups == group) for group in group_order}
    for rows in group_to_rows.values():
        assert len(rows) == 10 and labels[rows].sum() == 1
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        sampled = rng.choice(group_order, len(group_order), replace=True)
        rows = np.concatenate([group_to_rows[group] for group in sampled])
        values[sample] = average_precision_score(labels[rows], scores[rows])
    return {
        "auprc": float(average_precision_score(labels, scores)),
        "auprc_ci95_low": float(np.quantile(values, 0.025)),
        "auprc_ci95_high": float(np.quantile(values, 0.975)),
        "prevalence": float(labels.mean()),
        "test_rows": len(labels),
        "test_groups": len(group_order),
    }


def _label_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    metrics = bootstrap_label_auprc(labels, scores, groups, seed=seed)
    contrasts, _ = matched_contrasts(scores, labels, groups)
    low, high = bootstrap_mean_interval(contrasts, seed=seed + 1)
    return {
        **metrics,
        "matched_mean": float(contrasts.mean()),
        "matched_ci95_low": low,
        "matched_ci95_high": high,
        "permutation_pvalue": matched_permutation_pvalue(
            scores, labels, groups, seed=seed + 2
        ),
    }


def eligible_subset_classes(
    panel: pl.DataFrame,
) -> tuple[pl.DataFrame, tuple[str, ...]]:
    support = (
        panel.group_by("subset", "split")
        .agg(pl.col("match_group").n_unique().alias("groups"))
        .sort("subset", "split")
    )
    minima = support.group_by("subset").agg(
        pl.col("split").n_unique().alias("split_count"),
        pl.col("groups").min().alias("minimum_groups"),
    )
    eligible = tuple(
        sorted(
            minima.filter(
                (pl.col("split_count") == 3)
                & (pl.col("minimum_groups") >= MIN_GROUPS_PER_SPLIT)
            )["subset"].to_list()
        )
    )
    assert len(eligible) >= 2
    return support, eligible


def multiclass_auprc(
    truth: np.ndarray,
    scores: np.ndarray,
    classes: tuple[str, ...],
) -> tuple[dict[str, float], float]:
    assert scores.shape == (len(truth), len(classes))
    per_class = {
        target_class: float(
            average_precision_score(truth == target_class, scores[:, class_index])
        )
        for class_index, target_class in enumerate(classes)
    }
    return per_class, float(np.mean(list(per_class.values())))


def bootstrap_subset_auprc(
    truth: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    classes: tuple[str, ...],
    *,
    seed: int,
    samples: int = BOOTSTRAPS,
) -> dict[str, Any]:
    """Stratified group bootstrap for one-vs-rest and macro AUPRC."""

    assert truth.shape == groups.shape == (scores.shape[0],)
    group_to_rows = {
        group: np.flatnonzero(groups == group) for group in np.unique(groups)
    }
    groups_by_class: dict[str, np.ndarray] = {}
    for target_class in classes:
        class_groups = np.unique(groups[truth == target_class])
        assert len(class_groups) >= MIN_GROUPS_PER_SPLIT
        assert all(
            (truth[group_to_rows[group]] == target_class).all()
            for group in class_groups
        )
        groups_by_class[target_class] = class_groups
    rng = np.random.default_rng(seed)
    boot = np.empty((samples, len(classes) + 1), dtype=np.float64)
    for sample in range(samples):
        sampled = np.concatenate(
            [
                rng.choice(class_groups, len(class_groups), replace=True)
                for class_groups in groups_by_class.values()
            ]
        )
        rows = np.concatenate([group_to_rows[group] for group in sampled])
        per_class, macro = multiclass_auprc(truth[rows], scores[rows], classes)
        boot[sample, :-1] = [per_class[target_class] for target_class in classes]
        boot[sample, -1] = macro
    per_class, macro = multiclass_auprc(truth, scores, classes)
    per_class_output = {
        target_class: {
            "auprc": per_class[target_class],
            "auprc_ci95_low": float(np.quantile(boot[:, index], 0.025)),
            "auprc_ci95_high": float(np.quantile(boot[:, index], 0.975)),
            "prevalence": float(np.mean(truth == target_class)),
        }
        for index, target_class in enumerate(classes)
    }
    return {
        "macro_auprc": macro,
        "macro_auprc_ci95_low": float(np.quantile(boot[:, -1], 0.025)),
        "macro_auprc_ci95_high": float(np.quantile(boot[:, -1], 0.975)),
        "macro_random_baseline": 1 / len(classes),
        "per_class": per_class_output,
        "test_rows": len(truth),
        "test_groups": len(np.unique(groups)),
    }


def _plot(
    label_summary: pl.DataFrame,
    within_subset: pl.DataFrame,
    subset_summary: pl.DataFrame,
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    views = ["forward", "reverse_complement", "aggregate"]
    labels = ["FWD", "RC", "aggregate"]
    pooled = label_summary.sort(
        pl.col("view").replace_strict(views, list(range(len(views))))
    )
    axes[0].errorbar(
        np.arange(3),
        pooled["auprc"],
        yerr=np.vstack(
            (
                pooled["auprc"] - pooled["auprc_ci95_low"],
                pooled["auprc_ci95_high"] - pooled["auprc"],
            )
        ),
        marker="o",
        capsize=0,
        color="#0072B2",
    )
    axes[0].axhline(0.1, color="grey", linewidth=0.8, linestyle="--")
    axes[0].set_xticks(np.arange(3), labels)
    axes[0].set_title("Predict label overall")
    axes[0].set_ylabel("Held-out AUPRC")

    within = within_subset.filter(
        (pl.col("space") == "sae") & (pl.col("component") == "equal_weight_mean")
    ).sort("subset")
    axes[1].scatter(
        within["test_average_precision"], np.arange(within.height), color="#009E73"
    )
    axes[1].axvline(0.1, color="grey", linewidth=0.8, linestyle="--")
    axes[1].set_yticks(np.arange(within.height), within["subset"])
    axes[1].set_xlabel("Held-out AUPRC")
    axes[1].set_title("Predict label within subset")

    subset = subset_summary.sort(
        pl.col("view").replace_strict(views, list(range(len(views))))
    )
    axes[2].errorbar(
        np.arange(3),
        subset["macro_auprc"],
        yerr=np.vstack(
            (
                subset["macro_auprc"] - subset["macro_auprc_ci95_low"],
                subset["macro_auprc_ci95_high"] - subset["macro_auprc"],
            )
        ),
        marker="o",
        capsize=0,
        color="#D55E00",
    )
    axes[2].axhline(
        subset["macro_random_baseline"][0],
        color="grey",
        linewidth=0.8,
        linestyle="--",
    )
    axes[2].set_xticks(np.arange(3), labels)
    axes[2].set_title("Predict subset (pooled labels)")
    axes[2].set_ylabel("Held-out macro-AUPRC")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle("exp420: primary SAE variant-delta prediction targets")
    figure.savefig(output_dir / "prediction_targets.png", dpi=180)
    figure.savefig(output_dir / "prediction_targets.svg")
    plt.close(figure)


def _markdown(
    label_summary: pl.DataFrame,
    within_subset: pl.DataFrame,
    subset_summary: pl.DataFrame,
    subset_per_class: pl.DataFrame,
    classes: tuple[str, ...],
) -> str:
    lines = [
        "# exp420 primary prediction targets",
        "",
        "All scores are fixed by discovery/validation chromosomes and evaluated once on chr11/X. `label=1` prevalence is exactly 10% within every match group, subset, and pooled dataset, so chance label AUPRC is 0.10.",
        "",
        "## Predict label overall (no subset stratification)",
        "",
        "| view | feature | transform | gate | AUPRC (95% group-bootstrap CI) | matched effect (95% CI) | p |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    for row in label_summary.sort("view").to_dicts():
        feature = (
            "two selected IDs" if row["view"] == "aggregate" else row["feature_id"]
        )
        transform = "score mean" if row["view"] == "aggregate" else row["transform"]
        lines.append(
            f"| {row['view']} | {feature} | {transform} | {row['validation_direction_consistent']} | "
            f"{row['auprc']:.4f} [{row['auprc_ci95_low']:.4f}, {row['auprc_ci95_high']:.4f}] | "
            f"{row['matched_mean']:.4f} [{row['matched_ci95_low']:.4f}, {row['matched_ci95_high']:.4f}] | "
            f"{row['permutation_pvalue']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Predict label within each subset",
            "",
            "These are the previously frozen equal-weight SAE aggregate results; no feature was reselected.",
            "",
            "| subset | AUPRC | matched effect (95% CI) | p |",
            "|---|---:|---:|---:|",
        ]
    )
    within = within_subset.filter(
        (pl.col("space") == "sae") & (pl.col("component") == "equal_weight_mean")
    ).sort("subset")
    for row in within.to_dicts():
        lines.append(
            f"| {row['subset']} | {row['test_average_precision']:.4f} | "
            f"{row['test_matched_mean']:.4f} [{row['test_matched_ci95_low']:.4f}, "
            f"{row['test_matched_ci95_high']:.4f}] | {row['test_permutation_pvalue']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Predict subset (both labels pooled)",
            "",
            f"Classes with at least {MIN_GROUPS_PER_SPLIT} match groups in every split: "
            + ", ".join(f"`{target_class}`" for target_class in classes)
            + ".",
            "",
            "| view | macro-AUPRC (95% group-bootstrap CI) | random baseline | all gates |",
            "|---|---:|---:|---|",
        ]
    )
    for row in subset_summary.sort("view").to_dicts():
        lines.append(
            f"| {row['view']} | {row['macro_auprc']:.4f} "
            f"[{row['macro_auprc_ci95_low']:.4f}, {row['macro_auprc_ci95_high']:.4f}] | "
            f"{row['macro_random_baseline']:.4f} | {row['all_validation_gates']} |"
        )
    lines.extend(
        [
            "",
            "Per-class aggregate one-vs-rest results:",
            "",
            "| subset | AUPRC (95% group-bootstrap CI) | prevalence |",
            "|---|---:|---:|",
        ]
    )
    for row in (
        subset_per_class.filter(pl.col("view") == "aggregate")
        .sort("target_class")
        .to_dicts()
    ):
        lines.append(
            f"| {row['target_class']} | {row['auprc']:.4f} "
            f"[{row['auprc_ci95_low']:.4f}, {row['auprc_ci95_high']:.4f}] | "
            f"{row['prevalence']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def analyze_prediction_targets(
    *,
    panel_path: Path,
    extraction_dir: Path,
    within_subset_summary_path: Path,
    within_subset_manifest_path: Path,
    output_dir: Path,
    extraction_commit: str,
    analysis_commit: str,
) -> dict[str, Any]:
    assert panel_path.is_file() and extraction_dir.is_dir()
    assert (
        within_subset_summary_path.is_file() and within_subset_manifest_path.is_file()
    )
    assert not output_dir.exists()
    assert len(extraction_commit) == len(analysis_commit) == 40
    extraction_manifest_path = extraction_dir / "manifest.json"
    extraction_manifest = json.loads(extraction_manifest_path.read_text())
    assert extraction_manifest["experiment_commit"] == extraction_commit
    assert extraction_manifest["panel"]["sha256"] == sha256(panel_path)
    for name, metadata in extraction_manifest["artifacts"].items():
        path = extraction_dir / name
        assert path.stat().st_size == metadata["bytes"]
        assert sha256(path) == metadata["sha256"]
    within_manifest = json.loads(within_subset_manifest_path.read_text())
    within_name = within_subset_summary_path.name
    within_metadata = within_manifest["artifacts"][within_name]
    assert within_subset_summary_path.stat().st_size == within_metadata["bytes"]
    assert sha256(within_subset_summary_path) == within_metadata["sha256"]

    panel = pl.read_parquet(panel_path)
    validate_panel(panel)
    within_subset = pl.read_parquet(within_subset_summary_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    labels = panel["label"].to_numpy().astype(np.int8)
    groups = panel["match_group"].to_numpy()
    splits = panel["split"].to_numpy()
    subsets = panel["subset"].to_numpy()
    assert np.isclose(labels.mean(), 0.1)
    support, classes = eligible_subset_classes(panel)

    label_selections: dict[str, MatchedSelection] = {}
    subset_selections: dict[tuple[str, str], FeatureSelection] = {}
    candidate_rows: list[dict[str, Any]] = []
    task_rows = np.isin(subsets, classes)
    for orientation in ORIENTATIONS:
        matrix = load_dense_delta(
            extraction_dir / f"sae_activations_{orientation}.parquet",
            rows=panel.height,
            features=D_SAE,
        )
        label_selection, candidates = select_matched_label_feature(
            matrix, labels, groups, splits
        )
        label_selections[orientation] = label_selection
        candidate_rows.extend(
            {
                "analysis": "label_overall",
                "orientation": orientation,
                "target_class": "label=1",
                **candidate,
            }
            for candidate in candidates
        )
        for target_class in classes:
            target = subsets == target_class
            selection, candidates = select_feature(
                matrix,
                target,
                splits,
                task_rows,
                top_k=TOP_DISCOVERY_CANDIDATES,
                min_nonzero=MIN_NONZERO_GROUPS,
            )
            subset_selections[(orientation, target_class)] = selection
            candidate_rows.extend(
                {
                    "analysis": "subset_pooled_one_vs_rest",
                    "orientation": orientation,
                    "target_class": target_class,
                    **candidate,
                }
                for candidate in candidates
            )
        del matrix

    non_test = splits != "test"
    test = splits == "test"
    label_scores: dict[str, np.ndarray] = {}
    label_rows: list[dict[str, Any]] = []
    for orientation in ORIENTATIONS:
        selection = label_selections[orientation]
        feature_selection = FeatureSelection(
            feature_id=selection.feature_id,
            transform=selection.transform,
            direction=selection.direction,
            discovery_effect=selection.discovery_mean,
            validation_effect=selection.validation_mean,
            validation_direction_consistent=selection.validation_direction_consistent,
            discovery_positive_support=selection.discovery_nonzero_groups,
            discovery_negative_support=selection.discovery_nonzero_groups,
            validation_positive_support=selection.validation_nonzero_groups,
            validation_negative_support=selection.validation_nonzero_groups,
            scores=selection.scores,
        )
        standardized, center, scale = standardize_pretest(feature_selection, non_test)
        label_scores[orientation] = standardized
        label_rows.append(
            {
                "view": orientation,
                **selection.metadata(),
                "pretest_center": center,
                "pretest_scale": scale,
                **_label_metrics(
                    labels[test],
                    standardized[test],
                    groups[test],
                    seed=_seed("label", orientation),
                ),
            }
        )
    label_scores["aggregate"] = (
        label_scores["forward"] + label_scores["reverse_complement"]
    ) / 2
    label_rows.append(
        {
            "view": "aggregate",
            "feature_id": None,
            "transform": None,
            "direction": None,
            "discovery_mean": None,
            "discovery_t": None,
            "validation_mean": None,
            "validation_t": None,
            "validation_direction_consistent": all(
                selection.validation_direction_consistent
                for selection in label_selections.values()
            ),
            "discovery_nonzero_groups": None,
            "validation_nonzero_groups": None,
            "pretest_center": None,
            "pretest_scale": None,
            **_label_metrics(
                labels[test],
                label_scores["aggregate"][test],
                groups[test],
                seed=_seed("label", "aggregate"),
            ),
        }
    )

    subset_task_test = task_rows & test
    truth = subsets[subset_task_test]
    subset_score_by_view: dict[str, np.ndarray] = {}
    subset_selection_rows: list[dict[str, Any]] = []
    for orientation in ORIENTATIONS:
        columns: list[np.ndarray] = []
        for target_class in classes:
            selection = subset_selections[(orientation, target_class)]
            standardized, center, scale = standardize_pretest(
                selection, task_rows & non_test
            )
            columns.append(standardized[subset_task_test])
            subset_selection_rows.append(
                {
                    "orientation": orientation,
                    "target_class": target_class,
                    **selection.metadata(),
                    "pretest_center": center,
                    "pretest_scale": scale,
                }
            )
        subset_score_by_view[orientation] = np.column_stack(columns)
    subset_score_by_view["aggregate"] = (
        subset_score_by_view["forward"] + subset_score_by_view["reverse_complement"]
    ) / 2

    subset_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    for view, score_matrix in subset_score_by_view.items():
        metrics = bootstrap_subset_auprc(
            truth,
            score_matrix,
            groups[subset_task_test],
            classes,
            seed=_seed("subset", view),
        )
        per_class = metrics.pop("per_class")
        if view == "aggregate":
            gate = all(
                selection.validation_direction_consistent
                for selection in subset_selections.values()
            )
        else:
            gate = all(
                subset_selections[(view, target_class)].validation_direction_consistent
                for target_class in classes
            )
        subset_rows.append({"view": view, **metrics, "all_validation_gates": gate})
        for target_class, class_metrics in per_class.items():
            per_class_rows.append(
                {"view": view, "target_class": target_class, **class_metrics}
            )

    label_summary = pl.DataFrame(label_rows)
    subset_summary = pl.DataFrame(subset_rows)
    subset_per_class = pl.DataFrame(per_class_rows)
    output_tables = {
        "label_overall": label_summary,
        "label_within_subset": within_subset,
        "subset_pooled": subset_summary,
        "subset_pooled_per_class": subset_per_class,
        "subset_selections": pl.DataFrame(subset_selection_rows),
        "candidates": pl.DataFrame(candidate_rows),
        "class_support": support,
    }
    for name, table in output_tables.items():
        assert table.height > 0
        table.write_parquet(output_dir / f"{name}.parquet", compression="zstd")
    _plot(label_summary, within_subset, subset_summary, output_dir)
    (output_dir / "RESULTS.md").write_text(
        _markdown(
            label_summary,
            within_subset,
            subset_summary,
            subset_per_class,
            classes,
        )
    )

    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "issue": ISSUE,
        "analysis_commit": analysis_commit,
        "extraction_commit": extraction_commit,
        "panel": {"rows": panel.height, "sha256": sha256(panel_path)},
        "inputs": {
            "extraction_manifest": {
                "bytes": extraction_manifest_path.stat().st_size,
                "sha256": sha256(extraction_manifest_path),
            },
            "within_subset_manifest": {
                "bytes": within_subset_manifest_path.stat().st_size,
                "sha256": sha256(within_subset_manifest_path),
            },
            "within_subset_summary": {
                "bytes": within_subset_summary_path.stat().st_size,
                "sha256": sha256(within_subset_summary_path),
            },
        },
        "protocol": {
            "targets": [
                "label pooled across all subsets without stratification",
                "label within each subset (previously frozen)",
                "subset pooled across both labels",
            ],
            "primary_metric": "row-level average precision / AUPRC",
            "label_positive_class": 1,
            "label_prevalence": 0.1,
            "subset_classes": list(classes),
            "minimum_groups_per_split": MIN_GROUPS_PER_SPLIT,
            "feature_transforms": list(TRANSFORMS),
            "top_discovery_candidates_per_transform": TOP_DISCOVERY_CANDIDATES,
            "minimum_nonzero_label_group_contrasts": MIN_NONZERO_GROUPS,
            "minimum_nonzero_subset_rows_per_class": MIN_NONZERO_GROUPS,
            "bootstraps": BOOTSTRAPS,
            "orientation_aggregate": "equal mean of separately selected, direction-oriented, globally pretest-standardized scores",
            "test_split": "chr11/X",
        },
    }
    _write_json(output_dir / "results.json", result)
    files = sorted(path for path in output_dir.iterdir() if path.is_file())
    manifest = {
        **result,
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in files
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--extraction-dir", type=Path, required=True)
    parser.add_argument("--within-subset-summary", type=Path, required=True)
    parser.add_argument("--within-subset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--extraction-commit", required=True)
    parser.add_argument("--analysis-commit", required=True)
    args = parser.parse_args()
    result = analyze_prediction_targets(
        panel_path=args.panel,
        extraction_dir=args.extraction_dir,
        within_subset_summary_path=args.within_subset_summary,
        within_subset_manifest_path=args.within_subset_manifest,
        output_dir=args.output_dir,
        extraction_commit=args.extraction_commit,
        analysis_commit=args.analysis_commit,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
