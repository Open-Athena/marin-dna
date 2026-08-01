"""Compare biological feature yield across the eight issue-426 SAE arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.metrics import average_precision_score

from train import BLOCK_INDICES, BUDGETS, D_SAE, ISSUE, arm_label, assert_commit

TOP_CANDIDATES = 64
MIN_DISCOVERY_SUPPORT = 32
MIN_POSITIVE_SUPPORT = 8
BOOTSTRAPS = 250
RANDOM_SEED = 426
PRIMARY_VIEW = "signed_mean"
VIEWS = (
    "forward_signed",
    "reverse_complement_signed",
    PRIMARY_VIEW,
    "max_abs",
)
REPLICATED_CLASSES = (
    "splice_donor_5th_base_variant",
    "splice_acceptor_variant",
    "splice_donor_variant",
    "splice_polypyrimidine_tract_variant",
    "stop_gained",
)

Matrix = sparse.csr_matrix


@dataclass(frozen=True)
class Activations:
    ref: Matrix
    alt: Matrix
    delta: Matrix


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _sparse_matrix(
    values: np.ndarray,
    panel_rows: np.ndarray,
    feature_ids: np.ndarray,
    shape: tuple[int, int],
) -> Matrix:
    matrix = sparse.csr_matrix((values, (panel_rows, feature_ids)), shape=shape)
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix


def load_activations(path: Path, *, rows: int, columns: int) -> Activations:
    fields = [
        "panel_row",
        "feature_id",
        "ref_activation",
        "alt_activation",
        "delta",
    ]
    table = pq.read_table(path, columns=fields)
    panel_rows = table["panel_row"].to_numpy(zero_copy_only=False).astype(np.int64)
    feature_ids = table["feature_id"].to_numpy(zero_copy_only=False).astype(np.int64)
    ref = table["ref_activation"].to_numpy(zero_copy_only=False).astype(np.float32)
    alt = table["alt_activation"].to_numpy(zero_copy_only=False).astype(np.float32)
    delta = table["delta"].to_numpy(zero_copy_only=False).astype(np.float32)
    assert len(panel_rows) == len(feature_ids) == len(ref) == len(alt) == len(delta)
    assert (
        np.isfinite(ref).all() and np.isfinite(alt).all() and np.isfinite(delta).all()
    )
    assert (ref >= 0).all() and (alt >= 0).all()
    np.testing.assert_array_equal(delta, alt - ref)
    assert (panel_rows >= 0).all() and (panel_rows < rows).all()
    assert (feature_ids >= 0).all() and (feature_ids < columns).all()
    keys = panel_rows * columns + feature_ids
    assert len(np.unique(keys)) == len(keys), "duplicate panel_row/feature_id entries"
    shape = (rows, columns)
    output = Activations(
        ref=_sparse_matrix(ref, panel_rows, feature_ids, shape),
        alt=_sparse_matrix(alt, panel_rows, feature_ids, shape),
        delta=_sparse_matrix(delta, panel_rows, feature_ids, shape),
    )
    difference = output.alt - output.ref - output.delta
    assert difference.nnz == 0 or np.max(np.abs(difference.data)) == 0
    return output


def sparse_abs(matrix: Matrix) -> Matrix:
    output = matrix.copy()
    output.data = np.abs(output.data)
    output.eliminate_zeros()
    return output


def make_views(forward: Matrix, reverse_complement: Matrix) -> dict[str, Matrix]:
    assert forward.shape == reverse_complement.shape
    forward_abs = sparse_abs(forward)
    reverse_abs = sparse_abs(reverse_complement)
    signed_mean = ((forward + reverse_complement) * 0.5).tocsr()
    signed_mean.eliminate_zeros()
    output = {
        "forward_signed": forward,
        "reverse_complement_signed": reverse_complement,
        "signed_mean": signed_mean,
        "max_abs": forward_abs.maximum(reverse_abs).tocsr(),
    }
    assert tuple(output) == VIEWS
    for matrix in output.values():
        assert matrix.shape == forward.shape and np.isfinite(matrix.data).all()
    return output


def column_stats(
    matrix: Matrix, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert indices.ndim == 1 and len(indices) >= 2
    selected = matrix[indices].tocsr()
    means = np.asarray(selected.mean(axis=0)).ravel().astype(np.float64)
    squared = selected.copy()
    squared.data = squared.data.astype(np.float64) ** 2
    variances = np.maximum(np.asarray(squared.mean(axis=0)).ravel() - means**2, 0.0)
    support = np.asarray(selected.getnnz(axis=0)).ravel().astype(np.int64)
    assert np.isfinite(means).all() and np.isfinite(variances).all()
    return means, variances, support


def welch_statistics(
    matrix: Matrix, indices: np.ndarray, positive: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert positive.shape == (len(indices),) and positive.any() and (~positive).any()
    positive_indices = indices[positive]
    negative_indices = indices[~positive]
    positive_mean, positive_var, positive_support = column_stats(
        matrix, positive_indices
    )
    negative_mean, negative_var, _ = column_stats(matrix, negative_indices)
    effect = positive_mean - negative_mean
    standard_error = np.sqrt(
        positive_var / len(positive_indices) + negative_var / len(negative_indices)
    )
    statistic = np.divide(
        effect,
        standard_error,
        out=np.zeros_like(effect),
        where=standard_error > 0,
    )
    _, _, support = column_stats(matrix, indices)
    return effect, statistic, support, positive_support


def column_values(matrix: Matrix, indices: np.ndarray, feature_id: int) -> np.ndarray:
    values = matrix[indices, feature_id]
    output = np.asarray(values.toarray()).reshape(-1).astype(np.float64)
    assert output.shape == (len(indices),) and np.isfinite(output).all()
    return output


def select_feature(
    matrix: Matrix,
    labels: np.ndarray,
    split: np.ndarray,
    class_name: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    discovery = np.flatnonzero(split == "discovery")
    validation = np.flatnonzero(split == "validation")
    test = np.flatnonzero(split == "test")
    discovery_positive = labels[discovery] == class_name
    validation_positive = labels[validation] == class_name
    test_positive = labels[test] == class_name
    assert discovery_positive.sum() == 256
    assert validation_positive.sum() == test_positive.sum() == 128
    discovery_effect, discovery_t, support, positive_support = welch_statistics(
        matrix, discovery, discovery_positive
    )
    eligible = (support >= MIN_DISCOVERY_SUPPORT) & (
        positive_support >= MIN_POSITIVE_SUPPORT
    )
    assert eligible.any(), class_name
    ranking = np.where(eligible, np.abs(discovery_t), -np.inf)
    candidates = np.lexsort((np.arange(matrix.shape[1]), -ranking))[:TOP_CANDIDATES]
    candidates = candidates[np.isfinite(ranking[candidates])]
    assert len(candidates) > 0
    validation_effect, validation_t, _, _ = welch_statistics(
        matrix, validation, validation_positive
    )
    directions = np.where(discovery_effect[candidates] >= 0, 1, -1)
    oriented_validation = directions * validation_t[candidates]
    consistent = oriented_validation > 0
    choices = (
        np.flatnonzero(consistent) if consistent.any() else np.arange(len(candidates))
    )
    selected_local = choices[
        np.lexsort((candidates[choices], -oriented_validation[choices]))[0]
    ]
    feature_id = int(candidates[selected_local])
    direction = int(directions[selected_local])
    validation_scores = direction * column_values(matrix, validation, feature_id)
    test_scores = direction * column_values(matrix, test, feature_id)
    return (
        {
            "class": class_name,
            "feature_id": feature_id,
            "direction": direction,
            "discovery_rank": int(selected_local + 1),
            "discovery_effect": float(discovery_effect[feature_id]),
            "discovery_t": float(discovery_t[feature_id]),
            "discovery_support": int(support[feature_id]),
            "discovery_positive_support": int(positive_support[feature_id]),
            "validation_direction_consistent": bool(consistent[selected_local]),
            "validation_effect": float(validation_effect[feature_id]),
            "validation_t": float(validation_t[feature_id]),
            "validation_oriented_t": float(direction * validation_t[feature_id]),
            "validation_average_precision": float(
                average_precision_score(validation_positive, validation_scores)
            ),
            "test_average_precision": float(
                average_precision_score(test_positive, test_scores)
            ),
            "test_mean_difference": float(
                test_scores[test_positive].mean() - test_scores[~test_positive].mean()
            ),
            "test_support": int(np.count_nonzero(test_scores)),
        },
        test_scores,
        test_positive,
    )


def stratified_block_resample_indices(
    strata: np.ndarray,
    blocks: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resample genomic blocks independently within each fixed label stratum."""
    assert strata.shape == blocks.shape and strata.ndim == 1
    sampled_groups: list[np.ndarray] = []
    for stratum in np.unique(strata):
        stratum_rows = np.flatnonzero(strata == stratum)
        stratum_blocks = np.unique(blocks[stratum_rows])
        assert len(stratum_blocks) > 0
        rows_by_block = [
            stratum_rows[blocks[stratum_rows] == block] for block in stratum_blocks
        ]
        sampled_groups.append(
            np.concatenate(
                [
                    rows_by_block[index]
                    for index in rng.integers(
                        0, len(rows_by_block), size=len(rows_by_block)
                    )
                ]
            )
        )
    indices = np.concatenate(sampled_groups)
    assert len(indices) > 0
    return indices


def bootstrap_block_ap(
    scores: np.ndarray,
    positive: np.ndarray,
    blocks: np.ndarray,
    *,
    seed: int,
    samples: int = BOOTSTRAPS,
) -> tuple[float | None, float | None]:
    assert scores.shape == positive.shape == blocks.shape
    positive_blocks = np.unique(blocks[positive])
    negative_blocks = np.unique(blocks[~positive])
    if min(len(positive_blocks), len(negative_blocks)) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(samples):
        indices = stratified_block_resample_indices(positive, blocks, rng)
        sampled = positive[indices]
        assert sampled.any() and (~sampled).any()
        values.append(float(average_precision_score(sampled, scores[indices])))
    assert len(values) == samples
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def bootstrap_mean_ap(
    scores_by_class: dict[str, tuple[np.ndarray, np.ndarray]],
    blocks: np.ndarray,
    *,
    seed: int,
    samples: int = BOOTSTRAPS,
) -> tuple[float, float]:
    assert scores_by_class
    n_rows = len(blocks)
    assert blocks.shape == (n_rows,)
    class_labels = np.empty(n_rows, dtype=object)
    assignment_counts = np.zeros(n_rows, dtype=np.int16)
    for class_name, (scores, positive) in scores_by_class.items():
        assert scores.shape == positive.shape == (n_rows,)
        assert positive.dtype == np.bool_
        class_labels[positive] = class_name
        assignment_counts += positive
    assert np.all(assignment_counts == 1)
    for class_name in scores_by_class:
        assert len(np.unique(blocks[class_labels == class_name])) >= 2
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(samples):
        indices = stratified_block_resample_indices(class_labels, blocks, rng)
        aps: list[float] = []
        for scores, positive in scores_by_class.values():
            sampled = positive[indices]
            assert sampled.any() and (~sampled).any()
            aps.append(float(average_precision_score(sampled, scores[indices])))
        assert len(aps) == len(scores_by_class)
        values.append(float(np.mean(aps)))
    assert len(values) == samples
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def pair_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    assert first.shape == second.shape and len(first) >= 2
    if np.std(first) == 0 or np.std(second) == 0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def paired_state_rows(
    forward: Activations,
    reverse: Activations,
    selection: dict[str, Any],
    labels: np.ndarray,
    split: np.ndarray,
) -> list[dict[str, Any]]:
    test = np.flatnonzero(split == "test")
    positive = labels[test] == selection["class"]
    feature_id = int(selection["feature_id"])
    rows: list[dict[str, Any]] = []
    for orientation, activations in (
        ("forward", forward),
        ("reverse_complement", reverse),
    ):
        ref = column_values(activations.ref, test, feature_id)
        alt = column_values(activations.alt, test, feature_id)
        delta = alt - ref
        for group, mask in (("class", positive), ("other", ~positive)):
            ref_active = ref[mask] > 0
            alt_active = alt[mask] > 0
            rows.append(
                {
                    "class": selection["class"],
                    "feature_id": feature_id,
                    "orientation": orientation,
                    "group": group,
                    "rows": int(mask.sum()),
                    "ref_active_fraction": float(ref_active.mean()),
                    "alt_active_fraction": float(alt_active.mean()),
                    "both_fraction": float((ref_active & alt_active).mean()),
                    "turn_on_fraction": float((~ref_active & alt_active).mean()),
                    "turn_off_fraction": float((ref_active & ~alt_active).mean()),
                    "mean_delta": float(delta[mask].mean()),
                    "mean_absolute_delta": float(np.abs(delta[mask]).mean()),
                }
            )
    return rows


def plot_yield(summary: pl.DataFrame, output_dir: Path) -> None:
    primary = summary.filter(pl.col("view") == PRIMARY_VIEW).sort(
        ["budget", "reported_block"]
    )
    figure, axis = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    colors = {BUDGETS[0]: "#4C78A8", BUDGETS[1]: "#E45756"}
    for budget in BUDGETS:
        rows = primary.filter(pl.col("budget") == budget).sort("reported_block")
        x = rows["reported_block"].to_numpy()
        y = rows["test_mean_ap"].to_numpy()
        low = rows["test_mean_ap_ci95_low"].to_numpy()
        high = rows["test_mean_ap_ci95_high"].to_numpy()
        axis.errorbar(
            x,
            y,
            yerr=np.vstack((y - low, high - y)),
            marker="o",
            linewidth=2,
            capsize=0,
            color=colors[budget],
            label=f"{budget / 1_000_000:g}M activations",
        )
    axis.axhline(1 / 35, color="black", linestyle="--", linewidth=1, label="chance")
    axis.set_xticks([index + 1 for index in BLOCK_INDICES])
    axis.set_xlabel("Reported transformer block")
    axis.set_ylabel("Held-out mean one-vs-rest AUPRC")
    axis.set_title("Biological feature yield under signed FWD/RC mean")
    axis.legend()
    for suffix in ("png", "svg"):
        figure.savefig(output_dir / f"biological_yield.{suffix}", dpi=180)
    plt.close(figure)


def plot_health_biology(rows: pl.DataFrame, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    colors = {BUDGETS[0]: "#4C78A8", BUDGETS[1]: "#E45756"}
    for budget in BUDGETS:
        selected = rows.filter(pl.col("budget") == budget).sort("reported_block")
        label = f"{budget / 1_000_000:g}M"
        axes[0].scatter(
            selected["explained_variance"],
            selected["test_mean_ap"],
            s=65,
            color=colors[budget],
            label=label,
        )
        axes[1].plot(
            selected["reported_block"],
            selected["heldout_inactive_fraction"],
            marker="o",
            color=colors[budget],
            label=label,
        )
        for row in selected.iter_rows(named=True):
            axes[0].annotate(
                f"B{row['reported_block']}",
                (row["explained_variance"], row["test_mean_ap"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    axes[0].set_xlabel("Held-out SAE explained variance")
    axes[0].set_ylabel("Held-out mean consequence AUPRC")
    axes[0].set_title("Reconstruction health vs biological yield")
    axes[0].legend()
    axes[1].set_xticks([index + 1 for index in BLOCK_INDICES])
    axes[1].set_xlabel("Reported transformer block")
    axes[1].set_ylabel("Held-out inactive SAE fraction")
    axes[1].set_title("Feature inactivity across layers and budgets")
    axes[1].legend()
    for suffix in ("png", "svg"):
        figure.savefig(output_dir / f"health_vs_biology.{suffix}", dpi=180)
    plt.close(figure)


def analyze(
    *,
    extraction_dir: Path,
    extraction_manifest_path: Path,
    training_manifest_path: Path,
    panel_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    assert extraction_dir.is_dir() and extraction_manifest_path.is_file()
    assert training_manifest_path.is_file() and panel_path.is_file()
    assert not output_dir.exists()
    analysis_commit = os.environ.get("ANALYSIS_COMMIT", "")
    assert_commit(analysis_commit)
    extraction_manifest = json.loads(extraction_manifest_path.read_text())
    training_manifest = json.loads(training_manifest_path.read_text())
    assert extraction_manifest["issue"] == training_manifest["issue"] == ISSUE
    assert sha256_file(panel_path) == extraction_manifest["panel"]["sha256"]
    for relative, metadata in extraction_manifest["artifacts"].items():
        path = extraction_dir / relative
        assert path.is_file() and sha256_file(path) == metadata["sha256"]
    training_results_path = training_manifest_path.parent / "results.json"
    assert training_results_path.is_file()
    assert (
        sha256_file(training_results_path)
        == training_manifest["artifacts"]["results.json"]["sha256"]
    )

    panel = pl.read_parquet(panel_path)
    assert panel["panel_row"].to_list() == list(range(panel.height))
    labels = panel["consequence_cre"].to_numpy()
    split = panel["split"].to_numpy()
    blocks = panel["block_id"].to_numpy()
    classes = sorted(np.unique(labels).tolist())
    assert panel.height == 17_920 and len(classes) == 35
    assert set(REPLICATED_CLASSES) <= set(classes)
    assert set(np.unique(split)) == {"discovery", "validation", "test"}
    test = np.flatnonzero(split == "test")
    chance = 1 / len(classes)

    class_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    for block_index in BLOCK_INDICES:
        for budget in BUDGETS:
            label = arm_label(block_index, budget)
            forward = load_activations(
                extraction_dir / label / "sae_activations_forward.parquet",
                rows=panel.height,
                columns=D_SAE,
            )
            reverse = load_activations(
                extraction_dir / label / "sae_activations_reverse_complement.parquet",
                rows=panel.height,
                columns=D_SAE,
            )
            views = make_views(forward.delta, reverse.delta)
            for view_name, matrix in views.items():
                view_rows: list[dict[str, Any]] = []
                scores_by_class: dict[str, tuple[np.ndarray, np.ndarray]] = {}
                for class_name in classes:
                    selection, scores, positive = select_feature(
                        matrix, labels, split, class_name
                    )
                    low, high = bootstrap_block_ap(
                        scores,
                        positive,
                        blocks[test],
                        seed=RANDOM_SEED
                        + block_index * 100_000
                        + budget // 1_000
                        + sum(map(ord, class_name + view_name)),
                    )
                    forward_values = column_values(
                        forward.delta, test, int(selection["feature_id"])
                    )
                    reverse_values = column_values(
                        reverse.delta, test, int(selection["feature_id"])
                    )
                    row = {
                        "arm": label,
                        "reported_block": block_index + 1,
                        "block_index": block_index,
                        "budget": budget,
                        "view": view_name,
                        "is_current_replicated_class": class_name in REPLICATED_CLASSES,
                        **selection,
                        "test_ap_ci95_low": low,
                        "test_ap_ci95_high": high,
                        "ci95_above_chance": low is not None and low > chance,
                        "same_id_fwd_rc_test_pearson": pair_correlation(
                            forward_values, reverse_values
                        ),
                    }
                    class_rows.append(row)
                    view_rows.append(row)
                    scores_by_class[class_name] = (scores, positive)
                    if view_name == PRIMARY_VIEW:
                        for state in paired_state_rows(
                            forward, reverse, selection, labels, split
                        ):
                            state_rows.append(
                                {
                                    "arm": label,
                                    "reported_block": block_index + 1,
                                    "budget": budget,
                                    **state,
                                }
                            )
                values = np.asarray(
                    [row["test_average_precision"] for row in view_rows]
                )
                replicated_values = np.asarray(
                    [
                        row["test_average_precision"]
                        for row in view_rows
                        if row["is_current_replicated_class"]
                    ]
                )
                assert replicated_values.shape == (len(REPLICATED_CLASSES),)
                aggregate_low, aggregate_high = bootstrap_mean_ap(
                    scores_by_class,
                    blocks[test],
                    seed=RANDOM_SEED
                    + block_index * 100_000
                    + budget // 1_000
                    + sum(map(ord, view_name)),
                )
                summary_rows.append(
                    {
                        "arm": label,
                        "reported_block": block_index + 1,
                        "block_index": block_index,
                        "budget": budget,
                        "view": view_name,
                        "test_mean_ap": float(values.mean()),
                        "test_median_ap": float(np.median(values)),
                        "replicated_classes_test_mean_ap": float(
                            replicated_values.mean()
                        ),
                        "test_mean_ap_ci95_low": aggregate_low,
                        "test_mean_ap_ci95_high": aggregate_high,
                        "classes_ci95_above_chance": sum(
                            bool(row["ci95_above_chance"]) for row in view_rows
                        ),
                        "classes_validation_direction_consistent": sum(
                            bool(row["validation_direction_consistent"])
                            for row in view_rows
                        ),
                        "classes": len(view_rows),
                    }
                )
                print(
                    json.dumps(
                        {
                            "stage": "arm_view",
                            "arm": label,
                            "view": view_name,
                            "test_mean_ap": float(values.mean()),
                        }
                    ),
                    flush=True,
                )

    class_metrics = pl.DataFrame(class_rows).sort(["arm", "view", "class"])
    replicated_metrics = class_metrics.filter(
        pl.col("is_current_replicated_class")
    ).sort(["arm", "view", "class"])
    assert replicated_metrics.height == (
        len(BLOCK_INDICES) * len(BUDGETS) * len(VIEWS) * len(REPLICATED_CLASSES)
    )
    arm_summary = pl.DataFrame(summary_rows).sort(["view", "budget", "reported_block"])
    states = pl.DataFrame(state_rows).sort(["arm", "class", "orientation", "group"])
    primary_classes = class_metrics.filter(pl.col("view") == PRIMARY_VIEW)
    short = primary_classes.filter(pl.col("budget") == BUDGETS[0]).select(
        "reported_block",
        "class",
        pl.col("feature_id").alias("short_feature_id"),
        pl.col("test_average_precision").alias("short_test_ap"),
    )
    long = primary_classes.filter(pl.col("budget") == BUDGETS[1]).select(
        "reported_block",
        "class",
        pl.col("feature_id").alias("long_feature_id"),
        pl.col("test_average_precision").alias("long_test_ap"),
    )
    budget_differences = (
        short.join(long, on=["reported_block", "class"], how="inner", validate="1:1")
        .with_columns(
            (pl.col("long_feature_id") == pl.col("short_feature_id")).alias(
                "same_feature_id"
            ),
            (pl.col("long_test_ap") - pl.col("short_test_ap")).alias("test_ap_delta"),
        )
        .sort(["reported_block", "class"])
    )
    assert budget_differences.height == len(BLOCK_INDICES) * len(classes)

    primary_summary = arm_summary.filter(pl.col("view") == PRIMARY_VIEW)
    health_rows: list[dict[str, Any]] = []
    for row in primary_summary.iter_rows(named=True):
        health = training_manifest["health"][row["arm"]]
        health_rows.append({**row, **health})
    health_biology = pl.DataFrame(health_rows).sort(["budget", "reported_block"])

    output_dir.mkdir(parents=True)
    class_metrics.write_parquet(output_dir / "class_metrics.parquet")
    replicated_metrics.write_parquet(output_dir / "replicated_class_metrics.parquet")
    arm_summary.write_parquet(output_dir / "arm_summary.parquet")
    budget_differences.write_parquet(output_dir / "budget_differences.parquet")
    states.write_parquet(output_dir / "paired_state_summary.parquet")
    health_biology.write_parquet(output_dir / "health_biology.parquet")
    plot_yield(arm_summary, output_dir)
    plot_health_biology(health_biology, output_dir)

    best = primary_summary.sort(
        ["test_mean_ap", "reported_block", "budget"],
        descending=[True, False, False],
    ).row(0, named=True)
    result = {
        "analysis_commit": analysis_commit,
        "issue": ISSUE,
        "extraction_manifest_sha256": sha256_file(extraction_manifest_path),
        "training_manifest_sha256": sha256_file(training_manifest_path),
        "panel_sha256": sha256_file(panel_path),
        "rows": panel.height,
        "classes": len(classes),
        "current_replicated_classes": list(REPLICATED_CLASSES),
        "chance_one_vs_rest_ap": chance,
        "primary_endpoint": "mean held-out one-vs-rest AP across 35 consequence_cre classes under signed mean of paired FWD/RC deltas",
        "best_primary_arm": best,
        "protocol": {
            "primary_view": PRIMARY_VIEW,
            "diagnostic_views": [view for view in VIEWS if view != PRIMARY_VIEW],
            "top_candidates": TOP_CANDIDATES,
            "minimum_discovery_support": MIN_DISCOVERY_SUPPORT,
            "minimum_positive_support": MIN_POSITIVE_SUPPORT,
            "block_bootstraps": BOOTSTRAPS,
            "bootstrap_scheme": "consequence-stratified genomic-block resampling; binary label strata for per-class CIs",
            "random_seed": RANDOM_SEED,
            "feature_selection": "absolute discovery Welch t among eligible features; sign fixed on discovery; validation chooses among top 64; test untouched",
        },
    }
    write_json(output_dir / "results.json", result)
    artifacts = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    manifest = {**result, "artifacts": artifacts}
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction-dir", type=Path, required=True)
    parser.add_argument("--extraction-manifest", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = analyze(
        extraction_dir=args.extraction_dir,
        extraction_manifest_path=args.extraction_manifest,
        training_manifest_path=args.training_manifest,
        panel_path=args.panel,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
