"""Analyze paired ref/alt states and strand behavior for issue #424."""

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

TOP_CANDIDATES = 64
MIN_DISCOVERY_SUPPORT = 32
MIN_POSITIVE_SUPPORT = 8
BOOTSTRAPS = 250
RANDOM_SEED = 424
ORIENTATIONS = ("forward", "reverse_complement")
AGGREGATE_VIEWS = (
    "signed_mean",
    "abs_signed_mean",
    "mean_abs",
    "rms",
    "max_abs",
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
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


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
    assert np.array_equal(delta, alt - ref), (
        "delta must equal alt_activation - ref_activation"
    )
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
    mean_abs = ((forward_abs + reverse_abs) * 0.5).tocsr()
    rms = forward.multiply(forward) + reverse_complement.multiply(reverse_complement)
    rms = (rms * 0.5).tocsr()
    rms.data = np.sqrt(rms.data)
    max_abs = forward_abs.maximum(reverse_abs).tocsr()
    output = {
        "forward_signed": forward,
        "forward_abs": forward_abs,
        "reverse_complement_signed": reverse_complement,
        "reverse_complement_abs": reverse_abs,
        "signed_mean": signed_mean,
        "abs_signed_mean": sparse_abs(signed_mean),
        "mean_abs": mean_abs,
        "rms": rms,
        "max_abs": max_abs,
    }
    for matrix in output.values():
        assert matrix.shape == forward.shape
        assert np.isfinite(matrix.data).all()
    return output


def binary_matrix(matrix: Matrix) -> Matrix:
    output = matrix.copy()
    output.data = np.ones_like(output.data, dtype=np.uint8)
    output.eliminate_zeros()
    return output


def paired_states(activations: Activations) -> dict[str, Matrix]:
    ref = binary_matrix(activations.ref)
    alt = binary_matrix(activations.alt)
    both = ref.multiply(alt).tocsr()
    turn_on = (alt - both).tocsr()
    turn_off = (ref - both).tocsr()
    for matrix in (both, turn_on, turn_off):
        matrix.eliminate_zeros()
        assert matrix.nnz == 0 or np.array_equal(
            np.unique(matrix.data), np.array([1], dtype=matrix.data.dtype)
        )
    return {"both": both, "turn_on": turn_on, "turn_off": turn_off}


def _column_sums(matrix: Matrix) -> tuple[np.ndarray, np.ndarray]:
    sums = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float64)
    squared = matrix.copy()
    squared.data = squared.data.astype(np.float64) ** 2
    sum_squares = np.asarray(squared.sum(axis=0)).ravel()
    return sums, sum_squares


def same_id_correlations(
    first: Matrix, second: Matrix, *, rows: int
) -> tuple[np.ndarray, np.ndarray]:
    assert first.shape == second.shape and first.shape[0] == rows and rows >= 2
    first_sum, first_square = _column_sums(first)
    second_sum, second_square = _column_sums(second)
    products = np.asarray(first.multiply(second).sum(axis=0)).ravel().astype(np.float64)
    first_centered = np.maximum(first_square - first_sum**2 / rows, 0)
    second_centered = np.maximum(second_square - second_sum**2 / rows, 0)
    pearson_denominator = np.sqrt(first_centered * second_centered)
    pearson = np.divide(
        products - first_sum * second_sum / rows,
        pearson_denominator,
        out=np.full(first.shape[1], np.nan),
        where=pearson_denominator > 0,
    )
    cosine_denominator = np.sqrt(first_square * second_square)
    cosine = np.divide(
        products,
        cosine_denominator,
        out=np.full(first.shape[1], np.nan),
        where=cosine_denominator > 0,
    )
    return pearson, cosine


def column_jaccard(first: Matrix, second: Matrix) -> tuple[np.ndarray, np.ndarray]:
    first_binary = binary_matrix(first)
    second_binary = binary_matrix(second)
    intersection = np.asarray(first_binary.multiply(second_binary).sum(axis=0)).ravel()
    union = (
        np.asarray(first_binary.sum(axis=0)).ravel()
        + np.asarray(second_binary.sum(axis=0)).ravel()
        - intersection
    )
    jaccard = np.divide(
        intersection,
        union,
        out=np.full(first.shape[1], np.nan),
        where=union > 0,
    )
    return jaccard, intersection.astype(np.int64)


def strand_symmetry_rows(
    forward: Activations,
    reverse_complement: Activations,
    indices: np.ndarray,
    *,
    split_name: str,
) -> pl.DataFrame:
    assert indices.ndim == 1 and len(indices) >= 2
    forward_states = paired_states(forward)
    reverse_states = paired_states(reverse_complement)
    columns: dict[str, np.ndarray] = {
        "feature_id": np.arange(forward.delta.shape[1], dtype=np.int64),
        "split": np.repeat(split_name, forward.delta.shape[1]),
    }
    for name in ("ref", "alt", "delta"):
        first = getattr(forward, name)[indices].tocsr()
        second = getattr(reverse_complement, name)[indices].tocsr()
        pearson, cosine = same_id_correlations(first, second, rows=len(indices))
        jaccard, intersection = column_jaccard(first, second)
        columns[f"{name}_forward_support"] = np.asarray(first.getnnz(axis=0)).ravel()
        columns[f"{name}_reverse_support"] = np.asarray(second.getnnz(axis=0)).ravel()
        columns[f"{name}_pearson"] = pearson
        columns[f"{name}_cosine"] = cosine
        columns[f"{name}_jaccard"] = jaccard
        columns[f"{name}_intersection"] = intersection

    first_delta = forward.delta[indices].tocsr()
    second_delta = reverse_complement.delta[indices].tocsr()
    same_sign = binary_matrix(first_delta > 0).multiply(binary_matrix(second_delta > 0))
    same_sign += binary_matrix(first_delta < 0).multiply(
        binary_matrix(second_delta < 0)
    )
    same_sign_count = np.asarray(same_sign.sum(axis=0)).ravel()
    columns["delta_sign_agreement"] = np.divide(
        same_sign_count,
        columns["delta_intersection"],
        out=np.full(first_delta.shape[1], np.nan),
        where=columns["delta_intersection"] > 0,
    )
    for state in ("both", "turn_on", "turn_off"):
        jaccard, intersection = column_jaccard(
            forward_states[state][indices].tocsr(),
            reverse_states[state][indices].tocsr(),
        )
        columns[f"{state}_jaccard"] = jaccard
        columns[f"{state}_intersection"] = intersection
    return pl.DataFrame(columns)


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
) -> dict[str, Any]:
    discovery = np.flatnonzero(split == "discovery")
    validation = np.flatnonzero(split == "validation")
    test = np.flatnonzero(split == "test")
    discovery_positive = labels[discovery] == class_name
    validation_positive = labels[validation] == class_name
    test_positive = labels[test] == class_name
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
    return {
        "class": class_name,
        "feature_id": feature_id,
        "direction": direction,
        "discovery_rank": int(selected_local + 1),
        "discovery_effect": float(discovery_effect[feature_id]),
        "discovery_t": float(discovery_t[feature_id]),
        "discovery_support": int(support[feature_id]),
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
    }


def bootstrap_block_ap(
    scores: np.ndarray,
    positive: np.ndarray,
    blocks: np.ndarray,
    *,
    seed: int,
    samples: int = BOOTSTRAPS,
) -> tuple[float | None, float | None]:
    assert scores.shape == positive.shape == blocks.shape
    unique_blocks = np.unique(blocks)
    positive_blocks = np.unique(blocks[positive])
    if len(positive_blocks) < 2:
        return None, None
    rows = [np.flatnonzero(blocks == block) for block in unique_blocks]
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(samples):
        indices = np.concatenate(
            [rows[index] for index in rng.integers(0, len(rows), size=len(rows))]
        )
        sampled = positive[indices]
        if sampled.any() and (~sampled).any():
            values.append(float(average_precision_score(sampled, scores[indices])))
    assert len(values) >= samples * 0.9
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def correlations_to_all(source: np.ndarray, target: Matrix) -> np.ndarray:
    assert source.shape == (target.shape[0],) and len(source) >= 2
    source = source.astype(np.float64)
    target_sum, target_square = _column_sums(target)
    source_sum = float(source.sum())
    source_square = float(np.square(source).sum())
    products = np.asarray(source @ target).ravel().astype(np.float64)
    source_centered = max(source_square - source_sum**2 / len(source), 0)
    target_centered = np.maximum(target_square - target_sum**2 / len(source), 0)
    denominator = np.sqrt(source_centered * target_centered)
    return np.divide(
        products - source_sum * target_sum / len(source),
        denominator,
        out=np.full(target.shape[1], np.nan),
        where=denominator > 0,
    )


def pair_correlation(first: np.ndarray, second: np.ndarray) -> float:
    assert first.shape == second.shape and len(first) >= 2
    if np.std(first) == 0 or np.std(second) == 0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def cross_feature_matches(
    forward: Matrix,
    reverse_complement: Matrix,
    prior_selection: pl.DataFrame,
    split: np.ndarray,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    split_indices = {
        name: np.flatnonzero(split == name)
        for name in ("discovery", "validation", "test")
    }
    matrices = {"forward": forward, "reverse_complement": reverse_complement}
    for source_orientation, target_orientation in (
        ("forward", "reverse_complement"),
        ("reverse_complement", "forward"),
    ):
        source_matrix = matrices[source_orientation]
        target_matrix = matrices[target_orientation]
        selected = prior_selection.filter(pl.col("orientation") == source_orientation)
        assert selected.height > 0
        for result in selected.iter_rows(named=True):
            feature_id = int(result["dimension"])
            discovery = split_indices["discovery"]
            source_values = column_values(source_matrix, discovery, feature_id)
            correlations = correlations_to_all(
                source_values, target_matrix[discovery].tocsr()
            )
            finite = np.isfinite(correlations)
            assert finite.any()
            ranking = np.where(finite, np.abs(correlations), -np.inf)
            matched_feature_id = int(np.lexsort((np.arange(len(ranking)), -ranking))[0])
            row: dict[str, Any] = {
                "class": result["class"],
                "source_orientation": source_orientation,
                "source_feature_id": feature_id,
                "target_orientation": target_orientation,
                "matched_feature_id": matched_feature_id,
                "same_id_discovery_pearson": float(correlations[feature_id]),
                "matched_discovery_pearson": float(correlations[matched_feature_id]),
                "same_id_is_best_match": feature_id == matched_feature_id,
            }
            for name in ("validation", "test"):
                indices = split_indices[name]
                source_values = column_values(source_matrix, indices, feature_id)
                row[f"same_id_{name}_pearson"] = pair_correlation(
                    source_values,
                    column_values(target_matrix, indices, feature_id),
                )
                row[f"matched_{name}_pearson"] = pair_correlation(
                    source_values,
                    column_values(target_matrix, indices, matched_feature_id),
                )
            rows.append(row)
    return pl.DataFrame(rows).sort(["class", "source_orientation"])


def paired_state_summary(
    forward: Activations,
    reverse_complement: Activations,
    selected: pl.DataFrame,
    labels: np.ndarray,
    split: np.ndarray,
) -> pl.DataFrame:
    test = np.flatnonzero(split == "test")
    rows: list[dict[str, Any]] = []
    for selection in selected.iter_rows(named=True):
        class_name = selection["class"]
        feature_id = int(selection["feature_id"])
        positive = labels[test] == class_name
        for orientation, activations in (
            ("forward", forward),
            ("reverse_complement", reverse_complement),
        ):
            ref = column_values(activations.ref, test, feature_id)
            alt = column_values(activations.alt, test, feature_id)
            delta = alt - ref
            for group, group_mask in (("class", positive), ("other", ~positive)):
                group_ref = ref[group_mask]
                group_alt = alt[group_mask]
                group_delta = delta[group_mask]
                ref_active = group_ref > 0
                alt_active = group_alt > 0
                count = int(group_mask.sum())
                rows.append(
                    {
                        "class": class_name,
                        "feature_id": feature_id,
                        "view": selection["view"],
                        "orientation": orientation,
                        "group": group,
                        "rows": count,
                        "ref_active_fraction": float(ref_active.mean()),
                        "alt_active_fraction": float(alt_active.mean()),
                        "both_fraction": float((ref_active & alt_active).mean()),
                        "turn_on_fraction": float((~ref_active & alt_active).mean()),
                        "turn_off_fraction": float((ref_active & ~alt_active).mean()),
                        "neither_fraction": float((~ref_active & ~alt_active).mean()),
                        "positive_delta_fraction": float((group_delta > 0).mean()),
                        "negative_delta_fraction": float((group_delta < 0).mean()),
                        "mean_ref_activation": float(group_ref.mean()),
                        "mean_alt_activation": float(group_alt.mean()),
                        "mean_delta": float(group_delta.mean()),
                        "mean_absolute_delta": float(np.abs(group_delta).mean()),
                    }
                )
    return pl.DataFrame(rows).sort(["class", "orientation", "group"])


def plot_reducers(summary: pl.DataFrame, output_dir: Path) -> None:
    ordered = summary.sort("validation_mean_ap", descending=True)
    names = ordered["view"].to_list()
    validation = ordered["validation_mean_ap"].to_numpy()
    test = ordered["test_mean_ap"].to_numpy()
    positions = np.arange(len(names))
    figure, axis = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    axis.bar(positions - 0.2, validation, width=0.4, label="validation")
    axis.bar(positions + 0.2, test, width=0.4, label="held-out test")
    axis.axhline(1 / 35, color="black", linestyle="--", linewidth=1, label="chance")
    axis.set_xticks(positions, names, rotation=30, ha="right")
    axis.set_ylabel("Mean one-vs-rest AUPRC across consequence classes")
    axis.set_title("Feature/reducer selected without test labels")
    axis.legend()
    for suffix in ("png", "svg"):
        figure.savefig(output_dir / f"reducer_comparison.{suffix}", dpi=180)
    plt.close(figure)


def plot_symmetry(
    symmetry: pl.DataFrame, matches: pl.DataFrame, output_dir: Path
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    split_order = ["discovery", "validation", "test"]
    data = [
        symmetry.filter(pl.col("split") == name)["delta_pearson"]
        .drop_nulls()
        .drop_nans()
        .to_numpy()
        for name in split_order
    ]
    axes[0].boxplot(data, tick_labels=split_order, showfliers=False)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_ylabel("Same-ID FWD/RC Pearson correlation")
    axes[0].set_title("All nonconstant SAE delta features")
    same = np.abs(matches["same_id_validation_pearson"].to_numpy())
    matched = np.abs(matches["matched_validation_pearson"].to_numpy())
    axes[1].scatter(same, matched, alpha=0.65, s=24)
    limit = max(1.0, float(np.nanmax(np.concatenate((same, matched)))))
    axes[1].plot([0, limit], [0, limit], color="black", linestyle="--", linewidth=1)
    axes[1].set_xlim(0, limit)
    axes[1].set_ylim(0, limit)
    axes[1].set_xlabel("Absolute same-ID validation Pearson")
    axes[1].set_ylabel("Absolute cross-ID validation Pearson")
    axes[1].set_title("Discovery-matched feature transfer")
    for suffix in ("png", "svg"):
        figure.savefig(output_dir / f"strand_symmetry.{suffix}", dpi=180)
    plt.close(figure)


def analyze(
    *,
    extraction_dir: Path,
    panel_path: Path,
    prior_selection_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    assert extraction_dir.is_dir() and panel_path.is_file()
    assert prior_selection_path.is_file() and not output_dir.exists()
    analysis_commit = os.environ.get("ANALYSIS_COMMIT", "")
    assert len(analysis_commit) == 40
    extraction_manifest_path = extraction_dir / "manifest.json"
    extraction_manifest = json.loads(extraction_manifest_path.read_text())
    for filename, metadata in extraction_manifest["artifacts"].items():
        path = extraction_dir / filename
        assert path.is_file() and sha256_file(path) == metadata["sha256"]
    assert sha256_file(panel_path) == extraction_manifest["panel"]["sha256"]
    panel = pl.read_parquet(panel_path)
    assert panel["panel_row"].to_list() == list(range(panel.height))
    labels = panel["consequence_cre"].to_numpy()
    split = panel["split"].to_numpy()
    blocks = panel["block_id"].to_numpy()
    classes = sorted(np.unique(labels).tolist())
    assert set(np.unique(split)) == {"discovery", "validation", "test"}
    assert len(classes) == 35
    d_sae = int(extraction_manifest["sae"]["d_sae"])
    activations = {
        orientation: load_activations(
            extraction_dir / f"sae_activations_{orientation}.parquet",
            rows=panel.height,
            columns=d_sae,
        )
        for orientation in ORIENTATIONS
    }
    forward = activations["forward"]
    reverse = activations["reverse_complement"]

    symmetry = pl.concat(
        [
            strand_symmetry_rows(
                forward,
                reverse,
                np.flatnonzero(split == split_name),
                split_name=split_name,
            )
            for split_name in ("discovery", "validation", "test")
        ]
    ).sort(["split", "feature_id"])

    prior_selection = pl.read_parquet(prior_selection_path)
    assert {"class", "orientation", "dimension"}.issubset(prior_selection.columns)
    matches = cross_feature_matches(
        forward.delta, reverse.delta, prior_selection, split
    )

    views = make_views(forward.delta, reverse.delta)
    metric_rows: list[dict[str, Any]] = []
    for view_name, matrix in views.items():
        for class_name in classes:
            metric_rows.append(
                {"view": view_name, **select_feature(matrix, labels, split, class_name)}
            )
        print(json.dumps({"stage": "view", "view": view_name}), flush=True)
    metrics = pl.DataFrame(metric_rows).sort(["view", "class"])
    reducer_summary = (
        metrics.group_by("view")
        .agg(
            pl.col("validation_average_precision").mean().alias("validation_mean_ap"),
            pl.col("validation_average_precision")
            .median()
            .alias("validation_median_ap"),
            pl.col("test_average_precision").mean().alias("test_mean_ap"),
            pl.col("test_average_precision").median().alias("test_median_ap"),
        )
        .sort("view")
    )
    aggregate_summary = reducer_summary.filter(pl.col("view").is_in(AGGREGATE_VIEWS))
    selected_view = aggregate_summary.sort(
        ["validation_mean_ap", "view"], descending=[True, False]
    )["view"].item()
    selected = metrics.filter(pl.col("view") == selected_view).sort("class")
    test = np.flatnonzero(split == "test")
    selected_rows: list[dict[str, Any]] = []
    for row in selected.iter_rows(named=True):
        scores = row["direction"] * column_values(
            views[selected_view], test, int(row["feature_id"])
        )
        positive = labels[test] == row["class"]
        low, high = bootstrap_block_ap(
            scores,
            positive,
            blocks[test],
            seed=RANDOM_SEED + int(row["feature_id"]) + sum(map(ord, row["class"])),
        )
        selected_rows.append(
            {**row, "test_ap_ci95_low": low, "test_ap_ci95_high": high}
        )
    selected = pl.DataFrame(selected_rows).sort("class")
    state_summary = paired_state_summary(forward, reverse, selected, labels, split)

    output_dir.mkdir(parents=True, exist_ok=False)
    symmetry.write_parquet(output_dir / "same_id_strand_symmetry.parquet")
    matches.write_parquet(output_dir / "cross_id_strand_matches.parquet")
    metrics.write_parquet(output_dir / "view_feature_metrics.parquet")
    reducer_summary.write_parquet(output_dir / "reducer_summary.parquet")
    selected.write_parquet(output_dir / "selected_reducer_features.parquet")
    state_summary.write_parquet(output_dir / "paired_state_summary.parquet")
    plot_reducers(reducer_summary, output_dir)
    plot_symmetry(symmetry, matches, output_dir)

    result = {
        "analysis_commit": analysis_commit,
        "extraction_manifest_sha256": sha256_file(extraction_manifest_path),
        "panel_sha256": sha256_file(panel_path),
        "prior_selection_sha256": sha256_file(prior_selection_path),
        "rows": panel.height,
        "classes": len(classes),
        "d_sae": d_sae,
        "selected_aggregate_view": selected_view,
        "selection_rule": "highest mean validation AP across 35 classes",
        "protocol": {
            "top_candidates": TOP_CANDIDATES,
            "minimum_discovery_support": MIN_DISCOVERY_SUPPORT,
            "minimum_positive_support": MIN_POSITIVE_SUPPORT,
            "block_bootstraps": BOOTSTRAPS,
            "aggregate_views": list(AGGREGATE_VIEWS),
            "random_seed": RANDOM_SEED,
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
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--prior-selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = analyze(
        extraction_dir=args.extraction_dir,
        panel_path=args.panel,
        prior_selection_path=args.prior_selection,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
