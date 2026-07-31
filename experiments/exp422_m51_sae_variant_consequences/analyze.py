"""Analyze held-out chr21 variant-consequence signals from issue #422."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pyarrow.parquet as pq
from marin_dna.data.dna import reverse_complement
from marin_dna.data.genome import Genome
from scipy import sparse
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

ORIENTATIONS = ("forward", "reverse_complement", "mean")
SPACES = ("sae",)
TRANSFORMS = ("signed", "absolute")
TOP_CANDIDATES = 64
MIN_DISCOVERY_SUPPORT = 32
MIN_POSITIVE_SUPPORT = 8
BOOTSTRAPS = 250
PROBE_ALPHAS = (1e-5, 1e-4, 1e-3)
PROBE_EPOCHS = 100
RANDOM_SEED = 422
CONTEXT_RADIUS = 15

Matrix = np.ndarray | sparse.csr_matrix


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_sparse_delta(path: Path, *, rows: int, columns: int) -> sparse.csr_matrix:
    table = pq.read_table(path, columns=["panel_row", "feature_id", "delta"])
    panel_rows = table["panel_row"].to_numpy(zero_copy_only=False).astype(np.int64)
    feature_ids = table["feature_id"].to_numpy(zero_copy_only=False).astype(np.int64)
    values = table["delta"].to_numpy(zero_copy_only=False).astype(np.float32)
    assert len(panel_rows) == len(feature_ids) == len(values)
    assert np.isfinite(values).all()
    assert (panel_rows >= 0).all() and (panel_rows < rows).all()
    assert (feature_ids >= 0).all() and (feature_ids < columns).all()
    keys = panel_rows * columns + feature_ids
    assert len(np.unique(keys)) == len(keys), "duplicate panel_row/feature_id entries"
    matrix = sparse.csr_matrix(
        (values, (panel_rows, feature_ids)), shape=(rows, columns)
    )
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix


def transform_matrix(
    matrix: Matrix, transform: Literal["signed", "absolute"]
) -> Matrix:
    if transform == "signed":
        return matrix
    if sparse.issparse(matrix):
        output = matrix.copy()
        output.data = np.abs(output.data)
        return output
    return np.abs(np.asarray(matrix))


def column_stats(
    matrix: Matrix, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert indices.ndim == 1 and len(indices) >= 2
    selected = matrix[indices]
    if sparse.issparse(selected):
        selected = selected.tocsr()
        means = np.asarray(selected.mean(axis=0)).ravel().astype(np.float64)
        squared = selected.copy()
        squared.data = squared.data.astype(np.float64) ** 2
        mean_squares = np.asarray(squared.mean(axis=0)).ravel()
        variances = np.maximum(mean_squares - means**2, 0.0)
        support = np.asarray(selected.getnnz(axis=0)).ravel().astype(np.int64)
    else:
        selected = np.asarray(selected)
        means = selected.mean(axis=0, dtype=np.float64)
        variances = selected.var(axis=0, dtype=np.float64)
        support = np.count_nonzero(selected, axis=0).astype(np.int64)
    assert np.isfinite(means).all() and np.isfinite(variances).all()
    return means, variances, support


def welch_statistics(
    matrix: Matrix,
    indices: np.ndarray,
    positive: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert positive.shape == (len(indices),)
    assert positive.any() and (~positive).any()
    positive_indices = indices[positive]
    negative_indices = indices[~positive]
    positive_mean, positive_var, positive_support = column_stats(
        matrix, positive_indices
    )
    negative_mean, negative_var, _ = column_stats(matrix, negative_indices)
    effects = positive_mean - negative_mean
    standard_error = np.sqrt(
        positive_var / len(positive_indices) + negative_var / len(negative_indices)
    )
    statistics = np.divide(
        effects,
        standard_error,
        out=np.zeros_like(effects),
        where=standard_error > 0,
    )
    _, _, total_support = column_stats(matrix, indices)
    assert np.isfinite(effects).all() and np.isfinite(statistics).all()
    return effects, statistics, total_support, positive_support


def column_values(matrix: Matrix, indices: np.ndarray, dimension: int) -> np.ndarray:
    values = matrix[indices, dimension]
    if sparse.issparse(values):
        values = values.toarray()
    output = np.asarray(values).reshape(-1).astype(np.float64)
    assert output.shape == (len(indices),) and np.isfinite(output).all()
    return output


def select_individual_feature(
    matrix: Matrix,
    labels: np.ndarray,
    split: np.ndarray,
    class_name: str,
) -> dict[str, Any]:
    discovery_indices = np.flatnonzero(split == "discovery")
    validation_indices = np.flatnonzero(split == "validation")
    test_indices = np.flatnonzero(split == "test")
    discovery_positive = labels[discovery_indices] == class_name
    validation_positive = labels[validation_indices] == class_name
    test_positive = labels[test_indices] == class_name
    assert discovery_positive.sum() == 256
    assert validation_positive.sum() == test_positive.sum() == 128

    discovery_effect, discovery_t, support, positive_support = welch_statistics(
        matrix, discovery_indices, discovery_positive
    )
    eligible = (support >= MIN_DISCOVERY_SUPPORT) & (
        positive_support >= MIN_POSITIVE_SUPPORT
    )
    assert eligible.any(), class_name
    ranking_score = np.where(eligible, np.abs(discovery_t), -np.inf)
    ranked = np.lexsort((np.arange(matrix.shape[1]), -ranking_score))[:TOP_CANDIDATES]
    ranked = ranked[np.isfinite(ranking_score[ranked])]
    assert len(ranked) > 0

    validation_effect, validation_t, _, _ = welch_statistics(
        matrix, validation_indices, validation_positive
    )
    directions = np.where(discovery_effect[ranked] >= 0, 1, -1)
    oriented_validation = directions * validation_t[ranked]
    consistent = oriented_validation > 0
    if consistent.any():
        choices = np.flatnonzero(consistent)
        replicated = True
    else:
        choices = np.arange(len(ranked))
        replicated = False
    best_local = choices[
        np.lexsort((ranked[choices], -oriented_validation[choices]))[0]
    ]
    dimension = int(ranked[best_local])
    direction = int(directions[best_local])
    test_scores = direction * column_values(matrix, test_indices, dimension)
    test_ap = float(average_precision_score(test_positive, test_scores))
    return {
        "class": class_name,
        "dimension": dimension,
        "direction": direction,
        "validation_direction_consistent": replicated,
        "discovery_rank": int(best_local + 1),
        "discovery_effect": float(discovery_effect[dimension]),
        "discovery_t": float(discovery_t[dimension]),
        "discovery_support": int(support[dimension]),
        "discovery_positive_support": int(positive_support[dimension]),
        "validation_effect": float(validation_effect[dimension]),
        "validation_t": float(validation_t[dimension]),
        "validation_oriented_t": float(direction * validation_t[dimension]),
        "test_average_precision": test_ap,
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
) -> tuple[float | None, float | None, int, int]:
    assert scores.shape == positive.shape == blocks.shape
    unique_blocks = np.unique(blocks)
    positive_blocks = np.unique(blocks[positive])
    assert len(unique_blocks) >= 2
    assert len(positive_blocks) >= 1
    if len(positive_blocks) < 2:
        return None, None, len(unique_blocks), len(positive_blocks)

    block_rows = [np.flatnonzero(blocks == block) for block in unique_blocks]
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(samples):
        chosen = rng.integers(0, len(block_rows), size=len(block_rows))
        indices = np.concatenate([block_rows[index] for index in chosen])
        sampled_labels = positive[indices]
        if sampled_labels.any() and (~sampled_labels).any():
            values.append(
                float(average_precision_score(sampled_labels, scores[indices]))
            )
    assert len(values) >= samples * 0.9
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high), len(unique_blocks), len(positive_blocks)


def choose_transform(results: pl.DataFrame) -> pl.DataFrame:
    return (
        results.sort(
            ["class", "orientation", "space", "validation_oriented_t", "transform"],
            descending=[False, False, False, True, False],
        )
        .group_by(["class", "orientation", "space"], maintain_order=True)
        .head(1)
        .sort(["class", "orientation", "space"])
    )


def fit_probe(
    matrix: Matrix,
    labels: np.ndarray,
    split: np.ndarray,
    *,
    orientation: str,
    space: str,
    transform: str,
    probe_jobs: int,
) -> tuple[dict[str, Any], np.ndarray, list[str]]:
    assert probe_jobs >= 1
    encoder = LabelEncoder().fit(labels)
    encoded = encoder.transform(labels)
    discovery = np.flatnonzero(split == "discovery")
    validation = np.flatnonzero(split == "validation")
    test = np.flatnonzero(split == "test")
    scaler = StandardScaler(with_mean=False).fit(matrix[discovery])
    discovery_x = scaler.transform(matrix[discovery])
    validation_x = scaler.transform(matrix[validation])
    candidates: list[tuple[float, float]] = []
    for alpha in PROBE_ALPHAS:
        classifier = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=alpha,
            max_iter=PROBE_EPOCHS,
            tol=None,
            random_state=RANDOM_SEED,
            average=True,
            n_jobs=probe_jobs,
        ).fit(discovery_x, encoded[discovery])
        prediction = classifier.predict(validation_x)
        candidates.append(
            (float(f1_score(encoded[validation], prediction, average="macro")), alpha)
        )
    validation_macro_f1, alpha = max(candidates, key=lambda row: (row[0], -row[1]))

    train = np.concatenate((discovery, validation))
    scaler = StandardScaler(with_mean=False).fit(matrix[train])
    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        max_iter=PROBE_EPOCHS,
        tol=None,
        random_state=RANDOM_SEED,
        average=True,
        n_jobs=probe_jobs,
    ).fit(scaler.transform(matrix[train]), encoded[train])
    prediction = classifier.predict(scaler.transform(matrix[test]))
    matrix_confusion = confusion_matrix(
        encoded[test],
        prediction,
        labels=np.arange(len(encoder.classes_)),
        normalize="true",
    )
    return (
        {
            "orientation": orientation,
            "space": space,
            "transform": transform,
            "alpha": alpha,
            "epochs": int(classifier.n_iter_),
            "validation_macro_f1": validation_macro_f1,
            "test_macro_f1": float(
                f1_score(encoded[test], prediction, average="macro")
            ),
            "test_balanced_accuracy": float(
                balanced_accuracy_score(encoded[test], prediction)
            ),
            "test_accuracy": float(accuracy_score(encoded[test], prediction)),
            "classes": len(encoder.classes_),
            "features": matrix.shape[1],
        },
        matrix_confusion,
        encoder.classes_.tolist(),
    )


def context_rows(
    panel: pl.DataFrame,
    selected: pl.DataFrame,
    matrices: dict[tuple[str, str], Matrix],
    *,
    fasta_path: Path,
) -> pl.DataFrame:
    genome = Genome(fasta_path, subset_chroms={"21"})
    split = panel["split"].to_numpy()
    labels = panel["consequence_cre"].to_numpy()
    test = np.flatnonzero(split == "test")
    rows: list[dict[str, Any]] = []
    for result in selected.filter(
        (pl.col("space") == "sae")
        & pl.col("orientation").is_in(["forward", "reverse_complement"])
    ).iter_rows(named=True):
        matrix = transform_matrix(
            matrices[(result["orientation"], "sae")], result["transform"]
        )
        scores = result["direction"] * column_values(matrix, test, result["dimension"])
        positive = labels[test] == result["class"]
        for group, eligible in (("class", positive), ("other", ~positive)):
            local = np.flatnonzero(eligible)
            ordered = local[np.argsort(-scores[local], kind="stable")[:3]]
            for rank, local_index in enumerate(ordered, start=1):
                panel_index = int(test[local_index])
                row = panel.row(panel_index, named=True)
                pos0 = int(row["pos"]) - 1
                reference = genome(
                    "21", pos0 - CONTEXT_RADIUS, pos0 + CONTEXT_RADIUS + 1, "+"
                ).upper()
                alternate = (
                    reference[:CONTEXT_RADIUS]
                    + row["alt"]
                    + reference[CONTEXT_RADIUS + 1 :]
                )
                if result["orientation"] == "reverse_complement":
                    reference = reverse_complement(reference)
                    alternate = reverse_complement(alternate)
                rows.append(
                    {
                        "class": result["class"],
                        "orientation": result["orientation"],
                        "feature_id": result["dimension"],
                        "transform": result["transform"],
                        "direction": result["direction"],
                        "group": group,
                        "rank": rank,
                        "score": float(scores[local_index]),
                        "panel_row": panel_index,
                        "chrom": row["chrom"],
                        "pos": row["pos"],
                        "ref": row["ref"],
                        "alt": row["alt"],
                        "consequence": row["consequence"],
                        "consequence_cre": row["consequence_cre"],
                        "ref_context": reference,
                        "alt_context": alternate,
                    }
                )
    return pl.DataFrame(rows).sort(["class", "orientation", "group", "rank"])


def plot_summary(
    selected: pl.DataFrame,
    probes: pl.DataFrame,
    output_dir: Path,
) -> None:
    class_order = sorted(selected["class"].unique().to_list())
    columns = [
        ("forward", "sae", "FWD SAE"),
        ("reverse_complement", "sae", "RC SAE"),
        ("mean", "sae", "mean SAE"),
    ]
    heatmap = np.full((len(class_order), len(columns)), np.nan)
    for row_index, class_name in enumerate(class_order):
        for column_index, (orientation, space, _) in enumerate(columns):
            row = selected.filter(
                (pl.col("class") == class_name)
                & (pl.col("orientation") == orientation)
                & (pl.col("space") == space)
            )
            assert row.height == 1
            heatmap[row_index, column_index] = row["test_average_precision"].item()

    figure = plt.figure(figsize=(12, 15), constrained_layout=True)
    grid = figure.add_gridspec(2, 1, height_ratios=[8, 2])
    axis = figure.add_subplot(grid[0])
    image = axis.imshow(
        heatmap,
        aspect="auto",
        vmin=1 / 35,
        vmax=max(0.5, float(np.nanmax(heatmap))),
        cmap="viridis",
    )
    axis.set_xticks(
        range(len(columns)), [column[2] for column in columns], rotation=30, ha="right"
    )
    axis.set_yticks(range(len(class_order)), class_order, fontsize=8)
    axis.set_title("Held-out one-vs-rest AUPRC (transform selected on validation)")
    figure.colorbar(image, ax=axis, label="AUPRC")

    probe_axis = figure.add_subplot(grid[1])
    probe_rows = probes.sort(["space", "orientation", "transform"])
    labels = [
        f"{row['orientation']} {row['space']} {row['transform']}"
        for row in probe_rows.iter_rows(named=True)
    ]
    values = probe_rows["test_macro_f1"].to_numpy()
    probe_axis.barh(np.arange(len(values)), values, color="#4C78A8")
    probe_axis.set_yticks(np.arange(len(values)), labels, fontsize=8)
    probe_axis.invert_yaxis()
    probe_axis.axvline(
        1 / 35, color="black", linestyle="--", linewidth=1, label="chance"
    )
    probe_axis.set_xlabel("Held-out macro-F1")
    probe_axis.set_title("Multiclass linear probes")
    probe_axis.legend(loc="lower right")
    for suffix in ("png", "svg"):
        figure.savefig(output_dir / f"summary.{suffix}", dpi=180)
    plt.close(figure)


def plot_best_confusion(
    probes: pl.DataFrame,
    confusion_rows: pl.DataFrame,
    output_dir: Path,
) -> None:
    best = probes.sort(
        ["validation_macro_f1", "orientation", "space", "transform"],
        descending=[True, False, False, False],
    ).row(0, named=True)
    selected = confusion_rows.filter(
        (pl.col("orientation") == best["orientation"])
        & (pl.col("space") == best["space"])
        & (pl.col("transform") == best["transform"])
    )
    classes = sorted(selected["true_class"].unique().to_list())
    matrix = np.empty((len(classes), len(classes)), dtype=np.float64)
    for i, true_class in enumerate(classes):
        for j, predicted_class in enumerate(classes):
            row = selected.filter(
                (pl.col("true_class") == true_class)
                & (pl.col("predicted_class") == predicted_class)
            )
            assert row.height == 1
            matrix[i, j] = row["fraction"].item()
    figure, axis = plt.subplots(figsize=(13, 12), constrained_layout=True)
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap="magma")
    axis.set_xticks(range(len(classes)), classes, rotation=90, fontsize=6)
    axis.set_yticks(range(len(classes)), classes, fontsize=6)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title(
        f"Best validation-selected probe: {best['orientation']} {best['space']} {best['transform']}"
    )
    figure.colorbar(image, ax=axis, label="Fraction within true class")
    for suffix in ("png", "svg"):
        figure.savefig(output_dir / f"confusion.{suffix}", dpi=180)
    plt.close(figure)


def analyze(
    *,
    extraction_dir: Path,
    panel_path: Path,
    fasta_path: Path,
    output_dir: Path,
    probe_jobs: int,
) -> dict[str, Any]:
    assert extraction_dir.is_dir() and panel_path.is_file() and fasta_path.is_file()
    assert probe_jobs >= 1
    assert not output_dir.exists()
    analysis_commit = os.environ.get("ANALYSIS_COMMIT", "")
    assert len(analysis_commit) == 40
    manifest = json.loads((extraction_dir / "manifest.json").read_text())
    for filename, metadata in manifest["artifacts"].items():
        path = extraction_dir / filename
        assert path.is_file() and sha256_file(path) == metadata["sha256"]
    panel = pl.read_parquet(panel_path)
    assert sha256_file(panel_path) == manifest["panel"]["sha256"]
    assert panel.height == manifest["panel"]["rows"] == 17_920
    labels = panel["consequence_cre"].to_numpy()
    split = panel["split"].to_numpy()
    blocks = panel["block_id"].to_numpy()
    classes = sorted(np.unique(labels).tolist())
    assert len(classes) == 35
    d_sae = int(manifest["sae"]["d_sae"])

    matrices: dict[tuple[str, str], Matrix] = {}
    for orientation in ("forward", "reverse_complement"):
        matrices[(orientation, "sae")] = load_sparse_delta(
            extraction_dir / f"sae_activations_{orientation}.parquet",
            rows=panel.height,
            columns=d_sae,
        )
    mean_sae = (
        (matrices[("forward", "sae")] + matrices[("reverse_complement", "sae")])
        .multiply(0.5)
        .tocsr()
    )
    mean_sae.eliminate_zeros()
    mean_sae.sort_indices()
    matrices[("mean", "sae")] = mean_sae
    individual_rows: list[dict[str, Any]] = []
    for orientation in ORIENTATIONS:
        for space in SPACES:
            for transform in TRANSFORMS:
                transformed = transform_matrix(
                    matrices[(orientation, space)], transform
                )
                for class_name in classes:
                    individual_rows.append(
                        {
                            "orientation": orientation,
                            "space": space,
                            "transform": transform,
                            **select_individual_feature(
                                transformed, labels, split, class_name
                            ),
                        }
                    )
                print(
                    json.dumps(
                        {
                            "stage": "individual_features",
                            "orientation": orientation,
                            "space": space,
                            "transform": transform,
                            "classes_complete": len(classes),
                        }
                    ),
                    flush=True,
                )
    individual = pl.DataFrame(individual_rows).sort(
        ["class", "orientation", "space", "transform"]
    )
    selected = choose_transform(individual)

    test = np.flatnonzero(split == "test")
    selected_rows: list[dict[str, Any]] = []
    for row in selected.iter_rows(named=True):
        transformed = transform_matrix(
            matrices[(row["orientation"], row["space"])], row["transform"]
        )
        scores = row["direction"] * column_values(transformed, test, row["dimension"])
        positive = labels[test] == row["class"]
        low, high, test_blocks, test_positive_blocks = bootstrap_block_ap(
            scores,
            positive,
            blocks[test],
            seed=RANDOM_SEED + int(row["dimension"]) + sum(map(ord, row["class"])),
        )
        selected_rows.append(
            {
                **row,
                "test_ap_ci95_low": low,
                "test_ap_ci95_high": high,
                "test_blocks": test_blocks,
                "test_positive_blocks": test_positive_blocks,
            }
        )
    selected = pl.DataFrame(selected_rows).sort(["class", "orientation", "space"])

    probe_rows: list[dict[str, Any]] = []
    confusion_data: list[dict[str, Any]] = []
    for orientation in ORIENTATIONS:
        for space in SPACES:
            for transform in TRANSFORMS:
                transformed = transform_matrix(
                    matrices[(orientation, space)], transform
                )
                metrics, matrix_confusion, probe_classes = fit_probe(
                    transformed,
                    labels,
                    split,
                    orientation=orientation,
                    space=space,
                    transform=transform,
                    probe_jobs=probe_jobs,
                )
                probe_rows.append(metrics)
                print(
                    json.dumps(
                        {
                            "stage": "probe",
                            "orientation": orientation,
                            "space": space,
                            "transform": transform,
                            "validation_macro_f1": metrics["validation_macro_f1"],
                        }
                    ),
                    flush=True,
                )
                for i, true_class in enumerate(probe_classes):
                    for j, predicted_class in enumerate(probe_classes):
                        confusion_data.append(
                            {
                                "orientation": orientation,
                                "space": space,
                                "transform": transform,
                                "true_class": true_class,
                                "predicted_class": predicted_class,
                                "fraction": float(matrix_confusion[i, j]),
                            }
                        )

    probes = pl.DataFrame(probe_rows).sort(["space", "orientation", "transform"])
    confusion = pl.DataFrame(confusion_data).sort(
        ["space", "orientation", "transform", "true_class", "predicted_class"]
    )
    contexts = context_rows(panel, selected, matrices, fasta_path=fasta_path)

    output_dir.mkdir(parents=True, exist_ok=False)
    individual.write_parquet(output_dir / "individual_features.parquet")
    selected.write_parquet(output_dir / "selected_individual_features.parquet")
    probes.write_parquet(output_dir / "probe_metrics.parquet")
    confusion.write_parquet(output_dir / "probe_confusion.parquet")
    contexts.write_parquet(output_dir / "contexts.parquet")
    plot_summary(selected, probes, output_dir)
    plot_best_confusion(probes, confusion, output_dir)

    summary = {
        "analysis_commit": analysis_commit,
        "extraction_manifest_sha256": sha256_file(extraction_dir / "manifest.json"),
        "panel_sha256": sha256_file(panel_path),
        "classes": len(classes),
        "chance_one_vs_rest_ap": 1 / len(classes),
        "chance_multiclass_accuracy": 1 / len(classes),
        "protocol": {
            "top_candidates": TOP_CANDIDATES,
            "minimum_discovery_support": MIN_DISCOVERY_SUPPORT,
            "minimum_positive_support": MIN_POSITIVE_SUPPORT,
            "block_bootstraps": BOOTSTRAPS,
            "minimum_positive_blocks_for_ci": 2,
            "probe_alphas": list(PROBE_ALPHAS),
            "probe_epochs": PROBE_EPOCHS,
            "probe_jobs": probe_jobs,
            "spaces": list(SPACES),
            "context_radius": CONTEXT_RADIUS,
            "orientation_order": list(ORIENTATIONS),
        },
    }
    write_json(output_dir / "results.json", summary)
    artifacts = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    output_manifest = {**summary, "artifacts": artifacts}
    write_json(output_dir / "manifest.json", output_manifest)
    return output_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction-dir", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probe-jobs", type=int, default=1)
    args = parser.parse_args()
    manifest = analyze(
        extraction_dir=args.extraction_dir,
        panel_path=args.panel,
        fasta_path=args.fasta,
        output_dir=args.output_dir,
        probe_jobs=args.probe_jobs,
    )
    print(json.dumps(manifest["artifacts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
