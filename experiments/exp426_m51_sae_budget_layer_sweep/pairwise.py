"""Directly compare missense and synonymous variants across all SAE arms."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import seaborn as sns
from scipy import sparse
from sklearn.metrics import average_precision_score, roc_auc_score

from analyze import (
    Activations,
    make_views,
    select_feature,
    sha256_file,
    stratified_block_resample_indices,
    write_json,
)
from train import BLOCK_INDICES, BUDGETS, D_SAE, ISSUE, arm_label, assert_commit

POSITIVE_CLASS = "missense_variant"
NEGATIVE_CLASS = "synonymous_variant"
PAIR_CLASSES = (POSITIVE_CLASS, NEGATIVE_CLASS)
PAIRWISE_CHANCE = 0.5
BOOTSTRAPS = 250
RANDOM_SEED = 4_260


def _selected_sparse_matrix(
    values: np.ndarray,
    original_rows: np.ndarray,
    selected_rows: np.ndarray,
    feature_ids: np.ndarray,
    columns: int,
) -> sparse.csr_matrix:
    local_rows = np.searchsorted(selected_rows, original_rows)
    assert (local_rows >= 0).all() and (local_rows < len(selected_rows)).all()
    np.testing.assert_array_equal(selected_rows[local_rows], original_rows)
    matrix = sparse.csr_matrix(
        (values, (local_rows, feature_ids)), shape=(len(selected_rows), columns)
    )
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix


def load_selected_activations(
    path: Path, *, selected_rows: np.ndarray, columns: int
) -> Activations:
    """Load only requested panel rows, using Parquet row-group filtering."""
    assert path.is_file()
    assert selected_rows.ndim == 1 and len(selected_rows) > 0
    assert np.all(selected_rows[:-1] < selected_rows[1:])
    fields = [
        "panel_row",
        "feature_id",
        "ref_activation",
        "alt_activation",
        "delta",
    ]
    table = pq.read_table(
        path,
        columns=fields,
        filters=[("panel_row", "in", selected_rows.tolist())],
    )
    original_rows = table["panel_row"].to_numpy(zero_copy_only=False).astype(np.int64)
    feature_ids = table["feature_id"].to_numpy(zero_copy_only=False).astype(np.int64)
    ref = table["ref_activation"].to_numpy(zero_copy_only=False).astype(np.float32)
    alt = table["alt_activation"].to_numpy(zero_copy_only=False).astype(np.float32)
    delta = table["delta"].to_numpy(zero_copy_only=False).astype(np.float32)
    assert len(original_rows) == len(feature_ids) == len(ref) == len(alt) == len(delta)
    assert len(original_rows) > 0
    assert np.isin(original_rows, selected_rows).all()
    assert (feature_ids >= 0).all() and (feature_ids < columns).all()
    assert (
        np.isfinite(ref).all() and np.isfinite(alt).all() and np.isfinite(delta).all()
    )
    assert (ref >= 0).all() and (alt >= 0).all()
    np.testing.assert_array_equal(delta, alt - ref)
    keys = original_rows * columns + feature_ids
    assert len(np.unique(keys)) == len(keys), "duplicate panel_row/feature_id entries"
    output = Activations(
        ref=_selected_sparse_matrix(
            ref, original_rows, selected_rows, feature_ids, columns
        ),
        alt=_selected_sparse_matrix(
            alt, original_rows, selected_rows, feature_ids, columns
        ),
        delta=_selected_sparse_matrix(
            delta, original_rows, selected_rows, feature_ids, columns
        ),
    )
    difference = output.alt - output.ref - output.delta
    assert difference.nnz == 0 or np.max(np.abs(difference.data)) == 0
    return output


def bootstrap_pairwise_metrics(
    scores: np.ndarray,
    positive: np.ndarray,
    blocks: np.ndarray,
    *,
    seed: int,
    samples: int = BOOTSTRAPS,
) -> dict[str, float]:
    """Bootstrap paired AP and AUROC by label-stratified genomic blocks."""
    assert scores.shape == positive.shape == blocks.shape
    assert positive.dtype == np.bool_ and positive.any() and (~positive).any()
    assert len(np.unique(blocks[positive])) >= 2
    assert len(np.unique(blocks[~positive])) >= 2
    rng = np.random.default_rng(seed)
    average_precisions: list[float] = []
    aurocs: list[float] = []
    for _ in range(samples):
        indices = stratified_block_resample_indices(positive, blocks, rng)
        sampled_positive = positive[indices]
        assert sampled_positive.any() and (~sampled_positive).any()
        average_precisions.append(
            float(average_precision_score(sampled_positive, scores[indices]))
        )
        aurocs.append(float(roc_auc_score(sampled_positive, scores[indices])))
    ap_low, ap_high = np.quantile(average_precisions, [0.025, 0.975])
    auroc_low, auroc_high = np.quantile(aurocs, [0.025, 0.975])
    return {
        "test_average_precision_ci95_low": float(ap_low),
        "test_average_precision_ci95_high": float(ap_high),
        "test_auroc_ci95_low": float(auroc_low),
        "test_auroc_ci95_high": float(auroc_high),
    }


def plot_pairwise(rows: pl.DataFrame, output_dir: Path) -> None:
    plot_rows = rows.with_columns(
        pl.when(pl.col("view") == "forward_signed")
        .then(pl.lit("FWD"))
        .when(pl.col("view") == "reverse_complement_signed")
        .then(pl.lit("RC"))
        .when(pl.col("view") == "signed_mean")
        .then(pl.lit("signed mean"))
        .otherwise(pl.lit("max abs"))
        .alias("view_label"),
        pl.when(pl.col("budget") == BUDGETS[0])
        .then(pl.lit("5"))
        .otherwise(pl.lit("25"))
        .alias("budget_label"),
    ).to_pandas()
    view_order = ["FWD", "RC", "signed mean", "max abs"]
    palette = {
        "FWD": "#4C78A8",
        "RC": "#F58518",
        "signed mean": "#54A24B",
        "max abs": "#B279A2",
    }
    grid = sns.relplot(
        data=plot_rows,
        x="reported_block",
        y="test_auroc",
        hue="view_label",
        hue_order=view_order,
        style="view_label",
        style_order=view_order,
        col="budget_label",
        col_order=["5", "25"],
        kind="line",
        markers=True,
        dashes=False,
        errorbar=None,
        palette=palette,
        height=4.7,
        aspect=0.95,
    )
    for budget_label, axis in grid.axes_dict.items():
        facet = plot_rows[plot_rows["budget_label"] == budget_label]
        for view_label in view_order:
            selected = facet[facet["view_label"] == view_label]
            y = selected["test_auroc"].to_numpy()
            low = selected["test_auroc_ci95_low"].to_numpy()
            high = selected["test_auroc_ci95_high"].to_numpy()
            axis.errorbar(
                selected["reported_block"],
                y,
                yerr=np.vstack((np.maximum(0, y - low), np.maximum(0, high - y))),
                fmt="none",
                capsize=2.5,
                linewidth=1,
                color=palette[view_label],
                alpha=0.7,
            )
        axis.axhline(PAIRWISE_CHANCE, color="black", linestyle="--", linewidth=1)
        axis.set_xticks([index + 1 for index in BLOCK_INDICES])
    grid.set_axis_labels(
        "Reported transformer block", "Held-out missense vs synonymous AUROC"
    )
    grid.set_titles("{col_name}M activations")
    if grid.legend is not None:
        grid.legend.set_title("View")
    grid.figure.suptitle(
        "Direct missense vs synonymous discrimination\n"
        "error bars = genomic-block bootstrap 95% CI",
        y=1.06,
    )
    for suffix in ("png", "svg"):
        grid.figure.savefig(
            output_dir / f"missense_vs_synonymous.{suffix}",
            dpi=180,
            bbox_inches="tight",
        )
    plt.close(grid.figure)


def analyze_pairwise(
    *,
    extraction_dir: Path,
    extraction_manifest_path: Path,
    panel_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    assert extraction_dir.is_dir() and extraction_manifest_path.is_file()
    assert panel_path.is_file() and not output_dir.exists()
    analysis_commit = os.environ.get("ANALYSIS_COMMIT", "")
    assert_commit(analysis_commit)
    extraction_manifest = json.loads(extraction_manifest_path.read_text())
    assert extraction_manifest["issue"] == ISSUE
    assert sha256_file(panel_path) == extraction_manifest["panel"]["sha256"]

    panel = pl.read_parquet(panel_path)
    assert panel["panel_row"].to_list() == list(range(panel.height))
    pair_panel = panel.filter(pl.col("consequence_cre").is_in(PAIR_CLASSES))
    selected_rows = pair_panel["panel_row"].to_numpy().astype(np.int64)
    assert np.all(selected_rows[:-1] < selected_rows[1:])
    labels = pair_panel["consequence_cre"].to_numpy()
    split = pair_panel["split"].to_numpy()
    blocks = pair_panel["block_id"].to_numpy()
    assert set(np.unique(labels)) == set(PAIR_CLASSES)
    assert set(np.unique(split)) == {"discovery", "validation", "test"}
    for class_name in PAIR_CLASSES:
        for split_name, expected in (
            ("discovery", 256),
            ("validation", 128),
            ("test", 128),
        ):
            assert np.sum((labels == class_name) & (split == split_name)) == expected
    test = np.flatnonzero(split == "test")
    assert len(np.unique(blocks[test][labels[test] == POSITIVE_CLASS])) == 5
    assert len(np.unique(blocks[test][labels[test] == NEGATIVE_CLASS])) == 5

    result_rows: list[dict[str, Any]] = []
    for block_index in BLOCK_INDICES:
        for budget in BUDGETS:
            arm = arm_label(block_index, budget)
            paths = {
                "forward": extraction_dir / arm / "sae_activations_forward.parquet",
                "reverse_complement": extraction_dir
                / arm
                / "sae_activations_reverse_complement.parquet",
            }
            for path in paths.values():
                relative = str(path.relative_to(extraction_dir))
                assert relative in extraction_manifest["artifacts"]
                assert (
                    sha256_file(path)
                    == extraction_manifest["artifacts"][relative]["sha256"]
                )
            forward = load_selected_activations(
                paths["forward"], selected_rows=selected_rows, columns=D_SAE
            )
            reverse = load_selected_activations(
                paths["reverse_complement"],
                selected_rows=selected_rows,
                columns=D_SAE,
            )
            views = make_views(forward.delta, reverse.delta)
            for view_name, matrix in views.items():
                selection, scores, positive = select_feature(
                    matrix, labels, split, POSITIVE_CLASS
                )
                assert positive.sum() == 128 and (~positive).sum() == 128
                confidence = bootstrap_pairwise_metrics(
                    scores,
                    positive,
                    blocks[test],
                    seed=RANDOM_SEED
                    + block_index * 100_000
                    + budget // 1_000
                    + sum(map(ord, view_name)),
                )
                result_rows.append(
                    {
                        "arm": arm,
                        "reported_block": block_index + 1,
                        "block_index": block_index,
                        "budget": budget,
                        "view": view_name,
                        **selection,
                        "test_auroc": float(roc_auc_score(positive, scores)),
                        **confidence,
                    }
                )
            print(json.dumps({"stage": "arm", "arm": arm}), flush=True)
            del forward, reverse, views

    metrics = pl.DataFrame(result_rows).sort(["view", "budget", "reported_block"])
    assert metrics.height == len(BLOCK_INDICES) * len(BUDGETS) * 4
    output_dir.mkdir(parents=True)
    metrics.write_parquet(output_dir / "missense_vs_synonymous.parquet")
    plot_pairwise(metrics, output_dir)
    best_signed = (
        metrics.filter(pl.col("view") == "signed_mean")
        .sort(
            ["test_auroc", "reported_block", "budget"],
            descending=[True, False, False],
        )
        .row(0, named=True)
    )
    result = {
        "analysis_commit": analysis_commit,
        "issue": ISSUE,
        "extraction_manifest_sha256": sha256_file(extraction_manifest_path),
        "panel_sha256": sha256_file(panel_path),
        "rows": pair_panel.height,
        "positive_class": POSITIVE_CLASS,
        "negative_class": NEGATIVE_CLASS,
        "chance": PAIRWISE_CHANCE,
        "primary_metric": "held-out missense-vs-synonymous AUROC",
        "best_signed_mean_arm": best_signed,
        "protocol": {
            "views": [
                "forward_signed",
                "reverse_complement_signed",
                "signed_mean",
                "max_abs",
            ],
            "block_bootstraps": BOOTSTRAPS,
            "bootstrap_scheme": "binary-label-stratified genomic-block resampling",
            "random_seed": RANDOM_SEED,
            "feature_selection": "absolute discovery Welch t among eligible features; sign fixed on discovery; validation chooses among top 64; test untouched",
            "status": "exploratory follow-up specified after the registered 35-class analysis",
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
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = analyze_pairwise(
        extraction_dir=args.extraction_dir,
        extraction_manifest_path=args.extraction_manifest,
        panel_path=args.panel,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
