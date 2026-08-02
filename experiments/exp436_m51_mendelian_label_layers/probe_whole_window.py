"""Run the official nested chromosome-held-out probe on pooled SAE codes."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import polars as pl
import sklearn
from marin_dna.pipelines.evals.metrics import per_chrom_ap_table
from marin_dna.pipelines.evals.variant_probe import (
    DEFAULT_C_GRID,
    fit_full_classifier,
    summarize_selected_c,
    traitgym_nested_oof,
)
from sklearn.metrics import average_precision_score

from extract_focal import (
    D_SAE,
    EXPECTED_ROWS,
    ISSUE,
    ORIENTATIONS,
    sha256_file,
    validate_panel,
    write_json,
)
from extract_whole_window import POOLING, matrix_relative
from train import MARIN_DNA_REVISION, assert_commit

ARM = "block19-25m"
FEATURE_COMBO = "concat_ref_delta"
MIN_VARIANTS = 300
MIN_CHROMS = 3
INNER_SPLITS = 5
SCORE_COLUMNS = (
    "sparse_probe_score",
    "official_probe_score",
    "minus_llr_avg",
)


def nonconstant_components(
    ref: np.ndarray, alt: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return label-blind masks for nonconstant ref and delta columns."""

    assert ref.shape == alt.shape and ref.ndim == 2
    ref_min = ref.min(axis=0)
    ref_max = ref.max(axis=0)
    delta = np.asarray(alt) - np.asarray(ref)
    delta_min = delta.min(axis=0)
    delta_max = delta.max(axis=0)
    ref_keep = ref_min != ref_max
    delta_keep = delta_min != delta_max
    assert ref_keep.any() and delta_keep.any()
    return ref_keep, delta_keep


def build_probe_feature(
    ref: np.ndarray, alt: np.ndarray
) -> tuple[np.ndarray, pl.DataFrame]:
    """Build float32 [ref, alt-ref] after algebraically inert column removal."""

    assert ref.shape == alt.shape and ref.ndim == 2
    assert ref.dtype == alt.dtype == np.float32
    ref_keep, delta_keep = nonconstant_components(ref, alt)
    ref_ids = np.flatnonzero(ref_keep)
    delta_ids = np.flatnonzero(delta_keep)
    delta = np.asarray(alt, dtype=np.float32) - np.asarray(ref, dtype=np.float32)
    features = np.concatenate(
        [np.asarray(ref[:, ref_keep]), delta[:, delta_keep]], axis=1
    ).astype(np.float32, copy=False)
    assert features.shape == (ref.shape[0], ref_ids.size + delta_ids.size)
    assert np.isfinite(features).all()
    mapping = pl.DataFrame(
        {
            "probe_column": np.arange(features.shape[1], dtype=np.uint32),
            "component": ["ref"] * ref_ids.size + ["delta"] * delta_ids.size,
            "feature_id": np.concatenate([ref_ids, delta_ids]).astype(np.uint32),
        }
    )
    return features, mapping


def run_cell(
    features: np.ndarray,
    labels: np.ndarray,
    chroms: np.ndarray,
    *,
    c_grid: np.ndarray,
    n_jobs: int,
) -> tuple[np.ndarray, dict[str, Any], Any]:
    predictions, selected = traitgym_nested_oof(
        features,
        labels,
        chroms,
        c_grid=c_grid,
        inner_splits=INNER_SPLITS,
        n_jobs=n_jobs,
    )
    classifier, full_c, c_scores = fit_full_classifier(
        features,
        labels,
        chroms,
        c_grid=c_grid,
        inner_splits=INNER_SPLITS,
        n_jobs=n_jobs,
    )
    summary = summarize_selected_c(selected, full_c, c_grid, full_c_scores=c_scores)
    assert np.isfinite(predictions).all()
    return predictions, summary, classifier


def metric_table(predictions: pd.DataFrame, orientation: str) -> pd.DataFrame:
    assert orientation in ORIENTATIONS
    subset_rows = predictions[
        predictions["sparse_probe_score"].notna()
        & predictions["official_probe_score"].notna()
    ].copy()
    assert len(subset_rows) > 0
    subset_metrics = per_chrom_ap_table(
        subset_rows,
        list(SCORE_COLUMNS),
        n_bootstrap=1_000,
        rng=0,
        n_min=30,
    )
    subset_metrics["scope"] = "per_subset"

    raw_matched = predictions[predictions["official_probe_score"].notna()].copy()
    raw_matched["metric_subset"] = "overall"
    global_matched = per_chrom_ap_table(
        raw_matched,
        ["sparse_global_probe_score", "official_probe_score", "minus_llr_avg"],
        subset_col="metric_subset",
        n_bootstrap=1_000,
        rng=0,
        n_min=30,
    )
    global_matched = global_matched[global_matched["subset"] == "overall"].copy()
    global_matched["scope"] = "global_raw_matched"

    all_rows = predictions.copy()
    all_rows["metric_subset"] = "overall"
    global_all = per_chrom_ap_table(
        all_rows,
        ["sparse_global_probe_score", "minus_llr_avg"],
        subset_col="metric_subset",
        n_bootstrap=1_000,
        rng=0,
        n_min=30,
    )
    global_all = global_all[global_all["subset"] == "overall"].copy()
    global_all["scope"] = "global_all_rows"

    result = pd.concat([subset_metrics, global_matched, global_all], ignore_index=True)
    result["orientation"] = orientation
    return result


def pooled_diagnostics(predictions: pd.DataFrame, orientation: str) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes = {
        "global_raw_matched": predictions["official_probe_score"].notna(),
        "global_all_rows": np.ones(len(predictions), dtype=bool),
    }
    for scope, mask in scopes.items():
        frame = predictions.loc[mask]
        columns = ["sparse_global_probe_score", "minus_llr_avg"]
        if scope == "global_raw_matched":
            columns.append("official_probe_score")
        for column in columns:
            assert frame[column].notna().all()
            rows.append(
                {
                    "orientation": orientation,
                    "scope": scope,
                    "score_type": column,
                    "pooled_auprc_diagnostic": average_precision_score(
                        frame["label"], frame[column]
                    ),
                    "n": len(frame),
                    "n_positive": int(frame["label"].sum()),
                }
            )
    return pl.DataFrame(rows)


def verify_target_matrices(root: Path, manifest: dict[str, Any]) -> None:
    assert manifest["protocol"]["pooling"] == POOLING
    assert manifest["protocol"]["matrix_dtype"] == "float32"
    for orientation in ORIENTATIONS:
        for allele in ("ref", "alt"):
            relative = matrix_relative(ARM, orientation, allele)
            expected = manifest["artifacts"][str(relative)]
            path = root / relative
            assert path.is_file() and path.stat().st_size == expected["bytes"]
            assert sha256_file(path) == expected["sha256"]


def probe(
    *,
    panel_path: Path,
    panel_manifest_path: Path,
    extraction_root: Path,
    output_dir: Path,
    n_jobs: int,
    c_grid: np.ndarray = DEFAULT_C_GRID,
) -> dict[str, Any]:
    assert not output_dir.exists() and n_jobs > 0
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert_commit(experiment_commit)
    started = time.monotonic()

    panel_manifest = json.loads(panel_manifest_path.read_text())
    extraction_manifest = json.loads((extraction_root / "manifest.json").read_text())
    panel_pl = pl.read_parquet(panel_path)
    validate_panel(panel_pl, panel_manifest, panel_path)
    assert extraction_manifest["panel"]["sha256"] == sha256_file(panel_path)
    verify_target_matrices(extraction_root, extraction_manifest)
    panel = panel_pl.to_pandas()
    panel.insert(0, "panel_row", np.arange(EXPECTED_ROWS, dtype=np.uint32))
    labels = panel["label"].to_numpy(dtype=np.uint8)
    chroms = panel["chrom"].astype(str).to_numpy()

    output_dir.mkdir(parents=True)
    artifacts: dict[str, Any] = {}
    summaries: dict[str, Any] = {}
    all_metrics: list[pl.DataFrame] = []
    all_diagnostics: list[pl.DataFrame] = []
    for orientation in ORIENTATIONS:
        ref = np.load(
            extraction_root / matrix_relative(ARM, orientation, "ref"),
            mmap_mode="r",
        )
        alt = np.load(
            extraction_root / matrix_relative(ARM, orientation, "alt"),
            mmap_mode="r",
        )
        assert ref.shape == alt.shape == (EXPECTED_ROWS, D_SAE)
        features, mapping = build_probe_feature(ref, alt)
        mapping_path = output_dir / f"active_components_{orientation}.parquet"
        mapping.write_parquet(mapping_path, compression="zstd")
        artifacts[mapping_path.name] = {
            "bytes": mapping_path.stat().st_size,
            "sha256": sha256_file(mapping_path),
        }

        subset_predictions = np.full(EXPECTED_ROWS, np.nan, dtype=np.float64)
        cell_summaries: dict[str, Any] = {}
        classifiers: dict[str, Any] = {}
        for subset in sorted(panel["subset"].astype(str).unique()):
            mask = panel["subset"].astype(str).to_numpy() == subset
            n = int(mask.sum())
            n_chrom = int(np.unique(chroms[mask]).size)
            n_positive = int(labels[mask].sum())
            if n < MIN_VARIANTS or n_chrom < MIN_CHROMS:
                cell_summaries[subset] = {
                    "status": "skipped_support",
                    "n": n,
                    "n_positive": n_positive,
                    "n_chrom": n_chrom,
                }
                continue
            predictions, c_summary, classifier = run_cell(
                features[mask],
                labels[mask],
                chroms[mask],
                c_grid=c_grid,
                n_jobs=n_jobs,
            )
            subset_predictions[mask] = predictions
            cell_summaries[subset] = {
                "status": "complete",
                "n": n,
                "n_positive": n_positive,
                "n_chrom": n_chrom,
                "c_summary": c_summary,
            }
            classifiers[subset] = classifier

        global_predictions, global_c_summary, global_classifier = run_cell(
            features,
            labels,
            chroms,
            c_grid=c_grid,
            n_jobs=n_jobs,
        )
        classifiers["__global__"] = global_classifier
        classifier_path = output_dir / f"classifiers_{orientation}.joblib"
        joblib.dump(classifiers, classifier_path, compress=3)
        artifacts[classifier_path.name] = {
            "bytes": classifier_path.stat().st_size,
            "sha256": sha256_file(classifier_path),
        }

        prediction_frame = panel[
            [
                "panel_row",
                "chrom",
                "pos",
                "ref",
                "alt",
                "label",
                "subset",
                "match_group",
            ]
        ].copy()
        prediction_frame["sparse_probe_score"] = subset_predictions
        prediction_frame["sparse_global_probe_score"] = global_predictions
        prediction_frame["official_probe_score"] = panel["probe_score"].to_numpy()
        prediction_frame["minus_llr_avg"] = panel["minus_llr_avg"].to_numpy()
        prediction_path = output_dir / f"predictions_{orientation}.parquet"
        pl.from_pandas(prediction_frame).write_parquet(
            prediction_path, compression="zstd"
        )
        artifacts[prediction_path.name] = {
            "bytes": prediction_path.stat().st_size,
            "sha256": sha256_file(prediction_path),
        }

        all_metrics.append(pl.from_pandas(metric_table(prediction_frame, orientation)))
        all_diagnostics.append(pooled_diagnostics(prediction_frame, orientation))
        summaries[orientation] = {
            "active_probe_columns": features.shape[1],
            "active_ref_features": mapping.filter(pl.col("component") == "ref").height,
            "active_delta_features": mapping.filter(
                pl.col("component") == "delta"
            ).height,
            "subset_cells": cell_summaries,
            "global": {
                "n": EXPECTED_ROWS,
                "n_positive": int(labels.sum()),
                "n_chrom": int(np.unique(chroms).size),
                "c_summary": global_c_summary,
            },
        }
        del features, ref, alt, classifiers

    metrics = pl.concat(all_metrics, how="diagonal_relaxed")
    metrics_path = output_dir / "metrics.parquet"
    metrics.write_parquet(metrics_path, compression="zstd")
    artifacts[metrics_path.name] = {
        "bytes": metrics_path.stat().st_size,
        "sha256": sha256_file(metrics_path),
    }
    diagnostics = pl.concat(all_diagnostics, how="vertical")
    diagnostics_path = output_dir / "pooled_auprc_diagnostics.parquet"
    diagnostics.write_parquet(diagnostics_path, compression="zstd")
    artifacts[diagnostics_path.name] = {
        "bytes": diagnostics_path.stat().st_size,
        "sha256": sha256_file(diagnostics_path),
    }

    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "issue": ISSUE,
        "run_id": os.environ.get("RUN_ID", ""),
        "experiment_commit": experiment_commit,
        "elapsed_seconds": time.monotonic() - started,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "polars": pl.__version__,
        "sklearn": sklearn.__version__,
        "input": {
            "extraction_run_id": extraction_manifest["run_id"],
            "extraction_manifest_sha256": sha256_file(
                extraction_root / "manifest.json"
            ),
            "panel_sha256": sha256_file(panel_path),
            "arm": ARM,
        },
        "protocol": {
            "pooling": POOLING,
            "feature_combo": FEATURE_COMBO,
            "orientations": list(ORIENTATIONS),
            "orientation_aggregation": None,
            "constant_column_removal": "label-blind across all variants",
            "probe": "StandardScaler -> LogisticRegression(L2)",
            "outer_cv": "leave one chromosome out",
            "inner_cv": "GroupKFold",
            "inner_splits": INNER_SPLITS,
            "c_grid": c_grid.tolist(),
            "minimum_variants": MIN_VARIANTS,
            "minimum_chromosomes": MIN_CHROMS,
            "n_jobs": n_jobs,
            "headline_metric": "per-chromosome-weighted AUPRC",
            "metric_bootstrap": "1,000 chromosome-cluster resamples",
            "pooled_auprc": "diagnostic only",
            "marin_dna_revision": MARIN_DNA_REVISION,
        },
        "summaries": summaries,
        "artifacts": artifacts,
    }
    write_json(output_dir / "results.json", result)
    result["artifacts"]["results.json"] = {
        "bytes": (output_dir / "results.json").stat().st_size,
        "sha256": sha256_file(output_dir / "results.json"),
    }
    write_json(output_dir / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--extraction-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()
    result = probe(
        panel_path=args.panel,
        panel_manifest_path=args.panel_manifest,
        extraction_root=args.extraction_root,
        output_dir=args.output_dir,
        n_jobs=args.n_jobs,
    )
    print(json.dumps(result["artifacts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
