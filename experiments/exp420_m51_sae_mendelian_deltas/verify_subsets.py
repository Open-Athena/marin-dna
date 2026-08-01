"""Independently verify retrieved issue 420 extraction and subset artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl

from prediction_primitives import D_SAE, sha256

EXPECTED_PANEL_ROWS = 16_140
EXPECTED_BINARY_ROWS = 6
EXPECTED_CONTEXT_ROWS = 60
EXPECTED_MULTICLASS_ROWS = 6
EXPECTED_MULTICLASS_SELECTIONS = 16


def verify_manifest_files(directory: Path, manifest: dict[str, Any]) -> None:
    for name, metadata in manifest["artifacts"].items():
        path = directory / name
        assert path.is_file(), path
        assert path.stat().st_size == metadata["bytes"], path
        assert sha256(path) == metadata["sha256"], path


def verify_sparse(path: Path, expected_rows: int) -> dict[str, int]:
    frame = pl.read_parquet(path)
    assert frame.height == expected_rows
    assert frame.null_count().sum_horizontal().sum() == 0
    assert frame["row_index"].n_unique() == EXPECTED_PANEL_ROWS
    assert frame["row_index"].min() == 0
    assert frame["row_index"].max() == EXPECTED_PANEL_ROWS - 1
    assert frame["feature_id"].min() >= 0
    assert frame["feature_id"].max() < D_SAE
    assert frame.select(pl.struct("row_index", "feature_id").n_unique()).item() == (
        frame.height
    )
    assert frame.filter(
        (pl.col("delta") - (pl.col("alt_activation") - pl.col("ref_activation")))
        .abs()
        .gt(0)
    ).is_empty()
    assert frame.filter(
        (pl.col("ref_activation") < 0) | (pl.col("alt_activation") < 0)
    ).is_empty()
    return {
        "rows": frame.height,
        "variants": frame["row_index"].n_unique(),
        "features": frame["feature_id"].n_unique(),
    }


def verify_retrieval(
    *, directory: Path, panel_path: Path, expected_commit: str
) -> dict[str, Any]:
    assert directory.is_dir() and panel_path.is_file()
    assert len(expected_commit) == 40
    extraction_dir = directory / "extraction"
    analysis_dir = directory / "subset-analysis"
    extraction_manifest_path = extraction_dir / "manifest.json"
    analysis_manifest_path = analysis_dir / "manifest.json"
    extraction_manifest = json.loads(extraction_manifest_path.read_text())
    analysis_manifest = json.loads(analysis_manifest_path.read_text())
    assert extraction_manifest["experiment_commit"] == expected_commit
    assert analysis_manifest["experiment_commit"] == expected_commit
    assert extraction_manifest["panel"]["sha256"] == sha256(panel_path)
    assert analysis_manifest["panel"]["sha256"] == sha256(panel_path)
    verify_manifest_files(extraction_dir, extraction_manifest)
    verify_manifest_files(analysis_dir, analysis_manifest)
    assert analysis_manifest["extraction"]["manifest"] == extraction_manifest
    assert analysis_manifest["extraction"]["manifest_sha256"] == sha256(
        extraction_manifest_path
    )

    expected_sparse = {
        row["orientation"]: row for row in extraction_manifest["outputs"]
    }
    assert set(expected_sparse) == {"forward", "reverse_complement"}
    sparse = {
        orientation: verify_sparse(extraction_dir / metadata["path"], metadata["rows"])
        for orientation, metadata in expected_sparse.items()
    }
    contexts = pl.read_parquet(extraction_dir / "variant_contexts.parquet")
    assert contexts.height == contexts["row_index"].n_unique() == EXPECTED_PANEL_ROWS
    assert contexts.sort("row_index")["row_index"].to_list() == list(
        range(EXPECTED_PANEL_ROWS)
    )
    assert contexts.null_count().sum_horizontal().sum() == 0

    binary = pl.read_parquet(analysis_dir / "binary_summary.parquet")
    assert binary.height == EXPECTED_BINARY_ROWS
    assert set(binary["label"]) == {0, 1}
    assert set(binary["view"]) == {
        "forward",
        "reverse_complement",
        "aggregate",
    }
    for column in (
        "auc",
        "auc_ci95_low",
        "auc_ci95_high",
        "substitution_conditional_auc",
        "substitution_gc_conditional_auc",
    ):
        assert binary[column].is_finite().all()
        assert binary[column].min() >= 0 and binary[column].max() <= 1
    assert binary.filter(
        (pl.col("auc_ci95_low") > pl.col("auc"))
        | (pl.col("auc") > pl.col("auc_ci95_high"))
    ).is_empty()

    binary_contexts = pl.read_parquet(analysis_dir / "binary_contexts.parquet")
    assert binary_contexts.height == EXPECTED_CONTEXT_ROWS
    assert binary_contexts.null_count().sum_horizontal().sum() == 0
    support = pl.read_parquet(analysis_dir / "class_support.parquet")
    assert support["subset"].n_unique() == 8
    assert support["split"].n_unique() == 3
    assert support.height == 24
    multiclass = pl.read_parquet(analysis_dir / "multiclass_summary.parquet")
    assert multiclass.height == EXPECTED_MULTICLASS_ROWS
    assert set(multiclass["label"]) == {0, 1}
    assert set(multiclass["view"]) == {
        "forward",
        "reverse_complement",
        "aggregate",
    }
    for column in (
        "macro_f1",
        "macro_f1_ci95_low",
        "macro_f1_ci95_high",
        "balanced_accuracy",
        "accuracy",
    ):
        assert multiclass[column].is_finite().all()
        assert multiclass[column].min() >= 0 and multiclass[column].max() <= 1
    assert multiclass.filter(
        (pl.col("macro_f1_ci95_low") > pl.col("macro_f1"))
        | (pl.col("macro_f1") > pl.col("macro_f1_ci95_high"))
    ).is_empty()
    selections = pl.read_parquet(analysis_dir / "multiclass_selections.parquet")
    assert selections.height == EXPECTED_MULTICLASS_SELECTIONS
    assert selections["target_class"].n_unique() == 4

    return {
        "expected_commit": expected_commit,
        "panel_sha256": sha256(panel_path),
        "extraction_manifest_sha256": sha256(extraction_manifest_path),
        "analysis_manifest_sha256": sha256(analysis_manifest_path),
        "sparse": sparse,
        "binary_rows": binary.height,
        "multiclass_rows": multiclass.height,
        "artifact_bytes": sum(
            path.stat().st_size for path in directory.rglob("*") if path.is_file()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            verify_retrieval(
                directory=args.directory,
                panel_path=args.panel,
                expected_commit=args.expected_commit,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
