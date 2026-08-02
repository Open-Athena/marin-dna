from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from analyze_reference_capacity import load_sparse, verify_archive
from extract_common import D_SAE, EXTRACTION_RUN_ID, sha256_file


def test_load_sparse_preserves_exact_context_feature_values(tmp_path: Path) -> None:
    path = tmp_path / "activations.parquet"
    table = pa.table(
        {
            "context_id": pa.array([0, 0, 2], type=pa.uint32()),
            "feature_id": pa.array([1, 5, 3], type=pa.uint32()),
            "activation": pa.array([1.5, 2.0, 4.0], type=pa.float32()),
        }
    )
    pq.write_table(table, path)
    matrix = load_sparse(path)
    assert matrix.shape[1] == D_SAE and matrix.nnz == 3
    assert matrix[0, 1] == 1.5
    assert matrix[0, 5] == 2.0
    assert matrix[2, 3] == 4.0


def test_verify_archive_checks_every_declared_payload(tmp_path: Path) -> None:
    payload = tmp_path / "payload.txt"
    payload.write_text("pinned\n")
    manifest = {
        "issue": 435,
        "run_id": EXTRACTION_RUN_ID,
        "analysis_status": "frozen_reference_sae_extraction",
        "artifacts": {
            payload.name: {
                "bytes": payload.stat().st_size,
                "sha256": sha256_file(payload),
            }
        },
    }
    manifest_path = tmp_path / "archive_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    observed = verify_archive(tmp_path, sha256_file(manifest_path))
    assert observed == manifest
