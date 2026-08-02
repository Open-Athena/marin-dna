from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa

from extract_common import (
    D_SAE,
    MODEL_ID,
    MODEL_REVISION,
    SPARSE_SCHEMA,
    TRAINING_TOKENS,
    arm_label,
    read_model_provenance,
    sha256_file,
    sparse_activation_table,
)


def test_sparse_activation_table_is_ordered_and_omits_zeros() -> None:
    features = np.zeros((2, D_SAE), dtype=np.float32)
    features[0, 3] = 1.5
    features[1, 2] = 2.0
    features[1, 7] = 3.0
    table = sparse_activation_table(features, np.array([7, 9], dtype=np.uint32))
    assert table.schema == SPARSE_SCHEMA
    assert table.to_pylist() == [
        {"context_id": 7, "feature_id": 3, "activation": 1.5},
        {"context_id": 9, "feature_id": 2, "activation": 2.0},
        {"context_id": 9, "feature_id": 7, "activation": 3.0},
    ]
    assert table["activation"].type == pa.float32()


def test_model_provenance_validates_exact_files(tmp_path: Path) -> None:
    block_index = 9
    cfg = {
        "architecture": "jumprelu",
        "d_in": 1920,
        "d_sae": D_SAE,
        "metadata": {
            "model_name": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "block_index": block_index,
            "report_block": block_index + 1,
            "training_tokens": TRAINING_TOKENS,
        },
    }
    runner = {
        "model_name": MODEL_ID,
        "model_from_pretrained_kwargs": {"revision": MODEL_REVISION},
        "training_tokens": TRAINING_TOKENS,
        "sae": {"d_sae": D_SAE},
    }
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    (tmp_path / "runner_cfg.json").write_text(json.dumps(runner))
    (tmp_path / "sae_weights.safetensors").write_bytes(b"weights")
    (tmp_path / "sparsity.safetensors").write_bytes(b"sparsity")
    expected = {
        name: (path.stat().st_size, sha256_file(path))
        for name in (
            "cfg.json",
            "runner_cfg.json",
            "sae_weights.safetensors",
            "sparsity.safetensors",
        )
        if (path := tmp_path / name)
    }
    observed = read_model_provenance(
        tmp_path, block_index=block_index, expected_artifacts=expected
    )
    assert observed["architecture"] == "jumprelu"
    assert observed["d_sae"] == D_SAE
    assert (
        observed["files"]["sae_weights.safetensors"]["sha256"]
        == expected["sae_weights.safetensors"][1]
    )


def test_arm_labels_are_reported_layers() -> None:
    assert [arm_label(index) for index in (0, 9, 18)] == [
        "block01-25m",
        "block10-25m",
        "block19-25m",
    ]
