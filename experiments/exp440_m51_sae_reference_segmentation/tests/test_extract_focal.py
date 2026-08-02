from __future__ import annotations

import hashlib
import json

import numpy as np
import polars as pl
import pyarrow as pa
from marin_dna.data.dna import reverse_complement

from extract_focal import (
    D_SAE,
    MODEL_ID,
    MODEL_REVISION,
    SPARSE_SCHEMA,
    TRAINING_TOKENS,
    arm_label,
    oriented_sequences,
    read_model_provenance,
    sha256_file,
    sparse_activation_table,
)


def test_arm_labels_cover_first_middle_final() -> None:
    assert [arm_label(index) for index in (0, 9, 18)] == [
        "block01-25m",
        "block10-25m",
        "block19-25m",
    ]


def test_oriented_sequences_preserves_order() -> None:
    frame = pl.DataFrame({"sequence": ["A" * 255, "CG" + "T" * 253]})
    assert (
        oriented_sequences(frame, offset=0, length=2, orientation="forward")
        == frame["sequence"].to_list()
    )
    assert oriented_sequences(
        frame, offset=1, length=1, orientation="reverse_complement"
    ) == [reverse_complement("CG" + "T" * 253)]


def test_sparse_activation_table_is_ordered_and_omits_zeros() -> None:
    features = np.zeros((2, D_SAE), dtype=np.float32)
    features[0, 3] = 1.5
    features[1, 2] = 2.0
    features[1, 7] = 3.0
    table = sparse_activation_table(features, np.array([7, 9], dtype=np.uint32))
    assert table.schema == SPARSE_SCHEMA
    assert table.to_pylist() == [
        {"panel_row": 7, "feature_id": 3, "activation": 1.5},
        {"panel_row": 9, "feature_id": 2, "activation": 2.0},
        {"panel_row": 9, "feature_id": 7, "activation": 3.0},
    ]
    assert table["activation"].type == pa.float32()


def test_read_model_provenance_checks_exact_files(tmp_path) -> None:
    block_index = 18
    cfg = {
        "architecture": "jumprelu",
        "d_in": 1_920,
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
    result = read_model_provenance(
        tmp_path, block_index=block_index, expected_artifacts=expected
    )
    assert result["training_tokens"] == TRAINING_TOKENS
    assert result["metadata"]["report_block"] == 19
    assert (
        result["files"]["sae_weights.safetensors"]["sha256"]
        == hashlib.sha256(b"weights").hexdigest()
    )
