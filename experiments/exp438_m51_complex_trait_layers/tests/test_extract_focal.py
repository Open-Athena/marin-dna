from __future__ import annotations

import json

import numpy as np
import pyarrow as pa

from common import D_SAE, MODEL_ID, MODEL_REVISION, TRAINING_TOKENS, sha256_file
from extract_focal import (
    FOCAL_INDEX,
    SPARSE_SCHEMA,
    arm_label,
    model_path,
    read_model_provenance,
    sparse_union_table,
    variant_sequences,
)


def test_arm_labels_cover_first_middle_final() -> None:
    assert {arm_label(index) for index in (0, 9, 18)} == {
        "block01-25m",
        "block10-25m",
        "block19-25m",
    }


def test_model_path_uses_one_staging_root(tmp_path) -> None:
    path = tmp_path / "block10-25m"
    path.mkdir()
    assert model_path(block_index=9, models_root=tmp_path) == path


def test_variant_sequences_changes_only_center() -> None:
    reference = "A" * FOCAL_INDEX + "C" + "G" * FOCAL_INDEX
    ref, alt = variant_sequences(reference, "C", "T")
    assert ref == reference and alt[FOCAL_INDEX] == "T"
    assert sum(a != b for a, b in zip(ref, alt, strict=True)) == 1


def test_sparse_union_preserves_ref_alt_and_zero_delta() -> None:
    ref = np.array([[0, 2, 3, 0], [1, 0, 0, 0]], dtype=np.float32)
    alt = np.array([[0, 0, 3, 4], [0, 5, 0, 0]], dtype=np.float32)
    table = sparse_union_table(ref, alt, np.array([7, 9], dtype=np.uint32))
    assert table.schema == SPARSE_SCHEMA and table.num_rows == 5
    same = next(
        row
        for row in table.to_pylist()
        if row["panel_row"] == 7 and row["feature_id"] == 2
    )
    assert same["ref_activation"] == same["alt_activation"] == 3
    assert same["delta"] == 0
    assert pa.types.is_float32(table.schema.field("delta").type)


def test_read_model_provenance_checks_layer_and_budget(tmp_path) -> None:
    metadata = {
        "model_name": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "block_index": 18,
        "report_block": 19,
        "training_tokens": TRAINING_TOKENS,
    }
    cfg = {
        "metadata": metadata,
        "d_in": 1920,
        "d_sae": D_SAE,
        "architecture": "jumprelu",
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
        tmp_path, block_index=18, expected_artifacts=expected
    )
    assert result["training_tokens"] == TRAINING_TOKENS
    assert result["metadata"]["report_block"] == 19
