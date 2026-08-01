from __future__ import annotations

import json

import numpy as np
import pyarrow as pa

from extract import (
    FOCAL_INDEX,
    MODEL_ID,
    MODEL_REVISION,
    SPARSE_SCHEMA,
    read_model_provenance,
    sparse_union_table,
    variant_sequences,
)


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
    budget = 5_000_550
    metadata = {
        "model_name": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "block_index": 3,
        "report_block": 4,
        "training_tokens": budget,
    }
    cfg = {
        "metadata": metadata,
        "d_in": 1920,
        "d_sae": 15360,
        "architecture": "jumprelu",
    }
    runner = {
        "model_name": MODEL_ID,
        "model_from_pretrained_kwargs": {"revision": MODEL_REVISION},
        "training_tokens": budget,
        "sae": {"d_sae": 15360},
    }
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    (tmp_path / "runner_cfg.json").write_text(json.dumps(runner))
    (tmp_path / "sae_weights.safetensors").write_bytes(b"weights")
    (tmp_path / "sparsity.safetensors").write_bytes(b"sparsity")
    result = read_model_provenance(tmp_path, block_index=3, budget=budget)
    assert result["training_tokens"] == budget
    assert result["metadata"]["report_block"] == 4
