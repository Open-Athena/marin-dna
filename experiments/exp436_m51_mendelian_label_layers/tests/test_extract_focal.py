from __future__ import annotations

import json

import numpy as np
import pyarrow as pa

from extract_focal import (
    D_SAE,
    FOCAL_INDEX,
    MODEL_ID,
    MODEL_REVISION,
    SPARSE_SCHEMA,
    arm_label,
    model_path,
    read_model_provenance,
    sparse_union_table,
    variant_sequences,
)


def test_arm_labels_cover_first_middle_final_and_budgets() -> None:
    assert {
        arm_label(block_index, budget)
        for block_index in (0, 9, 18)
        for budget in (5_000_550, 25_000_200)
    } == {
        "block01-5m",
        "block01-25m",
        "block10-5m",
        "block10-25m",
        "block19-5m",
        "block19-25m",
    }


def test_model_path_routes_only_block01_to_new_root(tmp_path) -> None:
    block01 = tmp_path / "block01"
    existing = tmp_path / "existing"
    (block01 / "block01-5m").mkdir(parents=True)
    (existing / "block10-5m").mkdir(parents=True)
    assert (
        model_path(
            block_index=0,
            budget=5_000_550,
            block01_models_root=block01,
            existing_models_root=existing,
        )
        == block01 / "block01-5m"
    )
    assert (
        model_path(
            block_index=9,
            budget=5_000_550,
            block01_models_root=block01,
            existing_models_root=existing,
        )
        == existing / "block10-5m"
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
        "block_index": 18,
        "report_block": 19,
        "training_tokens": budget,
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
        "training_tokens": budget,
        "sae": {"d_sae": D_SAE},
    }
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    (tmp_path / "runner_cfg.json").write_text(json.dumps(runner))
    (tmp_path / "sae_weights.safetensors").write_bytes(b"weights")
    (tmp_path / "sparsity.safetensors").write_bytes(b"sparsity")
    result = read_model_provenance(tmp_path, block_index=18, budget=budget)
    assert result["training_tokens"] == budget
    assert result["metadata"]["report_block"] == 19
