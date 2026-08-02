from __future__ import annotations

import json

import numpy as np
import polars as pl
import pyarrow as pa
from marin_dna.data.dna import reverse_complement

from extract_focal import (
    BUDGET,
    D_SAE,
    MODEL_ID,
    MODEL_REVISION,
    SPARSE_SCHEMA,
    arm_label,
    batch_sequences,
    model_path,
    read_model_provenance,
    sparse_union_table,
)


def test_arm_labels_cover_first_middle_final() -> None:
    assert {arm_label(index) for index in (0, 9, 18)} == {
        "block01-25m",
        "block10-25m",
        "block19-25m",
    }


def test_model_path_uses_one_25m_root(tmp_path) -> None:
    (tmp_path / "block10-25m").mkdir()
    assert model_path(block_index=9, models_root=tmp_path) == (tmp_path / "block10-25m")


def test_batch_sequences_interleaves_alleles_and_reverse_complements() -> None:
    frame = pl.DataFrame(
        {
            "ref_sequence": ["A" * 255, "C" * 255],
            "alt_sequence": ["G" * 255, "T" * 255],
        }
    )
    assert batch_sequences(frame, offset=0, length=2, orientation="forward") == [
        "A" * 255,
        "G" * 255,
        "C" * 255,
        "T" * 255,
    ]
    assert batch_sequences(
        frame, offset=0, length=1, orientation="reverse_complement"
    ) == [reverse_complement("A" * 255), reverse_complement("G" * 255)]


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
        "training_tokens": BUDGET,
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
        "training_tokens": BUDGET,
        "sae": {"d_sae": D_SAE},
    }
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    (tmp_path / "runner_cfg.json").write_text(json.dumps(runner))
    (tmp_path / "sae_weights.safetensors").write_bytes(b"weights")
    (tmp_path / "sparsity.safetensors").write_bytes(b"sparsity")
    result = read_model_provenance(tmp_path, block_index=18)
    assert result["training_tokens"] == BUDGET
    assert result["metadata"]["report_block"] == 19
