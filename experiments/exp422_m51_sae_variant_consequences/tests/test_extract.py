from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import torch

from extract import (
    FOCAL_INDEX,
    MODEL_ID,
    MODEL_REVISION,
    SPARSE_SCHEMA,
    encode_sae_features,
    read_sae_provenance,
    sparse_union_table,
    variant_sequences,
)


def test_variant_sequences_changes_only_the_center() -> None:
    reference = "A" * FOCAL_INDEX + "C" + "G" * FOCAL_INDEX
    ref_sequence, alt_sequence = variant_sequences(reference, "C", "T")
    assert ref_sequence == reference
    assert alt_sequence[FOCAL_INDEX] == "T"
    assert sum(a != b for a, b in zip(ref_sequence, alt_sequence, strict=True)) == 1


def test_sparse_union_table_preserves_ref_alt_and_zero_delta_entries() -> None:
    ref = np.array([[0, 2, 3, 0], [1, 0, 0, 0]], dtype=np.float32)
    alt = np.array([[0, 0, 3, 4], [0, 5, 0, 0]], dtype=np.float32)
    table = sparse_union_table(ref, alt, np.array([7, 9], dtype=np.uint32))

    assert table.schema == SPARSE_SCHEMA
    assert table.num_rows == 5
    rows = table.to_pylist()
    assert {row["feature_id"] for row in rows if row["panel_row"] == 7} == {1, 2, 3}
    same = next(row for row in rows if row["panel_row"] == 7 and row["feature_id"] == 2)
    assert same == {
        "panel_row": 7,
        "feature_id": 2,
        "ref_activation": 3.0,
        "alt_activation": 3.0,
        "delta": 0.0,
    }
    assert pa.types.is_float32(table.schema.field("delta").type)


def test_read_sae_provenance_uses_artifact_metadata(tmp_path) -> None:
    metadata = {
        "model_name": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "block_index": 9,
        "report_block": 10,
        "seed": 77,
        "training_architecture": "batchtopk",
    }
    cfg = {
        "metadata": metadata,
        "d_in": 1920,
        "d_sae": 3840,
        "architecture": "jumprelu",
    }
    runner = {
        "model_name": MODEL_ID,
        "model_from_pretrained_kwargs": {"revision": MODEL_REVISION},
        "seed": 77,
        "training_tokens": 123_456,
        "sae": {"d_in": 1920, "d_sae": 3840},
    }
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    (tmp_path / "runner_cfg.json").write_text(json.dumps(runner))
    (tmp_path / "sae_weights.safetensors").write_bytes(b"weights")
    (tmp_path / "sparsity.safetensors").write_bytes(b"sparsity")

    provenance = read_sae_provenance(tmp_path, block_index=9)

    assert provenance["seed"] == 77
    assert provenance["training_tokens"] == 123_456
    assert provenance["d_sae"] == 3840
    assert set(provenance["files"]) == {
        "cfg.json",
        "runner_cfg.json",
        "sae_weights.safetensors",
        "sparsity.safetensors",
    }


class _TinySAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(3, 5)

    def encode(self, raw: torch.Tensor) -> torch.Tensor:
        return self.projection(raw)


def test_encode_sae_features_does_not_build_autograd_graph() -> None:
    sae = _TinySAE().requires_grad_(False).eval()
    raw = torch.ones((2, 3))

    features = encode_sae_features(sae, raw)

    assert features.shape == (2, 5)
    assert not features.requires_grad
