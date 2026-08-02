from __future__ import annotations

import json

from extract_multilayer import (
    BLOCK_INDICES,
    D_SAE,
    MODEL_ID,
    MODEL_REVISION,
    TRAINING_TOKENS,
    arm_label,
    read_model_provenance,
)


def test_arm_labels_are_first_middle_final_25m() -> None:
    assert [arm_label(index) for index in BLOCK_INDICES] == [
        "block01-25m",
        "block10-25m",
        "block19-25m",
    ]


def test_model_provenance_checks_layer_and_exact_budget(tmp_path) -> None:
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
    result = read_model_provenance(tmp_path, block_index=18)
    assert result["training_tokens"] == TRAINING_TOKENS
    assert result["metadata"]["report_block"] == 19
