"""Pinned model, SAE, panel, and sparse-output contracts for issue 435."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

ISSUE = 435
EXTRACTION_RUN_ID = "dna-exp435-repeat-reference-activations-r1"
PANEL_RUN_ID = "dna-exp435-repeat-reference-panel-r1"
PANEL_ARCHIVE_MANIFEST_SHA256 = (
    "420237266f074f154fea189281ecdcbb5893afc48d37c2db761e38dbed6d22f7"
)
MODEL_ID = "marin-dna/marin-dna-exp135-m5.1"
MODEL_REVISION = "c0676b2012b8b9c526deb26ff517f6b92b6d375d"
MARIN_DNA_REVISION = "c4c1c86bbfc0bd58ff76dda3bac1d2acea856a33"
SAELENS_REVISION = "8be14080485952f729ed58d674bcddf9778e0aa4"
D_SAE = 15_360
TRAINING_TOKENS = 25_000_200
CONTEXTS = 90_107
WINDOW_BP = 255
FOCAL_INDEX = 127
BLOCK_INDICES = (0, 9, 18)
HOOK_NAMES = tuple(f"model.layers.{index}" for index in BLOCK_INDICES)
ORIENTATIONS = ("forward", "reverse_complement")

SPARSE_SCHEMA = pa.schema(
    [
        pa.field("context_id", pa.uint32(), nullable=False),
        pa.field("feature_id", pa.uint32(), nullable=False),
        pa.field("activation", pa.float32(), nullable=False),
    ]
)

EXPECTED_SAE_ARTIFACTS = {
    "block01-25m": {
        "cfg.json": (
            1_286,
            "e96f615d346068a7f2b32343474bf2d08615aa9bedc167bf529984d12db71b4c",
        ),
        "runner_cfg.json": (
            6_184,
            "2373149c73663b87f3da77186ba46fc06fc4e4261617ad9f9833eb010f847d83",
        ),
        "sae_weights.safetensors": (
            236_060_560,
            "97b7c23c76abc1a38a45fd3dcea241285811c2e810bcc371c314af9bb332a353",
        ),
        "sparsity.safetensors": (
            61_520,
            "39f50d1bd29bb6ef1e3cc091271193a90e3625d07ebf120c04492d5398717098",
        ),
    },
    "block10-25m": {
        "cfg.json": (
            1_283,
            "cfc161c05921a787cdcd6a369c9416a00e356030d31d2c9c0a78b6bbdccd51b9",
        ),
        "runner_cfg.json": (
            12_049,
            "5082e318b95bdef98446556ff94b768dc1ca123da0df4a7bc79de71b065cf554",
        ),
        "sae_weights.safetensors": (
            236_060_560,
            "606b81e2cc34ad7225de0fbaf5e673e688c4f990fc748cb59223316893e826b6",
        ),
        "sparsity.safetensors": (
            61_520,
            "e6a2776c487d6a84de0fc0b5c093560611bb5252cc5f88cc09322f8d00c03082",
        ),
    },
    "block19-25m": {
        "cfg.json": (
            1_287,
            "8825220f296bea463f266bda9e0497be3ebcc956f8f846109178aaa45ff06848",
        ),
        "runner_cfg.json": (
            12_053,
            "d36576c21a33a4b64a507559266dff757156b5032a277dfbd68498fc3bfa62a8",
        ),
        "sae_weights.safetensors": (
            236_060_560,
            "e4f10ba59f10be943dbdc33f469f986f598c5e34fcba42577efad27717231533",
        ),
        "sparsity.safetensors": (
            61_520,
            "ef641aeb1be378356881a81563a9886d81ae0edc511d1ff5669d8ed71990d465",
        ),
    },
}

assert WINDOW_BP == 2 * FOCAL_INDEX + 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arm_label(block_index: int, budget: int = TRAINING_TOKENS) -> str:
    assert block_index in BLOCK_INDICES and budget == TRAINING_TOKENS
    return f"block{block_index + 1:02d}-{budget // 1_000_000}m"


def read_model_provenance(
    path: Path,
    *,
    block_index: int,
    expected_artifacts: dict[str, tuple[int, str]] | None = None,
) -> dict[str, Any]:
    paths = {
        name: path / name
        for name in (
            "cfg.json",
            "runner_cfg.json",
            "sae_weights.safetensors",
            "sparsity.safetensors",
        )
    }
    assert all(item.is_file() for item in paths.values())
    expected = expected_artifacts or EXPECTED_SAE_ARTIFACTS[arm_label(block_index)]
    assert set(expected) == set(paths)
    for name, item in paths.items():
        expected_bytes, expected_sha256 = expected[name]
        assert item.stat().st_size == expected_bytes
        assert sha256_file(item) == expected_sha256
    cfg = json.loads(paths["cfg.json"].read_text())
    runner = json.loads(paths["runner_cfg.json"].read_text())
    metadata = cfg["metadata"]
    assert metadata["model_name"] == MODEL_ID
    assert metadata["model_revision"] == MODEL_REVISION
    assert metadata["block_index"] == block_index
    assert metadata["report_block"] == block_index + 1
    assert metadata["training_tokens"] == TRAINING_TOKENS
    assert cfg["architecture"] == "jumprelu"
    assert cfg["d_in"] == 1_920 and cfg["d_sae"] == D_SAE
    assert runner["model_name"] == MODEL_ID
    assert runner["model_from_pretrained_kwargs"]["revision"] == MODEL_REVISION
    assert runner["training_tokens"] == TRAINING_TOKENS
    assert runner["sae"]["d_sae"] == D_SAE
    return {
        "architecture": cfg["architecture"],
        "d_in": cfg["d_in"],
        "d_sae": cfg["d_sae"],
        "training_tokens": TRAINING_TOKENS,
        "metadata": metadata,
        "files": {
            name: {"bytes": item.stat().st_size, "sha256": sha256_file(item)}
            for name, item in paths.items()
        },
    }


def sparse_activation_table(features: np.ndarray, context_ids: np.ndarray) -> pa.Table:
    assert features.ndim == 2 and features.shape[1] == D_SAE
    assert context_ids.shape == (features.shape[0],)
    assert np.isfinite(features).all() and np.all(features >= 0)
    row_index, feature_id = np.nonzero(features)
    activations = features[row_index, feature_id].astype(np.float32, copy=False)
    return pa.Table.from_arrays(
        [
            pa.array(context_ids[row_index], type=pa.uint32()),
            pa.array(feature_id, type=pa.uint32()),
            pa.array(activations, type=pa.float32()),
        ],
        schema=SPARSE_SCHEMA,
    )
