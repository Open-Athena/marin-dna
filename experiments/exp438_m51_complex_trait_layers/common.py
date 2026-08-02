"""Pinned constants and small helpers for experiment 438."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ISSUE = 438
SEED = 288
MODEL_ID = "marin-dna/marin-dna-exp135-m5.1"
MODEL_REVISION = "c0676b2012b8b9c526deb26ff517f6b92b6d375d"
SAELENS_REVISION = "8be14080485952f729ed58d674bcddf9778e0aa4"
MARIN_DNA_REVISION = "c4c1c86bbfc0bd58ff76dda3bac1d2acea856a33"
D_SAE = 15_360
TRAINING_TOKENS = 25_000_200

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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def assert_commit(value: str) -> None:
    assert len(value) == 40
    assert all(character in "0123456789abcdef" for character in value)
