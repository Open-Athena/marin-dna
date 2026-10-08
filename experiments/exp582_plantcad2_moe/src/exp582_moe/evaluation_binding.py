"""Checkpoint identity checks shared by the exp582 evaluation entry points."""

import hashlib
import json
from typing import Literal

from exp582_moe.config import (
    SOURCE_CHECKPOINT,
    SOURCE_METADATA_DIGEST,
    SOURCE_METADATA_RAW_SHA256,
)

CheckpointRole = Literal["dna-trained", "language-base"]


def validate_checkpoint_binding(
    *,
    checkpoint_role: CheckpointRole,
    condition: Literal["random", "pretrained"],
    checkpoint: str,
    checkpoint_metadata_digest: str,
    metadata_blob: bytes,
    tokenizer_sha256: str,
    allow_legacy_scratch: bool = False,
    legacy_scratch_checkpoint: str = "",
    legacy_scratch_digest: str = "",
) -> dict:
    """Validate a checkpoint and return the fields recorded in eval artifacts.

    The original language checkpoint predates tokenizer fingerprints. Its exception
    is deliberately pinned to the exact URI and both metadata digests. The
    character-bound language tokenizer remains an explicit evaluation input.
    """
    if checkpoint_role not in ("dna-trained", "language-base"):
        raise ValueError(f"Unknown checkpoint role: {checkpoint_role}")
    metadata = json.loads(metadata_blob)
    raw_digest = hashlib.sha256(metadata_blob).hexdigest()
    if checkpoint_role == "language-base":
        if (
            condition != "pretrained"
            or checkpoint != SOURCE_CHECKPOINT
            or checkpoint_metadata_digest != SOURCE_METADATA_DIGEST
            or raw_digest != SOURCE_METADATA_RAW_SHA256
            or metadata
            != {
                "step": 15128,
                "timestamp": "2026-08-19T11:45:26.977152",
                "is_temporary": False,
            }
        ):
            raise ValueError("Language-base binding differs from the pinned source")
        return {
            "checkpoint_role": checkpoint_role,
            "checkpoint_metadata_raw_sha256": raw_digest,
            "checkpoint_tokenizer_sha256": None,
            "evaluation_tokenizer_sha256": tokenizer_sha256,
        }
    if metadata.get("tokenizer_sha256") == tokenizer_sha256:
        return {
            "checkpoint_role": checkpoint_role,
            "checkpoint_metadata_raw_sha256": raw_digest,
            "checkpoint_tokenizer_sha256": tokenizer_sha256,
            "evaluation_tokenizer_sha256": tokenizer_sha256,
        }
    if (
        allow_legacy_scratch
        and condition == "random"
        and checkpoint == legacy_scratch_checkpoint
        and checkpoint_metadata_digest == legacy_scratch_digest
        and metadata.get("tokenizer_sha256") is None
    ):
        return {
            "checkpoint_role": checkpoint_role,
            "checkpoint_metadata_raw_sha256": raw_digest,
            "checkpoint_tokenizer_sha256": None,
            "evaluation_tokenizer_sha256": tokenizer_sha256,
            "legacy_scratch_binding": True,
        }
    raise ValueError("Checkpoint tokenizer binding differs")
