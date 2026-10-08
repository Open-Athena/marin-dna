import json

import pytest

from exp582_moe.config import SOURCE_CHECKPOINT, SOURCE_METADATA_DIGEST
from exp582_moe.evaluation_binding import validate_checkpoint_binding

SOURCE_METADATA = {
    "step": 15128,
    "timestamp": "2026-08-19T11:45:26.977152",
    "is_temporary": False,
}


def source_blob() -> bytes:
    return json.dumps(SOURCE_METADATA).encode()


def test_language_base_accepts_only_exact_pinned_source_and_records_eval_tokenizer():
    binding = validate_checkpoint_binding(
        checkpoint_role="language-base",
        condition="pretrained",
        checkpoint=SOURCE_CHECKPOINT,
        checkpoint_metadata_digest=SOURCE_METADATA_DIGEST,
        metadata_blob=source_blob(),
        tokenizer_sha256="a" * 64,
    )
    assert binding["checkpoint_role"] == "language-base"
    assert binding["checkpoint_tokenizer_sha256"] is None
    assert binding["evaluation_tokenizer_sha256"] == "a" * 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("condition", "random"),
        ("checkpoint", "s3://example/other"),
        ("checkpoint_metadata_digest", "b" * 64),
    ],
)
def test_language_base_rejects_other_source_bindings(field, value):
    args = {
        "checkpoint_role": "language-base",
        "condition": "pretrained",
        "checkpoint": SOURCE_CHECKPOINT,
        "checkpoint_metadata_digest": SOURCE_METADATA_DIGEST,
        "metadata_blob": source_blob(),
        "tokenizer_sha256": "a" * 64,
    }
    args[field] = value
    with pytest.raises(ValueError, match="pinned source"):
        validate_checkpoint_binding(**args)


def test_dna_trained_checkpoint_still_requires_tokenizer_fingerprint():
    metadata = json.dumps({"tokenizer_sha256": "a" * 64}).encode()
    validate_checkpoint_binding(
        checkpoint_role="dna-trained",
        condition="pretrained",
        checkpoint="s3://example/checkpoint",
        checkpoint_metadata_digest="b" * 64,
        metadata_blob=metadata,
        tokenizer_sha256="a" * 64,
    )
    with pytest.raises(ValueError, match="tokenizer"):
        validate_checkpoint_binding(
            checkpoint_role="dna-trained",
            condition="pretrained",
            checkpoint="s3://example/checkpoint",
            checkpoint_metadata_digest="b" * 64,
            metadata_blob=metadata,
            tokenizer_sha256="c" * 64,
        )
