"""Tests for the issue #402 structured document tokenizer."""

import tempfile

from transformers import AutoTokenizer

from marin_dna.pipelines.rag_glm.dataset import (
    DOCUMENT_TOKENS,
    HUMAN_SEGMENT_START,
    HUMAN_VARIANT_TOKEN_INDEX,
    MISSING_SEQUENCE,
    assemble_document,
)
from marin_dna.pipelines.rag_glm.tokenizer import create_rag_char_tokenizer


def _document() -> str:
    return assemble_document(
        (
            "A" * 255,
            "C" * 255,
            "G" * 255,
            "T" * 255,
            MISSING_SEQUENCE,
            "A" * 255,
            "C" * 255,
            "G" * 255,
        )
    )


def test_rag_tokenizer_exact_ids_and_positions() -> None:
    tokenizer = create_rag_char_tokenizer()
    ids = tokenizer.encode(_document())
    boundary_id = tokenizer.convert_tokens_to_ids("[SEQ]")

    assert len(ids) == DOCUMENT_TOKENS
    assert tokenizer.cls_token_id == tokenizer.bos_token_id == ids[0]
    assert tokenizer.eos_token_id is None
    assert [index for index, token_id in enumerate(ids) if token_id == boundary_id] == [
        256,
        512,
        768,
        1024,
        1280,
        1536,
        1792,
    ]
    assert ids[HUMAN_SEGMENT_START] == tokenizer.convert_tokens_to_ids("g")
    assert ids[HUMAN_VARIANT_TOKEN_INDEX] == tokenizer.convert_tokens_to_ids("g")


def test_rag_tokenizer_maps_n_to_unknown() -> None:
    tokenizer = create_rag_char_tokenizer()
    ids = tokenizer.encode(_document())
    missing_start = 1 + 4 * 256
    assert ids[missing_start : missing_start + 255] == [tokenizer.unk_token_id] * 255


def test_rag_tokenizer_save_load_preserves_special_ids() -> None:
    tokenizer = create_rag_char_tokenizer()
    with tempfile.TemporaryDirectory() as tmpdir:
        tokenizer.save_pretrained(tmpdir)
        loaded = AutoTokenizer.from_pretrained(tmpdir)
        assert loaded.encode(_document()) == tokenizer.encode(_document())
        assert loaded.cls_token_id == loaded.bos_token_id
        assert loaded.convert_tokens_to_ids("[SEQ]") == tokenizer.convert_tokens_to_ids(
            "[SEQ]"
        )
        assert loaded.vocab_size == tokenizer.vocab_size
