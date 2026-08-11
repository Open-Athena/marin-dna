"""Tests for the issue #402 Levanter batch tokenizer."""

import tempfile

import numpy as np
import pytest

pytest.importorskip("levanter", reason="install with the marin group to run")

from levanter.tokenizers import load_tokenizer
from marin_dna_evals.levanter.rag_batch_tokenizer import (
    RAGDNABatchTokenizer,
)
from marin_dna_rag_glm.dataset import (
    DOCUMENT_TOKENS,
    HUMAN_SEGMENT_START,
    MISSING_SEQUENCE,
    assemble_document,
)
from marin_dna_rag_glm.tokenizer import (
    create_rag_char_tokenizer,
)


def test_rag_batch_tokenizer_shape_boundaries_and_uniform_loss() -> None:
    hf_tokenizer = create_rag_char_tokenizer()
    document = assemble_document(
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
    with tempfile.TemporaryDirectory() as tmpdir:
        hf_tokenizer.save_pretrained(tmpdir)
        tokenizer = load_tokenizer(tmpdir)
        batch_tokenizer = RAGDNABatchTokenizer(tokenizer)
        result = batch_tokenizer([{"seq": document}])[0]

    assert result["input_ids"].shape == (DOCUMENT_TOKENS,)
    assert result["loss_weight"].shape == (DOCUMENT_TOKENS,)
    assert result["input_ids"][0] == tokenizer.bos_token_id
    assert result["input_ids"][
        HUMAN_SEGMENT_START
    ] == hf_tokenizer.convert_tokens_to_ids("g")
    np.testing.assert_array_equal(
        result["loss_weight"],
        np.array([1.0] * (DOCUMENT_TOKENS - 1) + [0.0], dtype=np.float32),
    )
