"""Geometry tests for issue #402's paired lm-eval request adapter."""

from __future__ import annotations

import pytest

from marin_dna.pipelines.rag_glm.lm_eval_adapter import encode_rag_request
from marin_dna.pipelines.rag_glm.tokenizer import create_rag_char_tokenizer


def _context() -> str:
    return "[SEQ]".join(["A" * 255] * 7) + "[SEQ]" + "C" * 127


def test_encode_rag_request_has_exact_page_aligned_prefix() -> None:
    tokenizer = create_rag_char_tokenizer()
    ref = "G" + "T" * 127
    alt = "A" + "T" * 127
    encoded = encode_rag_request(tokenizer, _context(), ref, alt)
    assert len(encoded.prefix_ids) == 1_920
    assert len(encoded.ref_completion_ids) == 128
    assert len(encoded.alt_completion_ids) == 128
    assert encoded.prefix_ids[0] == tokenizer.bos_token_id
    assert encoded.ref_completion_ids[1:] == encoded.alt_completion_ids[1:]
    assert len(set(encoded.nucleotide_token_ids)) == 4


def test_encode_rag_request_rejects_nonshared_suffix() -> None:
    tokenizer = create_rag_char_tokenizer()
    with pytest.raises(AssertionError):
        encode_rag_request(
            tokenizer,
            _context(),
            "G" + "T" * 127,
            "A" + "C" + "T" * 126,
        )
