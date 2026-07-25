"""Geometry tests for issue #402's paired lm-eval request adapter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from marin_dna.pipelines.rag_glm.lm_eval_adapter import (
    encode_rag_request,
    padded_rag_batches,
    rag_parity_diagnostic_records,
)
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


def test_padded_batches_preserve_order_and_report_real_rows() -> None:
    tokenizer = create_rag_char_tokenizer()
    row = encode_rag_request(tokenizer, _context(), "G" + "T" * 127, "A" + "T" * 127)
    rows = [row] * 17
    batches = padded_rag_batches(rows, batch_size=16)
    assert [n_real for _, n_real in batches] == [16, 1]
    assert [len(batch) for batch, _ in batches] == [16, 16]
    assert batches[1][0][0] == row
    assert batches[1][0][-1] == row


def test_parity_diagnostic_records_keep_only_metadata_and_raw_scores() -> None:
    requests = [
        SimpleNamespace(
            doc={
                "chrom": "chr1",
                "pos": 402,
                "ref": "A",
                "alt": "G",
                "strand": "+",
                "document_id": "variant-402:+",
                "context": "must-not-be-logged",
            }
        ),
        SimpleNamespace(
            doc={
                "chrom": "chr1",
                "pos": 402,
                "ref": "A",
                "alt": "G",
                "strand": "-",
                "document_id": "variant-402:-",
                "context": "must-not-be-logged",
            }
        ),
    ]
    records = rag_parity_diagnostic_records(
        requests,
        [(-3.0, -2.25, 0.75), (-4.0, -4.5, -0.5)],
        max_rows=1,
    )
    assert records == [
        {
            "chrom": "chr1",
            "pos": 402,
            "ref": "A",
            "alt": "G",
            "strand": "+",
            "document_id": "variant-402:+",
            "ref_loglikelihood": -3.0,
            "alt_loglikelihood": -2.25,
            "llr": 0.75,
        }
    ]
    assert "context" not in records[0]
