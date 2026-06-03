"""Tests for the causal-LM ChromBPNet embedding adapter (tiling + FWD/RC concat).

Uses a tiny randomly-initialised Llama as the gLM and our char tokenizer — we
test structural correctness (shapes, tiling coverage, RC handling, concat
layout, determinism), not learned content.
"""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from marin_dna.data.transforms import _get_nucleotide_token_ids
from marin_dna.pipelines.chrombpnet_eval.embedding import HFCausalChromBPNetEmbedder
from marin_dna.tokenizer.char import create_char_tokenizer

HIDDEN = 16
N_LAYERS = 4


@pytest.fixture(scope="module")
def tok():
    return create_char_tokenizer(bos=True, eos=True)


@pytest.fixture(scope="module")
def model(tok):
    cfg = LlamaConfig(
        vocab_size=len(tok),
        hidden_size=HIDDEN,
        intermediate_size=32,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=2,
        max_position_embeddings=4096,
    )
    return LlamaForCausalLM(cfg).eval()


def _seq_ids(tok, seq: str) -> torch.Tensor:
    nuc = _get_nucleotide_token_ids(tok)
    return torch.tensor([[nuc[c] for c in seq]], dtype=torch.long)


def _emb(model, tok, seq, **kw):
    return HFCausalChromBPNetEmbedder(
        model, tok, seq_input_size=len(seq), num_layers_avg=3, **kw
    )


def test_bidirectional_shape(model, tok):
    seq = "ACGTACGTACGTACGT"  # 16 bp
    emb = _emb(model, tok, seq, chunk_size=7, bidirectional=True)
    out = emb(_seq_ids(tok, seq))
    assert out.shape == (1, 16, 2 * HIDDEN)


def test_unidirectional_shape_and_equals_fwd_half(model, tok):
    seq = "ACGTACGTACGTACGT"
    ids = _seq_ids(tok, seq)
    bi = _emb(model, tok, seq, chunk_size=7, bidirectional=True)(ids)
    uni = _emb(model, tok, seq, chunk_size=7, bidirectional=False)(ids)
    assert uni.shape == (1, 16, HIDDEN)
    # the unidirectional output is exactly the forward half of the bidirectional one
    assert torch.allclose(uni, bi[..., :HIDDEN], atol=1e-6)


def test_tiling_covers_full_2114(model, tok):
    seq = "ACGT" * 529  # 2116 -> trim to 2114
    seq = seq[:2114]
    emb = HFCausalChromBPNetEmbedder(
        model, tok, seq_input_size=2114, chunk_size=255, num_layers_avg=3
    )
    out = emb(_seq_ids(tok, seq))
    assert out.shape == (1, 2114, 2 * HIDDEN)  # every position covered exactly once


def test_no_tiling_when_chunk_equals_seq(model, tok):
    seq = "ACGTACG"  # 7 bp
    emb = _emb(model, tok, seq, chunk_size=7, bidirectional=False)
    ids = _seq_ids(tok, seq)
    # _embed_strand with chunk==seq must equal a single _embed_chunk
    assert torch.allclose(emb._embed_strand(ids), emb._embed_chunk(ids), atol=1e-6)


def test_reverse_complement_ids(model, tok):
    emb = _emb(model, tok, "AACC", chunk_size=4)
    # RC("AACC") = complement("AACC")="TTGG" reversed = "GGTT"
    rc = emb.reverse_complement_ids(_seq_ids(tok, "AACC"))
    assert torch.equal(rc, _seq_ids(tok, "GGTT"))


def test_rc_half_is_flipped_rc_pass(model, tok):
    seq = "ACGTACGTACGTACGT"
    ids = _seq_ids(tok, seq)
    emb = _emb(model, tok, seq, chunk_size=7, bidirectional=True)
    out = emb(ids)
    # the rc half == flip(embed_strand(rc_ids)) along the position axis
    rc_stream = emb._embed_strand(emb.reverse_complement_ids(ids))
    assert torch.allclose(out[..., HIDDEN:], torch.flip(rc_stream, dims=[1]), atol=1e-6)


def test_deterministic(model, tok):
    seq = "ACGTACGTACGTACGT"
    ids = _seq_ids(tok, seq)
    emb = _emb(model, tok, seq, chunk_size=7, bidirectional=True)
    with torch.no_grad():
        a, b = emb(ids), emb(ids)
    assert torch.allclose(a, b, atol=0)


def test_out_dim_attribute(model, tok):
    seq = "ACGTACGT"
    assert _emb(model, tok, seq, chunk_size=4, bidirectional=True).out_dim == 2 * HIDDEN
    assert _emb(model, tok, seq, chunk_size=4, bidirectional=False).out_dim == HIDDEN


def test_gradients_flow_to_lm(model, tok):
    # No no_grad inside forward → LM stays in the graph when fine-tuning.
    seq = "ACGTACGTACGTACGT"
    emb = _emb(model, tok, seq, chunk_size=7, bidirectional=True)
    out = emb(_seq_ids(tok, seq))
    out.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "expected gradients to reach the LM parameters"
