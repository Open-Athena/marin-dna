"""Tests for GLMChromBPNet (gLM embeddings wired into the vendored ChromBPNet).

Tiny randomly-initialised Llama + our char tokenizer; a small BPNet head
(n_filters=8, n_layers=2) for speed. Structural correctness only.
"""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from marin_dna.data.transforms import (
    _get_nucleotide_token_ids,
    _get_special_token_counts,
)
from marin_dna.pipelines.chrombpnet_eval._vendor.chrombpnet.model_config import (
    ArsenalChromBPNetConfig,
)
from marin_dna.pipelines.chrombpnet_eval.model import GLMChromBPNet
from marin_dna.tokenizer.char import create_char_tokenizer

HIDDEN = 16


@pytest.fixture(scope="module")
def tok():
    return create_char_tokenizer(bos=True, eos=True)


def _fresh_hf(tok):
    cfg = LlamaConfig(
        vocab_size=len(tok),
        hidden_size=HIDDEN,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        max_position_embeddings=4096,
    )
    return LlamaForCausalLM(cfg).eval()


def _model(hf, tok, **kw):
    cfg = ArsenalChromBPNetConfig(input_len=2114, n_filters=8, n_layers=2)
    return GLMChromBPNet(
        hf, tok, input_len=2114, chunk_size=255, num_layers_avg=2, config=cfg, **kw
    )


def test_forward_shapes(tok):
    m = _model(_fresh_hf(tok), tok, bidirectional=True).eval()
    x = torch.zeros(2, 4, 2114)
    x[:, 0, :] = 1  # all-A one-hot
    with torch.no_grad():
        profile, counts = m(x)
    assert profile.shape == (2, 1000)
    assert counts.shape == (2, 1)


def test_iconv_sized_to_embedding_dim(tok):
    assert (
        _model(_fresh_hf(tok), tok, bidirectional=True).model.iconv.in_channels
        == 2 * HIDDEN
    )
    assert (
        _model(_fresh_hf(tok), tok, bidirectional=False).model.iconv.in_channels
        == HIDDEN
    )


def test_freeze_vs_finetune(tok):
    frozen = _model(_fresh_hf(tok), tok, finetune=False)
    assert all(not p.requires_grad for p in frozen.embedder.model.parameters())
    ft = _model(_fresh_hf(tok), tok, finetune=True)
    assert any(p.requires_grad for p in ft.embedder.model.parameters())


def test_one_hot_to_tokens(tok):
    m = _model(_fresh_hf(tok), tok)
    nuc = _get_nucleotide_token_ids(tok)
    n_prefix, _ = _get_special_token_counts(tok)
    # identity rows = one-hot for A,C,G,T in alphabetical channel order
    out = m.one_hot_to_tokens(torch.eye(4)[None])  # (1, 4, 4)
    assert out.tolist() == [[nuc["A"], nuc["C"], nuc["G"], nuc["T"]]]
    # all-zero channel -> N
    assert m.one_hot_to_tokens(torch.zeros(1, 1, 4)).item() == tok.encode("N")[n_prefix]


def test_head_params_trainable_when_lm_frozen(tok):
    # The ChromBPNet accessibility tower must train even with the LM frozen.
    m = _model(_fresh_hf(tok), tok, finetune=False)
    assert any(p.requires_grad for p in m.model.parameters())
