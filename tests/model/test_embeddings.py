"""Tests for ``marin_dna.model.embeddings`` (window embeddings, #246).

Two model doubles, mirroring ``test_interpretation.py``:

- ``_PerPositionEmbedModel`` — hidden state at ``(layer L, position t)`` is
  ``[token_id_at_t, L]``, so center mean-pooling (channel 0) and layer
  selection (channel 1) are hand-checkable.
- The real ``hf-internal-testing/tiny-random-GPTNeoXForCausalLM`` — validates
  the actual HF ``output_hidden_states`` / ``.hidden_states`` interface.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from marin_dna.data.dna import reverse_complement
from marin_dna.model.embeddings import window_embeddings
from marin_dna.tokenizer.char import create_char_tokenizer

TINY_CLM = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"


class _PerPositionEmbedModel(nn.Module):
    """Returns ``hidden_states`` whose vector at ``(layer L, position t)`` is
    ``[token_id_at_t, L]``: channel 0 carries the token id (so center-pooling is
    the mean of the central token ids), channel 1 the layer index (so
    ``layer_index`` selection is checkable). One parameter so device detection
    works."""

    def __init__(self, n_layers: int = 3):
        super().__init__()
        self.n_layers = n_layers
        self._p = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, output_hidden_states: bool = False, **kwargs):
        assert output_hidden_states, "kernel must request output_hidden_states"
        tok = input_ids.float()  # [B, L]
        hs = tuple(
            torch.stack([tok, torch.full_like(tok, float(layer))], dim=-1)
            for layer in range(self.n_layers + 1)  # embeddings + N layers
        )  # each [B, L, 2]
        return SimpleNamespace(hidden_states=hs)


def _expected_center_mean(tok, seq: str, n_center_bp: int, n_prefix: int = 1) -> float:
    ids = tok.encode(seq)
    c0 = (len(seq) - n_center_bp) // 2
    lo = n_prefix + c0
    return float(np.mean(ids[lo : lo + n_center_bp]))


def test_window_embeddings_shape_and_center_pool():
    tok = create_char_tokenizer(bos=True, eos=True)
    model = _PerPositionEmbedModel(n_layers=3).eval()
    seqs = ["ACGTACGTAC", "TTGGCCAATT"]  # W = 10
    emb = window_embeddings(model, tok, seqs, layer_index=-1, n_center_bp=4, rc=False)
    assert emb.shape == (2, 2)
    # channel 1 = last layer index (n_layers=3 → hidden_states[-1] is index 3)
    assert np.allclose(emb[:, 1], 3.0)
    # channel 0 = mean token id over the 4 center positions
    for k, s in enumerate(seqs):
        assert np.isclose(emb[k, 0], _expected_center_mean(tok, s, 4))


def test_window_embeddings_layer_index_selects_layer():
    tok = create_char_tokenizer(bos=True, eos=True)
    model = _PerPositionEmbedModel(n_layers=3).eval()
    seqs = ["ACGTACGTAC"]
    for layer_index, expected in [(0, 0.0), (2, 2.0), (-1, 3.0)]:
        emb = window_embeddings(
            model, tok, seqs, layer_index=layer_index, n_center_bp=4, rc=False
        )
        assert np.allclose(emb[:, 1], expected)


def test_window_embeddings_fwd_rc_average():
    tok = create_char_tokenizer(bos=True, eos=True)
    model = _PerPositionEmbedModel(n_layers=2).eval()
    seq = "ACGTACGTAC"
    emb = window_embeddings(model, tok, [seq], layer_index=-1, n_center_bp=4, rc=True)
    fwd = _expected_center_mean(tok, seq, 4)
    rc = _expected_center_mean(tok, reverse_complement(seq), 4)
    assert np.isclose(emb[0, 0], (fwd + rc) / 2)


def test_window_embeddings_unequal_length_rejected():
    tok = create_char_tokenizer(bos=True, eos=True)
    model = _PerPositionEmbedModel().eval()
    with pytest.raises(AssertionError, match="equal length"):
        window_embeddings(model, tok, ["ACGT", "ACGTA"], n_center_bp=2)


def test_window_embeddings_batch_size_invariant():
    tok = create_char_tokenizer(bos=True, eos=True)
    model = _PerPositionEmbedModel().eval()
    seqs = ["ACGTACGTAC", "TTGGCCAATT", "GGGGCCCCAA", "ATATATATAT"]
    a = window_embeddings(model, tok, seqs, n_center_bp=4, batch_size=1, rc=True)
    b = window_embeddings(model, tok, seqs, n_center_bp=4, batch_size=4, rc=True)
    np.testing.assert_allclose(a, b)


def test_window_embeddings_real_tiny_clm():
    """Real HF causal LM: the ``.hidden_states`` interface + determinism."""
    tok = create_char_tokenizer(bos=True, eos=True)
    model = AutoModelForCausalLM.from_pretrained(TINY_CLM).eval()
    seqs = ["ACGTACGTACGT", "TTTTGGGGCCCC"]  # W = 12
    emb = window_embeddings(model, tok, seqs, layer_index=-1, n_center_bp=6, rc=True)
    assert emb.ndim == 2 and emb.shape[0] == 2
    assert np.isfinite(emb).all()
    emb2 = window_embeddings(model, tok, seqs, layer_index=-1, n_center_bp=6, rc=True)
    np.testing.assert_allclose(emb, emb2)
