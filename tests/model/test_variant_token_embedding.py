"""Tests for the per-token variant-embedding kernel (issue #314).

``compute_variant_token_embeddings`` (model/scoring.py): full-window REF/ALT
hidden states with the single centered token swapped to the alt allele, BOS
prefix dropped. The full ``run_variant_embeddings`` HF-Trainer path is exercised
by the GPU extraction; this is the fast CPU unit test of the kernel.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn

from marin_dna.model.scoring import compute_variant_token_embeddings


class _HiddenStateModel(nn.Module):
    """Hidden state at ``(layer L, position t)`` = ``[token_id_at_t, L]`` — lets us
    hand-check which token sits at each position (channel 0) and which layer was
    read (channel 1). Mirrors the fake in ``tests/model/test_window_embedding.py``."""

    def __init__(self, n_layers: int = 3):
        super().__init__()
        self.n_layers = n_layers

    def _layer(self, tok: torch.Tensor, idx: int) -> torch.Tensor:
        return torch.stack([tok, torch.full_like(tok, float(idx))], dim=-1)

    def forward(self, input_ids, output_hidden_states: bool = False):
        tok = input_ids.float()
        last = self._layer(tok, self.n_layers)
        if output_hidden_states:
            hs = tuple(self._layer(tok, i) for i in range(self.n_layers + 1))
            return SimpleNamespace(last_hidden_state=last, hidden_states=hs)
        return SimpleNamespace(last_hidden_state=last)


def test_ref_alt_swap_and_prefix_drop():
    model = _HiddenStateModel(n_layers=3)
    n_prefix, var_pos = 1, 3  # BOS at 0; variant at token 3 → DNA index 2
    input_ids = torch.tensor([[99, 10, 11, 12, 13, 14], [99, 20, 21, 22, 23, 24]])
    alt_token_id = torch.tensor([7, 8])
    out = compute_variant_token_embeddings(
        model, input_ids, alt_token_id, var_pos=var_pos, n_prefix=n_prefix
    )
    assert out.shape == (2, 2, 5, 2)  # [B, 2(ref/alt), L_dna=5, D=2]
    assert out.dtype == torch.float16
    out = out.float()
    ref, alt = out[:, 0], out[:, 1]  # [B, L_dna, D]

    # channel 0 = token id at each DNA position (BOS dropped); ref = input DNA tokens
    torch.testing.assert_close(ref[:, :, 0], input_ids[:, n_prefix:].float())
    # alt = ref except the variant DNA position holds the alt token
    expected_alt = input_ids[:, n_prefix:].clone().float()
    expected_alt[:, var_pos - n_prefix] = alt_token_id.float()
    torch.testing.assert_close(alt[:, :, 0], expected_alt)
    # channel 1 = last layer index everywhere
    torch.testing.assert_close(ref[:, :, 1], torch.full((2, 5), 3.0))


def test_does_not_mutate_input_ids():
    model = _HiddenStateModel()
    input_ids = torch.tensor([[99, 10, 11, 12, 13, 14]])
    before = input_ids.clone()
    compute_variant_token_embeddings(
        model, input_ids, torch.tensor([7]), var_pos=3, n_prefix=1
    )
    torch.testing.assert_close(input_ids, before)  # alt path clones, no in-place edit


def test_layer_index_selects_layer():
    model = _HiddenStateModel(n_layers=3)
    input_ids = torch.tensor([[99, 10, 11, 12, 13, 14]])
    out = compute_variant_token_embeddings(
        model, input_ids, torch.tensor([7]), var_pos=3, n_prefix=1, layer_index=1
    )
    # channel 1 = the selected intermediate layer index
    torch.testing.assert_close(out.float()[:, 0, :, 1], torch.full((1, 5), 1.0))
