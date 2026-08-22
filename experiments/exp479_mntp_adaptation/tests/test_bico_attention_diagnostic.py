from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from exp479_mntp.bico_attention_diagnostic import (
    bico_attention_forward,
    excluded_selected_key_mask,
    reflected_future_rope,
)


class TinyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head_dim = 4
        self.num_key_value_groups = 1
        self.scaling = 0.5
        self.attention_dropout = 0.0
        self.q_proj = nn.Identity()
        self.k_proj = nn.Identity()
        self.v_proj = nn.Identity()
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.o_proj = nn.Identity()


def _position_embeddings(length: int) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(length, dtype=torch.float32)
    frequencies = torch.outer(positions, torch.tensor([1.0, 0.1]))
    angles = torch.cat((frequencies, frequencies), dim=-1)
    return angles.cos()[None, :, :], angles.sin()[None, :, :]


def test_excluded_selected_key_mask_excludes_shifted_target_for_every_query() -> None:
    token_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    mask = excluded_selected_key_mask(
        token_mask,
        torch.tensor([1, 0]),
        dtype=torch.float32,
    )
    assert mask.shape == (2, 1, 4, 4)
    assert torch.all(mask[0, 0, :, 2] < -1e30)
    assert torch.all(mask[1, 0, :, 1] < -1e30)
    assert torch.all(mask[1, 0, :, 3] < -1e30)
    assert torch.all(mask[0, 0, :, [0, 1, 3]] == 0)


def test_excluded_selected_key_mask_rejects_shifted_position_past_sequence() -> None:
    with pytest.raises(ValueError, match="outside"):
        excluded_selected_key_mask(
            torch.ones((1, 4), dtype=torch.long),
            torch.tensor([3]),
            dtype=torch.float32,
        )


def test_bico_attention_uses_reflected_rope_only_for_future_keys() -> None:
    module = TinyAttention().eval()
    hidden = torch.tensor([[[0.2, -0.4, 0.6, 0.8], [0.1, 0.7, -0.3, 0.5], [-0.6, 0.4, 0.2, 0.9]]])
    cos, sin = _position_embeddings(hidden.shape[1])
    output, weights = bico_attention_forward(module, hidden, (cos, sin), None)

    query = hidden[:, None, :, :]
    key = hidden[:, None, :, :]
    standard_query, standard_key = apply_rotary_pos_emb(query, key, cos, sin)
    reflected_query, reflected_key = apply_rotary_pos_emb(query, key, cos, -sin)
    standard = torch.matmul(standard_query, standard_key.transpose(2, 3))
    reflected = torch.matmul(reflected_query, reflected_key.transpose(2, 3))
    positions = torch.arange(hidden.shape[1])
    future = positions[None, :] > positions[:, None]
    expected_logits = torch.where(future[None, None], reflected, standard) * module.scaling
    expected_weights = expected_logits.softmax(dim=-1)
    expected_output = torch.matmul(expected_weights, hidden[:, None]).transpose(1, 2)
    expected_output = expected_output.reshape_as(hidden)
    torch.testing.assert_close(weights, expected_weights)
    torch.testing.assert_close(output, expected_output)


def test_bico_attention_matches_standard_rope_under_causal_mask() -> None:
    module = TinyAttention().eval()
    generator = torch.Generator().manual_seed(4)
    hidden = torch.randn((2, 4, 4), generator=generator)
    cos, sin = _position_embeddings(hidden.shape[1])
    allowed = torch.ones((4, 4), dtype=torch.bool).tril()
    mask = torch.zeros((2, 1, 4, 4)).masked_fill(~allowed[None, None], -torch.inf)
    output, weights = bico_attention_forward(module, hidden, (cos, sin), mask)

    query = hidden[:, None, :, :]
    key = hidden[:, None, :, :]
    query, key = apply_rotary_pos_emb(query, key, cos, sin)
    expected_weights = torch.matmul(query, key.transpose(2, 3)) * module.scaling + mask
    expected_weights = expected_weights.softmax(dim=-1)
    expected_output = torch.matmul(expected_weights, hidden[:, None]).transpose(1, 2)
    expected_output = expected_output.reshape_as(hidden)
    torch.testing.assert_close(weights, expected_weights)
    torch.testing.assert_close(output, expected_output)


def test_reflected_future_rope_restores_original_layer_forward() -> None:
    attention = TinyAttention()
    original = attention.forward
    model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)]))
    with reflected_future_rope(model):  # type: ignore[arg-type]
        assert attention.forward.__func__ is bico_attention_forward
    assert attention.forward == original


def test_reflected_future_rope_traverses_peft_wrapper_shape() -> None:
    attention = TinyAttention()
    original = attention.forward
    model = SimpleNamespace(
        base_model=SimpleNamespace(
            model=SimpleNamespace(
                model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)])
            )
        )
    )
    with reflected_future_rope(model):  # type: ignore[arg-type]
        assert attention.forward.__func__ is bico_attention_forward
    assert attention.forward == original
