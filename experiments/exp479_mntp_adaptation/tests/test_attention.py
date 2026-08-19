from __future__ import annotations

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from exp479_mntp.modeling import model_logits
from exp479_mntp.probes import context_dependence


def _model() -> Qwen3ForCausalLM:
    torch.manual_seed(7)
    config = Qwen3Config(
        vocab_size=8,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        max_position_embeddings=16,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config)
    model.eval()
    return model


def test_explicit_causal_mode_matches_default_logits() -> None:
    model = _model()
    input_ids = torch.tensor([[2, 3, 4, 7, 5, 6]])
    with torch.no_grad():
        default = model(input_ids=input_ids, use_cache=False).logits
        explicit = model_logits(model, input_ids=input_ids, attention_mode="causal")
    assert torch.equal(default, explicit)


def test_behavioral_attention_mode_and_right_flank_gradient() -> None:
    model = _model()
    left = torch.tensor([[2, 3, 4, 7, 5, 3]])
    right = left.clone()
    right[0, 5] = 6
    readout_position = 2

    with torch.no_grad():
        causal_left = model_logits(model, input_ids=left, attention_mode="causal")
        causal_right = model_logits(model, input_ids=right, attention_mode="causal")
        full_left = model_logits(model, input_ids=left, attention_mode="full")
        full_right = model_logits(model, input_ids=right, attention_mode="full")
    assert torch.equal(causal_left[:, readout_position], causal_right[:, readout_position])
    assert not torch.allclose(full_left[:, readout_position], full_right[:, readout_position])

    for mode, expect_nonzero in (("causal", False), ("full", True)):
        embeddings = model.get_input_embeddings()(left).detach().requires_grad_(True)
        logits = model_logits(model, inputs_embeds=embeddings, attention_mode=mode)
        logits[0, readout_position, 3].backward()
        right_gradient = embeddings.grad[0, 5]
        assert bool(torch.linalg.vector_norm(right_gradient) > 0) is expect_nonzero


def test_fixed_context_probe_separates_causal_and_full_right_context() -> None:
    model = _model()
    input_ids = torch.tensor([[2, 3, 4, 5, 7, 6, 3, 4, 5]])
    attention_mask = torch.ones_like(input_ids)
    causal = context_dependence(
        model,
        input_ids,
        attention_mask,
        target_input_position=4,
        mask_token_id=None,
        canonical_ids=(3, 4, 5, 6),
        attention_mode="causal",
        flank_offset=2,
    )
    full = context_dependence(
        model,
        input_ids,
        attention_mask,
        target_input_position=4,
        mask_token_id=7,
        canonical_ids=(3, 4, 5, 6),
        attention_mode="full",
        flank_offset=2,
    )
    assert causal["left_l1"] > 0
    assert causal["right_l1"] == 0
    assert full["left_l1"] > 0
    assert full["right_l1"] > 0
