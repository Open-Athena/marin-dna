"""One-hot ChromBPNet construction (issue #241)."""

import torch

from marin_dna.pipelines.chrombpnet_eval.onehot import (
    build_onehot_chrombpnet,
    count_trainable_params,
)


def _tiny() -> torch.nn.Module:
    # Small tower so the test is fast; out_dim must stay 1000 (the central
    # window) for the conv crop math to line up with a 2114 bp input.
    return build_onehot_chrombpnet(bias_h5=None, n_filters=8, n_layers=2)


def test_forward_shapes():
    model = _tiny()
    x = torch.zeros(3, 4, 2114)
    x[:, torch.randint(0, 4, (2114,)), torch.arange(2114)] = 1.0
    profile, counts = model(x)
    assert profile.shape == (3, 1000), profile.shape
    assert counts.numel() == 3, counts.shape  # [B,1] log-count per region


def test_no_bias_keeps_fresh_trainable_bias():
    # Without a bias .h5 the fresh bias is left trainable (smoke-only path).
    model = _tiny()
    assert all(p.requires_grad for p in model.bias.parameters())


def test_count_trainable_drops_when_bias_frozen():
    model = _tiny()
    before = count_trainable_params(model)
    for p in model.bias.parameters():
        p.requires_grad = False
    after = count_trainable_params(model)
    assert 0 < after < before, (before, after)
    # Frozen-bias params are exactly the difference.
    n_bias = sum(p.numel() for p in model.bias.parameters())
    assert before - after == n_bias
