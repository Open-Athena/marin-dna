"""Tests for EnhancerClassifier (random encoder, no pretrained weights)."""

import tempfile
from pathlib import Path

import torch

from marin_dna.pipelines.enhancer_classification.model import (
    ENCODER_OUTPUT_DIM,
    EnhancerClassifier,
)

SEQ_LEN = 255
BATCH = 2


def _random_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Create a random one-hot batch and binary labels."""
    indices = torch.randint(0, 4, (BATCH, SEQ_LEN))
    x = torch.nn.functional.one_hot(indices, num_classes=4).float()
    y = torch.tensor([0.0, 1.0])
    return x, y


def test_forward_output_shape():
    model = EnhancerClassifier(weights_path=None, freeze_backbone=False)
    model.eval()
    x, _ = _random_batch()
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (BATCH,)


def test_freeze_backbone():
    model = EnhancerClassifier(weights_path=None, freeze_backbone=True)
    for param in model.encoder.parameters():
        assert not param.requires_grad
    for param in model.head.parameters():
        assert param.requires_grad


def test_unfreeze_backbone():
    model = EnhancerClassifier(weights_path=None, freeze_backbone=False)
    encoder_params = list(model.encoder.parameters())
    assert len(encoder_params) > 0
    assert all(p.requires_grad for p in encoder_params)


def test_loss_computation():
    model = EnhancerClassifier(weights_path=None, freeze_backbone=False)
    x, y = _random_batch()
    logits = model(x)
    loss = model.loss_fn(logits, y)
    assert loss.shape == ()
    assert loss.item() > 0


def test_checkpoint_roundtrip():
    """Save and reload a checkpoint, verify forward pass produces the same output."""
    model = EnhancerClassifier(weights_path=None, freeze_backbone=False)
    model.eval()
    x, _ = _random_batch()

    with torch.no_grad():
        out_before = model(x)

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = Path(tmp) / "test.ckpt"
        torch.save(model.state_dict(), ckpt_path)

        restored = EnhancerClassifier(weights_path=None, freeze_backbone=False)
        restored.load_state_dict(torch.load(ckpt_path, weights_only=True))
        restored.eval()

        with torch.no_grad():
            out_after = restored(x)

    assert torch.allclose(out_before, out_after)


def test_head_input_dim_matches_encoder():
    """Verify the linear head expects ENCODER_OUTPUT_DIM features."""
    model = EnhancerClassifier(weights_path=None, freeze_backbone=False)
    linear = model.head[-1]
    assert linear.in_features == ENCODER_OUTPUT_DIM
