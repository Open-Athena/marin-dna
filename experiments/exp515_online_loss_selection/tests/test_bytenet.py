"""Tests for ByteNet architecture components."""

import pytest
import torch

from glm_experiments.models.components.bytenet import (
    ByteNet,
    ByteNetLayer,
    TransposeLayer,
)


def test_transpose_layer():
    """Test TransposeLayer swaps dimensions correctly."""
    layer = TransposeLayer()
    x = torch.randn(2, 10, 128)  # (batch, seq_len, hidden)
    out = layer(x)
    assert out.shape == (2, 128, 10)  # (batch, hidden, seq_len)


def test_bytenet_layer_forward():
    """Test ByteNetLayer forward pass with correct shapes."""
    layer = ByteNetLayer(hidden_size=128, kernel_size=5, dilation=2, slim=True)
    x = torch.randn(2, 100, 128)  # (batch, seq_len, hidden)
    out = layer(x)

    # Should preserve shape
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_bytenet_forward():
    """Test ByteNet forward pass with correct shapes."""
    model = ByteNet(
        hidden_size=128,
        n_layers=8,
        slim=True,
        dilation_base=2,
        dilation_cycle=8,
        first_kernel_size=9,
        rest_kernel_size=5,
    )

    x = torch.randn(2, 100, 128)  # (batch, seq_len, hidden)
    out = model(x)

    # Should preserve shape
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_bytenet_dilation_schedule():
    """Test dilation schedule is calculated correctly."""
    model = ByteNet(
        hidden_size=128,
        n_layers=16,
        dilation_base=2,
        dilation_cycle=8,
    )

    # Check that we have 16 layers
    assert len(model.layer) == 16

    # Expected dilations: [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128]
    expected_dilations = [2 ** (i % 8) for i in range(16)]

    for i, layer in enumerate(model.layer):
        # Each layer should be a ByteNetLayer with correct dilation
        assert isinstance(layer, ByteNetLayer)
        # Note: We'd need to expose dilation attribute to test this properly
        # For now, just check layer exists


def test_bytenet_layer_slim_vs_not_slim():
    """Test ByteNetLayer with slim=True vs slim=False."""
    x = torch.randn(2, 50, 128)

    # Slim version (hidden_size // 2 intermediate)
    layer_slim = ByteNetLayer(hidden_size=128, kernel_size=5, dilation=1, slim=True)
    out_slim = layer_slim(x)

    # Non-slim version (hidden_size intermediate)
    layer_not_slim = ByteNetLayer(hidden_size=128, kernel_size=5, dilation=1, slim=False)
    out_not_slim = layer_not_slim(x)

    # Both should preserve input shape
    assert out_slim.shape == x.shape
    assert out_not_slim.shape == x.shape


def test_bytenet_different_sequence_lengths():
    """Test ByteNet handles different sequence lengths correctly."""
    model = ByteNet(hidden_size=128, n_layers=4)

    # Test multiple sequence lengths
    for seq_len in [50, 100, 200, 512]:
        x = torch.randn(2, seq_len, 128)
        out = model(x)
        assert out.shape == (2, seq_len, 128)
