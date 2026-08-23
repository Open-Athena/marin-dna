"""Tests for attention pattern utility functions."""

import pytest

from glm_experiments.models.utils.attention_patterns import (
    all_global,
    all_local,
    alternating_global_local,
    custom_pattern,
    first_k_global_rest_local,
    longformer_style,
    sparse_transformer,
)

# =============================================================================
# Test alternating_global_local
# =============================================================================


def test_alternating_global_local_start_with_global():
    """Test alternating pattern starting with global."""
    pattern = alternating_global_local(n_layers=4, window_size=256, start_with_global=True)
    assert pattern == [None, 256, None, 256]


def test_alternating_global_local_start_with_local():
    """Test alternating pattern starting with local."""
    pattern = alternating_global_local(n_layers=4, window_size=256, start_with_global=False)
    assert pattern == [256, None, 256, None]


def test_alternating_global_local_odd_layers():
    """Test alternating pattern with odd number of layers."""
    pattern = alternating_global_local(n_layers=5, window_size=128, start_with_global=True)
    assert pattern == [None, 128, None, 128, None]


def test_alternating_global_local_single_layer():
    """Test alternating pattern with single layer."""
    pattern = alternating_global_local(n_layers=1, window_size=256, start_with_global=True)
    assert pattern == [None]

    pattern = alternating_global_local(n_layers=1, window_size=256, start_with_global=False)
    assert pattern == [256]


# =============================================================================
# Test all_local
# =============================================================================


def test_all_local():
    """Test all local attention pattern."""
    pattern = all_local(n_layers=4, window_size=256)
    assert pattern == [256, 256, 256, 256]


def test_all_local_different_sizes():
    """Test all local with different window sizes."""
    pattern = all_local(n_layers=3, window_size=128)
    assert pattern == [128, 128, 128]


# =============================================================================
# Test all_global
# =============================================================================


def test_all_global():
    """Test all global attention pattern."""
    pattern = all_global(n_layers=4)
    assert pattern == [None, None, None, None]


def test_all_global_single_layer():
    """Test all global with single layer."""
    pattern = all_global(n_layers=1)
    assert pattern == [None]


# =============================================================================
# Test sparse_transformer
# =============================================================================


def test_sparse_transformer_default():
    """Test sparse transformer with default global_every=3."""
    pattern = sparse_transformer(n_layers=6, window_size=256)
    assert pattern == [None, 256, 256, None, 256, 256]


def test_sparse_transformer_custom_interval():
    """Test sparse transformer with custom global interval."""
    pattern = sparse_transformer(n_layers=8, window_size=128, global_every=2)
    assert pattern == [None, 128, None, 128, None, 128, None, 128]


def test_sparse_transformer_global_every_4():
    """Test sparse transformer with global every 4 layers."""
    pattern = sparse_transformer(n_layers=8, window_size=256, global_every=4)
    assert pattern == [None, 256, 256, 256, None, 256, 256, 256]


# =============================================================================
# Test longformer_style
# =============================================================================


def test_longformer_style_default():
    """Test Longformer-style decreasing windows."""
    pattern = longformer_style(n_layers=12, base_window=512)
    assert pattern == [512, 512, 512, 512, 256, 256, 256, 256, 128, 128, 128, 128]


def test_longformer_style_smaller_base():
    """Test Longformer-style with smaller base window."""
    pattern = longformer_style(n_layers=8, base_window=256)
    assert pattern == [256, 256, 256, 256, 128, 128, 128, 128]


def test_longformer_style_min_window():
    """Test Longformer-style respects minimum window of 64."""
    pattern = longformer_style(n_layers=16, base_window=128)
    # First 4: 128, next 4: 64, rest: 64 (minimum)
    assert pattern == [128, 128, 128, 128, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64]


# =============================================================================
# Test first_k_global_rest_local
# =============================================================================


def test_first_k_global_rest_local():
    """Test first k layers global, rest local."""
    pattern = first_k_global_rest_local(n_layers=6, k=2, window_size=256)
    assert pattern == [None, None, 256, 256, 256, 256]


def test_first_k_global_rest_local_all_global():
    """Test when k equals n_layers (all global)."""
    pattern = first_k_global_rest_local(n_layers=4, k=4, window_size=256)
    assert pattern == [None, None, None, None]


def test_first_k_global_rest_local_all_local():
    """Test when k=0 (all local)."""
    pattern = first_k_global_rest_local(n_layers=4, k=0, window_size=256)
    assert pattern == [256, 256, 256, 256]


def test_first_k_global_rest_local_invalid_k():
    """Test that k > n_layers raises ValueError."""
    with pytest.raises(ValueError, match="k=5 cannot be greater than n_layers=4"):
        first_k_global_rest_local(n_layers=4, k=5, window_size=256)


# =============================================================================
# Test custom_pattern
# =============================================================================


def test_custom_pattern_repeat():
    """Test custom pattern with repeating."""
    pattern = custom_pattern(n_layers=6, window_sizes=[None, 256])
    assert pattern == [None, 256, None, 256, None, 256]


def test_custom_pattern_complex_repeat():
    """Test custom pattern with complex repeating."""
    pattern = custom_pattern(n_layers=9, window_sizes=[None, 128, 128])
    assert pattern == [None, 128, 128, None, 128, 128, None, 128, 128]


def test_custom_pattern_truncate():
    """Test custom pattern truncates if list is longer."""
    pattern = custom_pattern(n_layers=3, window_sizes=[512, 256, 128, 64])
    assert pattern == [512, 256, 128]


def test_custom_pattern_exact_match():
    """Test custom pattern with exact length match."""
    pattern = custom_pattern(n_layers=4, window_sizes=[None, 256, 128, 64])
    assert pattern == [None, 256, 128, 64]


def test_custom_pattern_empty_list():
    """Test custom pattern raises error on empty list."""
    with pytest.raises(ValueError, match="window_sizes cannot be empty"):
        custom_pattern(n_layers=4, window_sizes=[])


# =============================================================================
# Integration tests with Transformer
# =============================================================================


def test_pattern_integration_with_transformer():
    """Test that generated patterns work with Transformer class."""
    import torch

    from glm_experiments.models.components.transformer import Transformer

    n_layers = 4
    pattern = alternating_global_local(n_layers=n_layers, window_size=32)

    transformer = Transformer(
        hidden_size=128,
        n_layers=n_layers,
        num_heads=4,
        sliding_window=pattern,
        context_length=64,
    )

    # Forward pass
    x = torch.randn(2, 16, 128)
    output = transformer(x)

    assert output.shape == (2, 16, 128)
    assert not torch.isnan(output).any()


def test_all_patterns_valid_length():
    """Test that all pattern functions return correct length."""
    n_layers = 8

    patterns = [
        alternating_global_local(n_layers, 256),
        all_local(n_layers, 256),
        all_global(n_layers),
        sparse_transformer(n_layers, 256, global_every=3),
        longformer_style(n_layers, 512),
        first_k_global_rest_local(n_layers, 2, 256),
        custom_pattern(n_layers, [None, 128]),
    ]

    for pattern in patterns:
        assert len(pattern) == n_layers
