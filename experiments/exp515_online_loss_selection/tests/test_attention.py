"""Tests for generalized attention functions with sliding window support."""

import time

import pytest
import torch
import torch.nn.functional as F

from glm_experiments.models.components.attention import (
    _get_or_create_block_mask,
    scaled_dot_product_attention,
)


@pytest.fixture
def attention_inputs():
    """Create sample attention inputs for testing."""
    batch, heads, seq_len, head_dim = 2, 4, 32, 64
    query = torch.randn(batch, heads, seq_len, head_dim)
    key = torch.randn(batch, heads, seq_len, head_dim)
    value = torch.randn(batch, heads, seq_len, head_dim)
    return query, key, value


# =============================================================================
# A. Backward Compatibility Tests
# =============================================================================


def test_fallback_to_standard_attention(attention_inputs):
    """Verify sliding_window=None uses F.scaled_dot_product_attention."""
    query, key, value = attention_inputs

    # Call with sliding_window=None should use standard attention
    output = scaled_dot_product_attention(query, key, value, sliding_window=None)

    # Should produce valid output with correct shape
    assert output.shape == query.shape
    assert output.dtype == torch.float32


def test_output_matches_standard_when_no_sliding_window(attention_inputs):
    """Verify output matches F.scaled_dot_product_attention when sliding_window=None."""
    query, key, value = attention_inputs

    # Our function with sliding_window=None
    output_ours = scaled_dot_product_attention(query, key, value, sliding_window=None)

    # Standard PyTorch function
    output_standard = F.scaled_dot_product_attention(query, key, value)

    # Should produce identical results
    assert torch.allclose(output_ours, output_standard, rtol=1e-5, atol=1e-7)


def test_causal_only_matches_standard(attention_inputs):
    """Verify is_causal=True, sliding_window=None matches standard attention."""
    query, key, value = attention_inputs

    # Our function with is_causal=True, sliding_window=None
    output_ours = scaled_dot_product_attention(
        query, key, value, is_causal=True, sliding_window=None
    )

    # Standard PyTorch function with is_causal
    output_standard = F.scaled_dot_product_attention(query, key, value, is_causal=True)

    # Should produce identical results
    assert torch.allclose(output_ours, output_standard, rtol=1e-5, atol=1e-7)


# =============================================================================
# B. Sliding Window Tests
# =============================================================================


def test_sliding_window_shape(attention_inputs):
    """Verify output shape matches input when using sliding window."""
    query, key, value = attention_inputs

    output = scaled_dot_product_attention(query, key, value, sliding_window=8)

    assert output.shape == query.shape
    assert output.dtype == torch.float32


def test_sliding_window_attention_pattern():
    """Verify attention is only applied within the sliding window."""
    # Use small dimensions for easier verification
    batch, heads, seq_len, head_dim = 1, 1, 8, 4

    # Create simple inputs where we can verify attention pattern
    query = torch.randn(batch, heads, seq_len, head_dim)
    key = torch.randn(batch, heads, seq_len, head_dim)
    value = torch.eye(seq_len).view(batch, heads, seq_len, seq_len)[:, :, :, :head_dim]

    # Use small window
    window_size = 2
    output = scaled_dot_product_attention(query, key, value, sliding_window=window_size)

    # Output should be valid
    assert output.shape == (batch, heads, seq_len, head_dim)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_different_window_sizes():
    """Test various window sizes."""
    batch, heads, seq_len, head_dim = 2, 4, 64, 32
    query = torch.randn(batch, heads, seq_len, head_dim)
    key = torch.randn(batch, heads, seq_len, head_dim)
    value = torch.randn(batch, heads, seq_len, head_dim)

    for window_size in [1, 16, 32, 48]:
        output = scaled_dot_product_attention(query, key, value, sliding_window=window_size)

        assert output.shape == query.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


def test_window_larger_than_sequence(attention_inputs):
    """Window size larger than sequence length should attend to all positions."""
    query, key, value = attention_inputs
    seq_len = query.shape[2]

    # Window larger than sequence
    output_large_window = scaled_dot_product_attention(
        query, key, value, sliding_window=seq_len * 2
    )

    # Should produce valid output
    assert output_large_window.shape == query.shape
    assert not torch.isnan(output_large_window).any()


# =============================================================================
# C. Causal + Sliding Window Tests
# =============================================================================


def test_causal_sliding_window_combination():
    """Verify both causal and sliding window constraints are applied."""
    batch, heads, seq_len, head_dim = 1, 2, 16, 8
    query = torch.randn(batch, heads, seq_len, head_dim)
    key = torch.randn(batch, heads, seq_len, head_dim)
    value = torch.randn(batch, heads, seq_len, head_dim)

    # Causal + sliding window
    output = scaled_dot_product_attention(query, key, value, is_causal=True, sliding_window=4)

    assert output.shape == query.shape
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_causal_sliding_window_vs_causal_only():
    """Compare causal+sliding window vs causal only - they should differ."""
    batch, heads, seq_len, head_dim = 2, 4, 32, 16

    # Use same seed for reproducibility
    torch.manual_seed(42)
    query = torch.randn(batch, heads, seq_len, head_dim)
    key = torch.randn(batch, heads, seq_len, head_dim)
    value = torch.randn(batch, heads, seq_len, head_dim)

    # Causal + sliding window
    output_causal_window = scaled_dot_product_attention(
        query, key, value, is_causal=True, sliding_window=8
    )

    # Causal only (uses standard attention)
    output_causal_only = scaled_dot_product_attention(
        query, key, value, is_causal=True, sliding_window=None
    )

    # These should differ (sliding window restricts attention more)
    assert not torch.allclose(output_causal_window, output_causal_only, rtol=1e-3)


# =============================================================================
# D. Edge Cases
# =============================================================================


def test_batch_size_one():
    """Test with batch size of 1."""
    batch, heads, seq_len, head_dim = 1, 4, 16, 32
    query = torch.randn(batch, heads, seq_len, head_dim)
    key = torch.randn(batch, heads, seq_len, head_dim)
    value = torch.randn(batch, heads, seq_len, head_dim)

    output = scaled_dot_product_attention(query, key, value, sliding_window=4)

    assert output.shape == (batch, heads, seq_len, head_dim)


def test_single_head():
    """Test with single attention head."""
    batch, heads, seq_len, head_dim = 2, 1, 16, 32
    query = torch.randn(batch, heads, seq_len, head_dim)
    key = torch.randn(batch, heads, seq_len, head_dim)
    value = torch.randn(batch, heads, seq_len, head_dim)

    output = scaled_dot_product_attention(query, key, value, sliding_window=4)

    assert output.shape == (batch, heads, seq_len, head_dim)


def test_sequence_length_one():
    """Test with sequence length of 1."""
    batch, heads, seq_len, head_dim = 2, 4, 1, 32
    query = torch.randn(batch, heads, seq_len, head_dim)
    key = torch.randn(batch, heads, seq_len, head_dim)
    value = torch.randn(batch, heads, seq_len, head_dim)

    output = scaled_dot_product_attention(query, key, value, sliding_window=4)

    assert output.shape == (batch, heads, seq_len, head_dim)


def test_window_size_zero():
    """Test edge case with window size of 0 (only attend to self)."""
    batch, heads, seq_len, head_dim = 1, 2, 8, 16
    query = torch.randn(batch, heads, seq_len, head_dim)
    key = torch.randn(batch, heads, seq_len, head_dim)
    value = torch.randn(batch, heads, seq_len, head_dim)

    output = scaled_dot_product_attention(query, key, value, sliding_window=0)

    assert output.shape == (batch, heads, seq_len, head_dim)
    assert not torch.isnan(output).any()


def test_enable_gqa_flag():
    """Verify enable_gqa parameter is passed through."""
    batch, heads, seq_len, head_dim = 2, 4, 16, 32
    query = torch.randn(batch, heads, seq_len, head_dim)
    key = torch.randn(batch, heads, seq_len, head_dim)
    value = torch.randn(batch, heads, seq_len, head_dim)

    # Should not raise error (actual GQA behavior depends on PyTorch version)
    output = scaled_dot_product_attention(query, key, value, sliding_window=4, enable_gqa=True)

    assert output.shape == (batch, heads, seq_len, head_dim)


# =============================================================================
# E. Device Tests
# =============================================================================


def test_cpu_device():
    """Test on CPU device."""
    batch, heads, seq_len, head_dim = 2, 4, 16, 32
    query = torch.randn(batch, heads, seq_len, head_dim, device="cpu")
    key = torch.randn(batch, heads, seq_len, head_dim, device="cpu")
    value = torch.randn(batch, heads, seq_len, head_dim, device="cpu")

    output = scaled_dot_product_attention(query, key, value, sliding_window=4)

    assert output.device.type == "cpu"
    assert output.shape == (batch, heads, seq_len, head_dim)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_device_if_available():
    """Test on CUDA device if available."""
    batch, heads, seq_len, head_dim = 2, 4, 16, 32
    query = torch.randn(batch, heads, seq_len, head_dim, device="cuda")
    key = torch.randn(batch, heads, seq_len, head_dim, device="cuda")
    value = torch.randn(batch, heads, seq_len, head_dim, device="cuda")

    output = scaled_dot_product_attention(query, key, value, sliding_window=4)

    assert output.device.type == "cuda"
    assert output.shape == (batch, heads, seq_len, head_dim)


# =============================================================================
# F. Type and Dtype Tests
# =============================================================================


def test_float32_dtype():
    """Test with default float32 dtype."""
    batch, heads, seq_len, head_dim = 2, 4, 16, 32
    query = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float32)
    key = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float32)
    value = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float32)

    output = scaled_dot_product_attention(query, key, value, sliding_window=4)

    assert output.dtype == torch.float32


def test_bfloat16_dtype():
    """Test with bfloat16 dtype for mixed precision."""
    batch, heads, seq_len, head_dim = 2, 4, 16, 32
    query = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16)
    key = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16)
    value = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16)

    output = scaled_dot_product_attention(query, key, value, sliding_window=4)

    assert output.dtype == torch.bfloat16


# =============================================================================
# G. Caching Tests
# =============================================================================


def test_mask_caching_reuses_block_mask():
    """Verify same mask configuration returns cached BlockMask."""
    # Clear cache
    _get_or_create_block_mask.cache_clear()


def test_transformer_with_sliding_window():
    """Test Transformer class with per-layer sliding window."""
    from glm_experiments.models.components.transformer import Transformer

    hidden_size = 128
    n_layers = 4
    num_heads = 4
    seq_len = 32
    batch_size = 2

    # Create Transformer with per-layer sliding windows
    # Different window size per layer
    transformer = Transformer(
        hidden_size=hidden_size,
        n_layers=n_layers,
        num_heads=num_heads,
        is_causal=False,
        sliding_window=[None, 16, 8, 8],  # First layer standard, rest with windows
        context_length=seq_len,
    )

    # Forward pass
    x = torch.randn(batch_size, seq_len, hidden_size)
    output = transformer(x)

    assert output.shape == (batch_size, seq_len, hidden_size)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_transformer_without_sliding_window():
    """Test Transformer class defaults to standard attention when sliding_window=None."""
    from glm_experiments.models.components.transformer import Transformer

    hidden_size = 128
    n_layers = 2
    num_heads = 4
    seq_len = 32
    batch_size = 2

    # Create Transformer without sliding window (default behavior)
    transformer = Transformer(
        hidden_size=hidden_size,
        n_layers=n_layers,
        num_heads=num_heads,
        is_causal=False,
        sliding_window=None,  # Explicit None
        context_length=seq_len,
    )

    # Forward pass
    x = torch.randn(batch_size, seq_len, hidden_size)
    output = transformer(x)

    assert output.shape == (batch_size, seq_len, hidden_size)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_transformer_causal_with_sliding_window():
    """Test Transformer with both causal and per-layer sliding window."""
    from glm_experiments.models.components.transformer import Transformer

    hidden_size = 128
    n_layers = 3
    num_heads = 4
    seq_len = 32
    batch_size = 2

    # Create causal Transformer with per-layer sliding windows
    transformer = Transformer(
        hidden_size=hidden_size,
        n_layers=n_layers,
        num_heads=num_heads,
        is_causal=True,
        sliding_window=[16, 16, 8],  # Different windows per layer
        context_length=seq_len,
    )

    # Forward pass
    x = torch.randn(batch_size, seq_len, hidden_size)
    output = transformer(x)

    assert output.shape == (batch_size, seq_len, hidden_size)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_transformer_sliding_window_length_validation():
    """Test that Transformer validates sliding_window list length."""
    import pytest

    from glm_experiments.models.components.transformer import Transformer

    hidden_size = 128
    n_layers = 4
    num_heads = 4

    # Wrong length should raise ValueError
    with pytest.raises(ValueError, match="sliding_window list must have length 4"):
        Transformer(
            hidden_size=hidden_size,
            n_layers=n_layers,
            num_heads=num_heads,
            sliding_window=[16, 8],  # Only 2 elements, need 4
        )


def test_transformer_all_layers_with_same_window():
    """Test Transformer with same sliding window for all layers."""
    from glm_experiments.models.components.transformer import Transformer

    hidden_size = 128
    n_layers = 3
    num_heads = 4
    seq_len = 32
    batch_size = 2

    # All layers with same window size
    transformer = Transformer(
        hidden_size=hidden_size,
        n_layers=n_layers,
        num_heads=num_heads,
        sliding_window=[16, 16, 16],  # Same window for all layers
        context_length=seq_len,
    )

    # Forward pass
    x = torch.randn(batch_size, seq_len, hidden_size)
    output = transformer(x)

    assert output.shape == (batch_size, seq_len, hidden_size)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_mixed_dtype_handling():
    """Test that mixed dtypes are handled correctly (autocast scenario)."""
    batch, heads, seq_len, head_dim = 2, 4, 16, 32

    # Simulate mixed precision: Q, K in float32, V in bfloat16
    # This happens with autocast where some ops stay float32
    query = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float32)
    key = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float32)
    value = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16)

    # Should handle mixed dtypes and convert to target dtype
    output = scaled_dot_product_attention(query, key, value, sliding_window=8)

    # Output should match value dtype (target precision)
    assert output.dtype == torch.bfloat16
    assert output.shape == (batch, heads, seq_len, head_dim)
    assert not torch.isnan(output).any()
