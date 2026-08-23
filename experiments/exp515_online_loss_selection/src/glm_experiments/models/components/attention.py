"""Generalized attention functions with sliding window support.

This module provides a drop-in replacement for F.scaled_dot_product_attention
that supports sliding window attention using PyTorch's FlexAttention API.
"""

from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention.flex_attention import create_block_mask, flex_attention


@lru_cache(maxsize=16)
def _get_or_create_block_mask(
    mask_type: str,
    window_size: int,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    device: str,
):
    """Cache BlockMask objects to avoid recompilation overhead.

    FlexAttention compiles attention kernels based on the mask function. Creating new mask
    functions on every call causes expensive recompilation (can be 2x slower). This cache
    ensures masks are created once per configuration and reused.

    Args:
        mask_type: Type of mask - "sliding_window" or "causal_sliding_window"
        window_size: Size of the sliding window
        batch_size: Batch size
        num_heads: Number of attention heads
        seq_len: Sequence length
        device: Device string (e.g., "cpu" or "cuda:0"). Must be string since
            torch.device isn't hashable for cache key.

    Returns:
        BlockMask object that can be reused across flex_attention calls.
    """
    device_obj = torch.device(device)

    if mask_type == "sliding_window":

        def sliding_window_mask(b, h, q_idx, kv_idx):
            return (q_idx - kv_idx).abs() <= window_size

        mask_fn = sliding_window_mask
    elif mask_type == "causal_sliding_window":

        def causal_sliding_window_mask(b, h, q_idx, kv_idx):
            return (q_idx >= kv_idx) & ((q_idx - kv_idx) <= window_size)

        mask_fn = causal_sliding_window_mask
    else:
        raise ValueError(f"Unknown mask_type: {mask_type}")

    return create_block_mask(
        mask_fn,
        B=batch_size,
        H=num_heads,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device_obj,
    )


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    is_causal: bool = False,
    sliding_window: int | None = None,
    enable_gqa: bool = False,
) -> Tensor:
    """Generalized scaled dot-product attention with sliding window support.

    This function is a drop-in replacement for F.scaled_dot_product_attention that
    adds support for sliding window attention using PyTorch's FlexAttention API.

    When sliding_window=None, falls back to F.scaled_dot_product_attention for
    backward compatibility and optimal performance (no FlexAttention overhead).

    Performance notes:
        - First call with sliding_window creates and compiles the attention kernel (~100-500ms)
        - Subsequent calls with same config reuse cached kernel (fast)
        - For fixed sequence lengths, masks are cached globally using functools.lru_cache

    Args:
        query: Query tensor of shape (batch, num_heads, seq_len, head_dim)
        key: Key tensor of shape (batch, num_heads, seq_len, head_dim)
        value: Value tensor of shape (batch, num_heads, seq_len, head_dim)
        is_causal: Whether to apply causal masking (default: False for bidirectional)
        sliding_window: Window size for sliding window attention. If None, falls back
            to standard attention. If set, each position attends only to positions
            within ±window_size. When combined with is_causal=True, attends to
            previous window_size positions only (default: None)
        enable_gqa: Enable grouped query attention (default: False)

    Returns:
        Attention output of shape (batch, num_heads, seq_len, head_dim)

    Examples:
        Standard bidirectional attention (backward compatible):
        >>> output = scaled_dot_product_attention(q, k, v)

        Causal attention (backward compatible):
        >>> output = scaled_dot_product_attention(q, k, v, is_causal=True)

        Sliding window attention (bidirectional):
        >>> output = scaled_dot_product_attention(q, k, v, sliding_window=256)

        Causal + sliding window attention:
        >>> output = scaled_dot_product_attention(q, k, v, is_causal=True, sliding_window=256)
    """
    # Backward compatible path: use standard PyTorch implementation when no sliding window
    if sliding_window is None:
        return F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            is_causal=is_causal,
            enable_gqa=enable_gqa,
        )

    # FlexAttention path: use sliding window
    batch_size, num_heads, seq_len, head_dim = query.shape

    # FlexAttention requires all tensors to have the same dtype.
    # Unlike F.scaled_dot_product_attention, flex_attention is not in the autocast
    # whitelist, so we need to manually ensure dtype consistency.
    # We match to value.dtype since autocast has already converted it to the target precision.
    target_dtype = value.dtype
    if query.dtype != target_dtype:
        query = query.to(target_dtype)
    if key.dtype != target_dtype:
        key = key.to(target_dtype)

    # Determine mask type based on is_causal flag
    mask_type = "causal_sliding_window" if is_causal else "sliding_window"

    # Get or create cached block mask
    block_mask = _get_or_create_block_mask(
        mask_type=mask_type,
        window_size=sliding_window,
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        device=str(query.device),
    )

    # Use FlexAttention with block mask
    return flex_attention(query, key, value, block_mask=block_mask, enable_gqa=enable_gqa)
