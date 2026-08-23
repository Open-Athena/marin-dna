"""Transformer encoder components.

Adapted from Stanford CS336 (Spring 2025):
https://github.com/stanford-cs336/assignment4-data/blob/main/cs336-basics/cs336_basics/model.py

Key modifications:
- Removed language model wrapper (we only need encoder)
- Made causal masking configurable via is_causal parameter
- Adapted to match GLM Experiments encoder interface
- Added Transformer encoder class that wraps TransformerBlocks
"""

from __future__ import annotations

import math

import einx
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange
from jaxtyping import Float, Int
from torch import Tensor

from glm_experiments.models.components.attention import scaled_dot_product_attention


class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        """A linear layer initialized with truncated normal fan-in fan-out.

        Args:
            d_in: int
                The number of input features.
            d_out: int
                The number of output features.
        """

        super().__init__()
        std = math.sqrt(2 / (d_in + d_out))
        self.weight: Float[Tensor, " d_out d_in"] = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(d_out, d_in), std=std, a=-3 * std, b=3 * std),
            requires_grad=True,
        )

    def forward(self, x: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

    def extra_repr(self):
        return f"d_out={self.weight.shape[0]}, d_in={self.weight.shape[1]}"


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        std = 1.0
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(vocab_size, d_model), std=std, a=-3 * std, b=3 * std
            ),
            requires_grad=True,
        )

    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, " ... d_model"]:
        return self.weight[token_ids, :]

    def extra_repr(self):
        return f"vocab_size={self.weight.shape[0]}, d={self.weight.shape[1]}"


class RotaryEmbedding(nn.Module):
    def __init__(self, context_length: int, dim: int, theta: float = 10000.0):
        super().__init__()
        self.register_buffer(
            "_freq_cis_cache",
            RotaryEmbedding._init_cache(context_length, dim, theta),
            persistent=False,
        )

    @staticmethod
    def _init_cache(
        context_length: int, dim: int, theta: float
    ) -> Float[Tensor, " 2 context_length half_dim"]:
        assert dim % 2 == 0

        d = torch.arange(0, dim, 2) / dim
        freqs = theta**-d
        t = torch.arange(context_length)

        freqs = einsum(t, freqs, "t, f -> t f")

        cos, sin = torch.cos(freqs), torch.sin(freqs)
        return torch.stack((cos, sin))

    def forward(
        self, x: Float[Tensor, " ... seq d"], pos_ids: Int[Tensor, " ... seq"]
    ) -> Float[Tensor, " ... seq d"]:
        x1, x2 = rearrange(x, "... (half_d xy) -> xy ... half_d", xy=2)

        # einx
        cos, sin = einx.get_at(
            "cos_sin [pos] half_dim, ... -> cos_sin ... half_dim", self._freq_cis_cache, pos_ids
        )

        # 2D rotation matrix applied to pairs in x
        x1_rot = cos * x1 - sin * x2
        x2_rot = sin * x1 + cos * x2
        result = einx.rearrange(
            "... x_half, ... x_half -> ... (x_half (1 + 1))", x1_rot, x2_rot
        ).contiguous()
        return result

    def extra_repr(self):
        return f"context_length={self._freq_cis_cache.shape[0]}, dim/2={self._freq_cis_cache.shape[1]}"


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention with configurable causal masking and sliding window.

    This function implements section 3.2.2 of the Transformer paper. In particular,
    given an input tensor of shape `(batch_size, sequence_length, d_model)`, we project
    it to create queries, keys, and values, and then perform multi-headed attention with
    those queries, keys, and values.

    Args:
        d_model: int
            The dimensionality of the model embeddings and sublayer outputs.
        num_heads: int
            Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        positional_encoder: RotaryEmbedding
            The RoPE module to use.
        is_causal: bool
            Whether to use causal masking (default: False for bidirectional attention).
        sliding_window: int | None
            Window size for sliding window attention. If None, uses standard attention
            (default: None).

    Returns:
        Tensor of shape `(batch_size, sequence_length, d_model)`.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        positional_encoder: RotaryEmbedding,
        is_causal: bool = False,
        sliding_window: int | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.is_causal = is_causal
        self.sliding_window = sliding_window

        self.d_k = d_model // num_heads
        self.d_v = self.d_k

        self.q_proj = Linear(self.d_model, self.num_heads * self.d_k)
        self.k_proj = Linear(self.d_model, self.num_heads * self.d_k)
        self.v_proj = Linear(self.d_model, self.num_heads * self.d_v)

        self.output_proj = Linear(self.num_heads * self.d_v, self.d_model)

        self.positional_encoder = positional_encoder  # RoPE

    def forward(
        self,
        x: Float[Tensor, " ... seq d_k"],
        token_positions: Int[Tensor, " ... seq"] | None = None,
    ) -> Float[Tensor, " ... seq d_v"]:
        """
        Args:
            x: The input to perform multi-headed self-attention on.
            positional_ids: The positional indices along the sequence dimension of the input embeddings.

        Returns:
            Self-attention outputs.
        """
        *b, sequence_length, d_model = x.size()
        assert d_model == self.d_model

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Take apart each head from the embedding dimension of Q, K, V to shape (..., num_heads, seq_len, d_k).
        Q, K, V = (
            rearrange(X, "... seq (heads d) -> ... heads seq d", heads=self.num_heads) for X in (Q, K, V)
        )  # fmt: skip

        if token_positions is None:
            token_positions = einx.rearrange(
                "seq -> b... seq", torch.arange(sequence_length, device=x.device), b=[1] * len(b)
            )

        # Duplicate token positions for each head
        token_positions = rearrange(token_positions, "... seq -> ... 1 seq")

        Q = self.positional_encoder(Q, token_positions)
        K = self.positional_encoder(K, token_positions)

        # Shape: (..., num_heads, sequence_length, d_k)
        attn_output = scaled_dot_product_attention(
            query=Q,
            key=K,
            value=V,
            is_causal=self.is_causal,
            sliding_window=self.sliding_window,
            enable_gqa=False,
        )

        # Concatenate the attention output from all heads.
        # (..., sequence_length, num_heads * d_v).
        attn_output = rearrange(
            attn_output, "batch heads seq d_v -> batch seq (heads d_v)"
        ).contiguous()

        # Apply the output projection
        output = self.output_proj(attn_output)
        return output


class TransformerBlock(nn.Module):
    """A single Transformer layer.

    This implements a single layer of the Transformer, as described in section 3.1
    of the paper.

    Args:
        d_model: int
            The dimensionality of the model embeddings and sublayer outputs.
        num_heads: int
            Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff: int
            Dimensionality of the feed-forward inner layer (section 3.3).
        positional_encoder: RotaryEmbedding
            The RoPE module to use.
        is_causal: bool
            Whether to use causal masking (default: False).
        sliding_window: int | None
            Window size for sliding window attention (default: None).

    Returns:
        FloatTensor of shape `(batch_size, sequence_length, d_model)`.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        positional_encoder: RotaryEmbedding,
        is_causal: bool = False,
        sliding_window: int | None = None,
    ):
        super().__init__()
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            positional_encoder=positional_encoder,
            is_causal=is_causal,
            sliding_window=sliding_window,
        )
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)
        self.ln1 = nn.RMSNorm(d_model)
        self.ln2 = nn.RMSNorm(d_model)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: FloatTensor of shape `(batch_size, sequence_length, d_model)`.
                The input to process with the Transformer block.

        Returns:
            FloatTensor of shape `(batch_size, sequence_length, d_model)`.
        """
        # NOTE: this is a pre-norm Transformer, and differs from the original
        # description in the paper.
        # Apply the multi-head self-attention sublayer
        x_attn = self.attn(self.ln1(x))
        attn_sublayer_output = x + x_attn

        # Apply the feed-forward sublayer
        x_ffn = self.ffn(self.ln2(attn_sublayer_output))
        ffn_sublayer_output = attn_sublayer_output + x_ffn
        return ffn_sublayer_output


class Transformer(nn.Module):
    """Transformer encoder (stack of TransformerBlocks).

    Encoder interface: (batch, seq_len, hidden_size) -> (batch, seq_len, hidden_size)

    This is a wrapper around the CS336 TransformerBlock that matches the encoder
    interface expected by the BERT model in this project.

    Args:
        hidden_size: int
            Embedding dimension (d_model).
        n_layers: int
            Number of transformer blocks.
        num_heads: int
            Number of attention heads (must divide hidden_size).
        d_ff: int | None
            FFN intermediate dimension. If None, auto-computed using CS336 formula:
            floor(hidden_size * 8/3 / 64) * 64
        rope_theta: float
            RoPE frequency base (default: 10000.0).
        is_causal: bool
            Enable causal masking (default: False for MLM).
        sliding_window: list[int | None] | None
            Per-layer window sizes for sliding window attention. Can be:
            - None: No sliding window (standard attention for all layers)
            - List of length n_layers: Specific window size per layer (None = standard attention)
            Example: [None, 256, 256, 128] for 4 layers
            (default: None).
        context_length: int
            Maximum sequence length for RoPE cache (default: 512).
    """

    def __init__(
        self,
        hidden_size: int,
        n_layers: int,
        num_heads: int,
        d_ff: int | None = None,
        rope_theta: float = 10000.0,
        is_causal: bool = False,
        sliding_window: list[int | None] | None = None,
        context_length: int = 512,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.num_heads = num_heads
        self.is_causal = is_causal

        # Process sliding_window parameter
        if sliding_window is None:
            # No sliding window for any layer
            self.sliding_window = [None] * n_layers
        else:
            # Validate list length
            if len(sliding_window) != n_layers:
                raise ValueError(
                    f"sliding_window list must have length {n_layers}, got {len(sliding_window)}"
                )
            self.sliding_window = sliding_window

        # Auto-compute d_ff using CS336 formula: floor(d_model * 8/3 / 64) * 64
        if d_ff is None:
            d_ff = int(hidden_size * 8 / 3 / 64) * 64

        self.d_ff = d_ff

        # Create shared RoPE module
        d_head = hidden_size // num_heads
        self.positional_encoder = RotaryEmbedding(
            context_length=context_length, dim=d_head, theta=rope_theta
        )

        # Stack of transformer blocks with per-layer sliding windows
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=hidden_size,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    positional_encoder=self.positional_encoder,
                    is_causal=is_causal,
                    sliding_window=self.sliding_window[i],
                )
                for i in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: FloatTensor of shape `(batch_size, sequence_length, hidden_size)`.

        Returns:
            FloatTensor of shape `(batch_size, sequence_length, hidden_size)`.
        """
        for layer in self.layers:
            x = layer(x)
        return x
