"""ByteNet encoder architecture for genomic language models.

This module implements the ByteNet architecture with dilated convolutions for hierarchical
pattern recognition, adapted from the GPN (Genomic Pre-trained Network) repository.
"""

import torch
import torch.nn as nn


class TransposeLayer(nn.Module):
    """Transpose layer for switching between (B, L, C) and (B, C, L) format.

    Copied from GPN repository.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose dimensions 1 and 2.

        Args:
            x: Input tensor of shape (B, L, C) or (B, C, L)

        Returns:
            Transposed tensor of shape (B, C, L) or (B, L, C)
        """
        return torch.transpose(x, 1, 2)


class ByteNetLayer(nn.Module):
    """Single ByteNet layer with dilated convolution and residual connection.

    Architecture:
        LayerNorm → GELU → Linear (down-projection)
        → LayerNorm → GELU → Transpose → Conv1d → Transpose
        → LayerNorm → GELU → Linear (up-projection)
        + Residual connection

    Copied from GPN repository.

    Args:
        hidden_size: Hidden dimension size
        kernel_size: Convolution kernel size
        dilation: Dilation rate for dilated convolution
        slim: If True, use hidden_size//2 for intermediate size (default: True)
        bias: Whether to use bias in layers (default: True)
        groups: Number of groups for grouped convolution (default: 1)
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        dilation: int,
        slim: bool = True,
        bias: bool = True,
        groups: int = 1,
    ):
        super().__init__()
        intermediate_size = hidden_size // 2 if slim else hidden_size
        self.layer = nn.Sequential(
            nn.LayerNorm(hidden_size, bias=bias),
            nn.GELU(),
            nn.Linear(hidden_size, intermediate_size, bias=bias),
            nn.LayerNorm(intermediate_size, bias=bias),
            nn.GELU(),
            TransposeLayer(),
            nn.Conv1d(
                in_channels=intermediate_size,
                out_channels=intermediate_size,
                kernel_size=kernel_size,
                dilation=dilation,
                padding="same",
                bias=bias,
                groups=groups,
            ),
            TransposeLayer(),
            nn.LayerNorm(intermediate_size, bias=bias),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size)

        Returns:
            Output tensor of shape (batch, seq_len, hidden_size)
        """
        return x + self.layer(x)


class ByteNet(nn.Module):
    """Stack of ByteNetLayers with dilation schedule.

    Args:
        hidden_size: Hidden dimension size
        n_layers: Number of ByteNet layers
        slim: Use slim version (fewer parameters)
        dilation_base: Base for dilation schedule (default: 2)
        dilation_cycle: Cycle length for dilation (default: 8)
        first_kernel_size: Kernel size for first layer (default: 9)
        rest_kernel_size: Kernel size for other layers (default: 5)
    """

    def __init__(
        self,
        hidden_size: int = 512,
        n_layers: int = 16,
        slim: bool = True,
        dilation_base: int = 2,
        dilation_cycle: int = 8,
        first_kernel_size: int = 9,
        rest_kernel_size: int = 5,
        bias: bool = True,
    ):
        super().__init__()

        # ByteNet uses bidirectional context (not causal)
        self.is_causal = False

        # Dilation schedule: [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, ...]
        dilations = [dilation_base ** (i % dilation_cycle) for i in range(n_layers)]

        # Kernel sizes: first layer uses first_kernel_size, rest use rest_kernel_size
        kernel_sizes = [first_kernel_size] + [rest_kernel_size] * (n_layers - 1)

        # Stack of ByteNetLayers
        self.layer = nn.Sequential(
            *[
                ByteNetLayer(
                    hidden_size=hidden_size,
                    kernel_size=kernel_sizes[i],
                    dilation=dilations[i],
                    slim=slim,
                    bias=bias,
                )
                for i in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ByteNet stack.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size)

        Returns:
            Output tensor of shape (batch, seq_len, hidden_size)
        """
        return self.layer(x)
