"""A fused fixed-window short convolution for full-sequence inference."""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


@triton.jit
def _short_conv_kernel(
    values,
    weights,
    output,
    elements: tl.constexpr,
    channels: tl.constexpr,
    sequence_length: tl.constexpr,
    kernel_size: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    present = offsets < elements
    token = offsets // channels
    channel = offsets - token * channels
    position = token % sequence_length
    result = tl.zeros((block_size,), dtype=tl.float32)
    for lag in tl.static_range(kernel_size):
        source = offsets - lag * channels
        value = tl.load(
            values + source,
            mask=present & (position >= lag),
            other=0.0,
        )
        weight = tl.load(weights + lag * channels + channel, mask=present, other=0.0)
        result += value * weight
    tl.store(output + offsets, result, mask=present)


@triton_op("exp586::short_conv_prefill_out", mutates_args={"output"})
def _short_conv_prefill_out(
    values: torch.Tensor,
    weights: torch.Tensor,
    output: torch.Tensor,
    count: int,
    sequence_length: int,
) -> None:
    channels = values.shape[-1]
    kernel_size = weights.shape[0]
    elements = count * channels
    block_size = 256
    wrap_triton(_short_conv_kernel)[(triton.cdiv(elements, block_size),)](
        values,
        weights,
        output,
        elements=elements,
        channels=channels,
        sequence_length=sequence_length,
        kernel_size=kernel_size,
        block_size=block_size,
        num_warps=4,
        num_stages=1,
    )


def short_conv_prefill_out(
    values: torch.Tensor,
    weights: torch.Tensor,
    output: torch.Tensor,
    count: int,
    sequence_length: int = 8192,
) -> None:
    """Apply lag-zero-first depthwise convolution into ``output``."""
    if values.ndim != 2 or output.shape != values.shape:
        raise ValueError("Expected matching flattened [tokens, channels] tensors")
    if weights.ndim != 2 or weights.shape[1] != values.shape[1]:
        raise ValueError("Short-convolution weights do not match the hidden size")
    _short_conv_prefill_out(values, weights, output, count, sequence_length)
