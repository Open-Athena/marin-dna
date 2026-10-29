"""A one-warp router top-k for narrow expert dimensions."""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


@triton.jit
def _load_tile(row_ptr, n: tl.constexpr, width: tl.constexpr, tile: tl.constexpr):
    column = tile * width + tl.arange(0, width)
    present = column < n
    bits = tl.load(row_ptr + column, mask=present, other=0)
    key = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)
    return tl.where(present, key, -2147483648), tl.where(present, column, n)


@triton.jit
def _take(key, column, selected, n: tl.constexpr):
    taken = column == selected
    return tl.where(taken, -2147483648, key), tl.where(taken, n, column)


@triton.jit
def _select(
    output,
    slot,
    key0,
    column0,
    key1,
    column1,
    key2,
    column2,
    n: tl.constexpr,
    tiles: tl.constexpr,
):
    best = key0
    if tiles == 3:
        best = tl.maximum(tl.maximum(best, key1), key2)
    best = tl.max(best, axis=0)
    selected = tl.where(key0 == best, column0, n)
    if tiles == 3:
        selected = tl.minimum(selected, tl.where(key1 == best, column1, n))
        selected = tl.minimum(selected, tl.where(key2 == best, column2, n))
    selected = tl.min(selected, axis=0)
    tl.store(output + slot, selected)
    key0, column0 = _take(key0, column0, selected, n)
    if tiles == 3:
        key1, column1 = _take(key1, column1, selected, n)
        key2, column2 = _take(key2, column2, selected, n)
    return key0, column0, key1, column1, key2, column2


@triton.jit
def _top_k_rows_kernel(
    bits,
    output,
    n: tl.constexpr,
    width: tl.constexpr,
    tiles: tl.constexpr,
    k: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    row_bits = bits + row * n
    row_output = output + row * k
    key0, column0 = _load_tile(row_bits, n, width, 0)
    key1, column1, key2, column2 = key0, column0, key0, column0
    if tiles == 3:
        key1, column1 = _load_tile(row_bits, n, width, 1)
        key2, column2 = _load_tile(row_bits, n, width, 2)
    for slot in tl.static_range(k):
        key0, column0, key1, column1, key2, column2 = _select(
            row_output,
            slot,
            key0,
            column0,
            key1,
            column1,
            key2,
            column2,
            n,
            tiles,
        )


@triton_op("exp586::router_top_k", mutates_args={})
def _router_top_k(bits: torch.Tensor, k: int) -> torch.Tensor:
    rows, n = bits.shape
    one = triton.next_power_of_2(n)
    third = triton.next_power_of_2(triton.cdiv(n, 3))
    width, tiles = (third, 3) if 3 * third < one else (one, 1)
    output = torch.empty((rows, k), device=bits.device, dtype=torch.int64)
    wrap_triton(_top_k_rows_kernel)[(rows,)](
        bits,
        output,
        n=n,
        width=width,
        tiles=tiles,
        k=k,
        num_warps=1,
        num_stages=1,
    )
    return output


def router_top_k(x: torch.Tensor, k: int) -> torch.Tensor:
    """Return stable top-k indices for a CUDA float32 ``[tokens, experts]`` array."""
    if x.ndim != 2:
        raise ValueError(f"Expected [tokens, experts], got {tuple(x.shape)}")
    if x.dtype != torch.float32 or not x.is_cuda or x.shape[-1] >= 1024:
        return torch.topk(x, k=k, dim=-1).indices
    return _router_top_k(x.view(torch.int32), k)
