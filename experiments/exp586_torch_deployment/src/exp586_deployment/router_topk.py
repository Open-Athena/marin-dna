"""Historical one-warp router, copied from immutable commit 8b0b7997.

Used only by FastMarinDNAMoE; the ordinary class never imports this module.
"""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


@triton.jit
def _load_tile(
    row_ptr: tl.tensor, n: tl.constexpr, width: tl.constexpr, tile: tl.constexpr
) -> tuple[tl.tensor, tl.tensor]:
    column = tile * width + tl.arange(0, width)
    present = column < n
    bits = tl.load(row_ptr + column, mask=present, other=0)
    key = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)
    return tl.where(present, key, -2147483648), tl.where(present, column, n)


@triton.jit
def _take(
    key: tl.tensor, column: tl.tensor, selected: tl.tensor, n: tl.constexpr
) -> tuple[tl.tensor, tl.tensor]:
    taken = column == selected
    return tl.where(taken, -2147483648, key), tl.where(taken, n, column)


@triton.jit
def _select(
    output: tl.tensor,
    slot: tl.constexpr,
    key0: tl.tensor,
    column0: tl.tensor,
    key1: tl.tensor,
    column1: tl.tensor,
    key2: tl.tensor,
    column2: tl.tensor,
    n: tl.constexpr,
    tiles: tl.constexpr,
) -> tuple[tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor]:
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
    bits: tl.tensor,
    output: tl.tensor,
    n: tl.constexpr,
    width: tl.constexpr,
    tiles: tl.constexpr,
    k: tl.constexpr,
) -> None:
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


@triton_op("exp586_deployment_fast::router_top_k", mutates_args={})
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
