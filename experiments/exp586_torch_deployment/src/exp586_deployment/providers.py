"""Explicit, lazy optional attention and routed-expert providers."""

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache, lru_cache
from typing import Any, Literal

import torch


@dataclass(frozen=True)
class RuntimeConfig:
    """Provider availability never silently changes a selected implementation."""

    attention_provider: Literal["flash_attn", "vllm"] = "flash_attn"
    flash_attention_version: Literal[2, 3] = 2
    moe_backend: Literal["torch", "vllm"] = "torch"

    def __post_init__(self) -> None:
        if self.attention_provider not in {"flash_attn", "vllm"}:
            raise ValueError("attention_provider must be flash_attn or vllm")
        if self.flash_attention_version not in {2, 3}:
            raise ValueError("flash_attention_version must be 2 or 3")
        if self.moe_backend not in {"torch", "vllm"}:
            raise ValueError("moe_backend must be torch or vllm")


def optional_module(name: str) -> Any:
    """Explain a missing provider without falling back to a different kernel."""
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise RuntimeError(
            f"Selected provider requires importable {name}: {error}"
        ) from error


@cache
def load_attention(provider: str, version: int) -> Callable[..., torch.Tensor]:
    """Wrap the selected varlen implementation as a fullgraph-compatible operation."""
    RuntimeConfig(attention_provider=provider, flash_attention_version=version)
    if provider == "vllm":
        function = optional_module("vllm.vllm_flash_attn").flash_attn_varlen_func
    elif version == 2:
        function = optional_module("flash_attn").flash_attn_varlen_func
    else:
        function = optional_module("flash_attn_interface").flash_attn_varlen_func

    @torch.library.custom_op(
        f"exp586_deployment::attention_{provider}_fa{version}", mutates_args=()
    )
    def attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        sequence_length: int,
        window_left: int,
    ) -> torch.Tensor:
        if provider == "vllm":
            result = function(
                q,
                k,
                v,
                sequence_length,
                cu_seqlens,
                sequence_length,
                cu_seqlens,
                causal=True,
                window_size=None if window_left < 0 else [window_left, 0],
                fa_version=version,
            )
        else:
            result = function(
                q,
                k,
                v,
                cu_seqlens,
                cu_seqlens,
                sequence_length,
                sequence_length,
                causal=True,
                window_size=(-1, -1) if window_left < 0 else (window_left, 0),
            )
        result = result[0] if isinstance(result, tuple) else result
        return result.contiguous()

    @attention.register_fake
    def fake_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        sequence_length: int,
        window_left: int,
    ) -> torch.Tensor:
        return torch.empty(q.shape, device=q.device, dtype=q.dtype)

    return attention


@lru_cache(maxsize=1)
def load_fused_experts() -> Callable[..., torch.Tensor]:
    """Import vLLM expert utilities only for an explicitly selected fused backend."""
    return optional_module(
        "vllm.model_executor.layers.fused_moe.fused_moe"
    ).fused_experts
