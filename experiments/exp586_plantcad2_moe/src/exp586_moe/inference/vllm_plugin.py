"""Opt-in vLLM patches for CUDA-graph-safe GrugMoE DNA prefills."""

import os
from typing import Any

_ENABLE_ENV = "EXP586_VLLM_CUDAGRAPH"
_PATCHED = False


def register_plugin() -> None:
    """Install exact-workload GrugMoE patches before vLLM builds the model."""
    global _PATCHED
    if os.environ.get(_ENABLE_ENV) != "1" or _PATCHED:
        return

    import torch
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.models.config import GrugMoeForCausalLMConfig
    from vllm.model_executor.models.grugmoe import GrugMoeRouter, GrugMoeShortConv

    from exp586_moe.inference.torch_sconv import short_conv_prefill_out
    from exp586_moe.inference.torch_topk import router_top_k

    # Upstream disables CUDA graphs because the general short-convolution path
    # mutates recurrent state. This benchmark replaces complete 8,192-token
    # prefills with a stateless kernel before model compilation and capture.
    def allow_exact_prefill_graphs(_vllm_config: Any) -> None:
        return

    GrugMoeForCausalLMConfig.verify_and_update_config = staticmethod(
        allow_exact_prefill_graphs
    )

    original_short_conv = GrugMoeShortConv.forward_impl

    def exact_prefill_short_conv(self, hidden_states, output):
        metadata_by_layer = get_forward_context().attn_metadata
        metadata = (
            metadata_by_layer.get(self.prefix)
            if isinstance(metadata_by_layer, dict)
            else None
        )
        if (
            metadata is None
            or metadata.num_decode_tokens != 0
            or metadata.num_prefills == 0
            or metadata.num_prefill_tokens != metadata.num_prefills * 8192
        ):
            return original_short_conv(self, hidden_states, output)
        short_conv_prefill_out(
            hidden_states,
            self.weight,
            output,
            metadata.num_prefill_tokens,
        )

    def exact_prefill_routing(
        self,
        hidden_states,
        router_logits,
        indices_type,
        *,
        input_ids=None,
    ):
        del hidden_states, input_ids
        selected = router_top_k(
            router_logits.float() + self.bias.float(), self.top_k + 1
        )[:, : self.top_k]
        weights = torch.sigmoid(
            torch.gather(router_logits.float(), dim=-1, index=selected)
        )
        weights = weights * (2.5 / (weights.sum(dim=-1, keepdim=True) + 1e-9))
        return weights, selected.to(
            torch.int32 if indices_type is None else indices_type
        )

    GrugMoeShortConv.forward_impl = exact_prefill_short_conv
    GrugMoeRouter._compute_routing = exact_prefill_routing
    _PATCHED = True
