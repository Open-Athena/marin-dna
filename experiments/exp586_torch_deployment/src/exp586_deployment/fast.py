"""H100 ceiling experiments, separate from the ordinary deployment path.

Router and shared-projection choices reproduce the BF16 historical winner at
8b0b7997c4fd4c3c5a63377c047b4e52c103d8ff without importing that project.
Compiler mode, freezing and batch shape remain explicit benchmark settings;
constructing this module never changes process-global compiler configuration.
"""

from pathlib import Path

import torch
import torch.nn.functional as F

from exp586_deployment.model import MarinDNAMoE, resolve_checkpoint


def combined_shared_experts(
    x: torch.Tensor,
    gate_up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Combine fixed shared projections, preserving each expert's SwiGLU."""
    projections = F.linear(x, gate_up_weights).chunk(2 * num_experts, dim=-1)
    activations = torch.cat(
        [
            F.silu(projections[2 * expert]) * projections[2 * expert + 1]
            for expert in range(num_experts)
        ],
        dim=-1,
    )
    return F.linear(activations, down_weights)


class FastMarinDNAMoE(MarinDNAMoE):
    """BF16 model with our Triton router and combined shared projections.

    Defaults to vLLM fused experts and FA3; both dependencies are mandatory.
    It must pass exactly the ordinary model's native-JAX correctness gate.
    """

    def __init__(
        self,
        directory: str | Path,
        device: str | torch.device = "cuda",
        *,
        dtype: torch.dtype = torch.bfloat16,
        input_vocab: str | None = "ACGTN",
        output_vocab: str | None = "ACGT",
        attention_provider: str = "vllm",
        flash_attention_version: int = 3,
        moe_backend: str = "vllm",
    ) -> None:
        from exp586_deployment.router_topk import router_top_k

        super().__init__(
            directory,
            device,
            dtype=dtype,
            input_vocab=input_vocab,
            output_vocab=output_vocab,
            attention_provider=attention_provider,
            flash_attention_version=flash_attention_version,
            moe_backend=moe_backend,
            sconv_backend="direct",
            cache_rope=True,
        )
        self.router_top_k = router_top_k
        for layer in range(self.config.num_layers):
            prefixes = [
                f"model.layers.{layer}.shared_experts.{expert}"
                for expert in range(self.config.num_shared_experts)
            ]
            self.register_buffer(
                f"combined_shared_w1_{layer}",
                torch.cat(
                    [
                        self._weight(f"{prefix}.{projection}_proj.weight")
                        for prefix in prefixes
                        for projection in ("gate", "up")
                    ],
                    dim=0,
                ).contiguous(),
                persistent=False,
            )
            self.register_buffer(
                f"combined_shared_w2_{layer}",
                torch.cat(
                    [self._weight(f"{prefix}.down_proj.weight") for prefix in prefixes],
                    dim=1,
                ).contiguous(),
                persistent=False,
            )

    @classmethod
    def from_pretrained(
        cls,
        repository: str | Path,
        *,
        revision: str | None = None,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        input_vocab: str | None = "ACGTN",
        output_vocab: str | None = "ACGT",
        attention_provider: str = "vllm",
        flash_attention_version: int = 3,
        moe_backend: str = "vllm",
    ) -> "FastMarinDNAMoE":
        """Use the same checkpoint loading contract with the fast defaults."""
        return cls(
            resolve_checkpoint(repository, revision),
            device=device,
            dtype=dtype,
            input_vocab=input_vocab,
            output_vocab=output_vocab,
            attention_provider=attention_provider,
            flash_attention_version=flash_attention_version,
            moe_backend=moe_backend,
        )

    def _select_experts(self, biased: torch.Tensor) -> torch.Tensor:
        return self.router_top_k(biased, self.config.experts_per_token)

    def _shared_experts(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        return combined_shared_experts(
            x,
            self.get_buffer(f"combined_shared_w1_{layer}"),
            self.get_buffer(f"combined_shared_w2_{layer}"),
            self.config.num_shared_experts,
        )
