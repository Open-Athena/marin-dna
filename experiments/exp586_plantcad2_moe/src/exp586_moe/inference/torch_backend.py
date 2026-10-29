"""Correctness-first eager PyTorch implementation of the exported GrugMoE model."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)

try:
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        fused_experts as _vllm_fused_experts,
    )
except ImportError:
    _vllm_fused_experts = None

TorchVariant = Literal[
    "full_head",
    "acgt_head",
    "acgt_grouped_moe",
    "acgt_vllm_moe",
]
TorchMoeBackend = Literal["naive", "grouped", "vllm"]
TorchAttentionBackend = Literal[
    "sdpa_expanded",
    "sdpa_gqa",
    "flex_expanded",
    "flex_gqa",
]
TorchGlobalAttentionBackend = Literal["auto", "cudnn", "flash", "efficient", "math"]


@dataclass(frozen=True)
class TorchConfig:
    """Inference-relevant fields from the exported Hugging Face config."""

    vocab_size: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    sliding_window: int
    global_every: int
    layer_norm_eps: float
    num_experts: int
    experts_per_token: int
    intermediate_dim: int
    num_shared_experts: int
    latent_dim: int
    qk_mult: float
    rope_theta: float
    sconv_kernel: int

    @classmethod
    def load(cls, directory: Path) -> "TorchConfig":
        """Decode the exact field aliases emitted by the exp586 exporter."""
        raw = json.loads((directory / "config.json").read_text())
        return cls(
            vocab_size=int(raw["vocab_size"]),
            hidden_dim=int(raw["hidden_size"]),
            num_layers=int(raw["num_hidden_layers"]),
            num_heads=int(raw["num_attention_heads"]),
            num_kv_heads=int(raw["num_key_value_heads"]),
            head_dim=int(raw["head_dim"]),
            sliding_window=int(raw["sliding_window"]),
            global_every=int(raw["global_every"]),
            layer_norm_eps=float(raw["rms_norm_eps"]),
            num_experts=int(raw["num_experts"]),
            experts_per_token=int(raw["num_experts_per_tok"]),
            intermediate_dim=int(raw["moe_intermediate_size"]),
            num_shared_experts=int(raw["num_shared_experts"]),
            latent_dim=int(raw["latent_dim"]),
            qk_mult=float(raw["qk_mult"]),
            rope_theta=float(raw["rope_theta"]),
            sconv_kernel=int(raw["sconv_kernel"]),
        )


def load_weights(directory: Path, device: torch.device) -> dict[str, torch.Tensor]:
    """Load each content-addressed BF16 shard directly onto the selected GPU."""
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    weights: dict[str, torch.Tensor] = {}
    for shard in shards:
        weights.update(load_file(directory / shard, device=str(device)))
    if set(weights) != set(index["weight_map"]):
        raise ValueError("Loaded weight names differ from the HF index")
    return weights


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """Match the training model's FP32 variance and BF16 output."""
    dtype = x.dtype
    value = x.float()
    value = value * torch.rsqrt(torch.mean(value.square(), dim=-1, keepdim=True) + eps)
    if weight is not None:
        value = value * weight.float()
    return value.to(dtype)


def short_conv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Apply Marin's lag-zero-first depthwise causal convolution."""
    channels = x.shape[-1]
    kernel = weight.transpose(0, 1).flip(-1).unsqueeze(1)
    padded = F.pad(x.transpose(1, 2), (weight.shape[0] - 1, 0))
    return F.conv1d(padded, kernel, groups=channels).transpose(1, 2)


def fused_half_rope(x: torch.Tensor, theta: float) -> torch.Tensor:
    """Apply the Hero run's adjacent-pair RoPE to the first half of each head."""
    sequence_length = x.shape[-2]
    rotary_dim = x.shape[-1] // 2
    half = rotary_dim // 2
    positions = torch.arange(sequence_length, device=x.device, dtype=torch.float32)
    inv_freq = 1.0 / (
        theta ** (torch.arange(half, device=x.device, dtype=torch.float32) / half)
    )
    angles = positions[:, None] * inv_freq[None, :]
    cos = torch.cos(angles).repeat_interleave(2, dim=-1)
    sin = torch.stack((-torch.sin(angles), torch.sin(angles)), dim=-1).reshape(
        sequence_length, rotary_dim
    )
    rotated = x[..., :rotary_dim]
    flipped = rotated.reshape(*rotated.shape[:-1], half, 2).flip(-1).reshape_as(rotated)
    rotated = (cos[None, None] * rotated + sin[None, None] * flipped).to(x.dtype)
    return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)


def grouped_experts(
    routed: torch.Tensor,
    selected: torch.Tensor,
    combine: torch.Tensor,
    gate_weights: torch.Tensor,
    up_weights: torch.Tensor,
    down_weights: torch.Tensor,
) -> torch.Tensor:
    """Run all routed assignments with three native grouped GEMMs."""
    tokens, top_k = selected.shape
    assignment_experts = selected.reshape(-1)
    assignment_tokens = (
        torch.arange(tokens, device=routed.device)[:, None]
        .expand(tokens, top_k)
        .reshape(-1)
    )
    order = torch.argsort(assignment_experts, stable=True)
    sorted_experts = assignment_experts[order]
    sorted_tokens = assignment_tokens[order]
    offsets = (
        torch.bincount(sorted_experts, minlength=gate_weights.shape[0]).cumsum(0).int()
    )
    inputs = routed[sorted_tokens]
    gate = torch._grouped_mm(inputs, gate_weights.transpose(1, 2), offs=offsets)
    up = torch._grouped_mm(inputs, up_weights.transpose(1, 2), offs=offsets)
    outputs = torch._grouped_mm(
        F.silu(gate) * up,
        down_weights.transpose(1, 2),
        offs=offsets,
    )
    weights = combine.reshape(-1)[order, None].to(outputs.dtype)
    result = torch.zeros_like(routed)
    result.index_add_(0, sorted_tokens, outputs * weights)
    return result


class EagerGrugMoE(torch.nn.Module):
    """Plain PyTorch operations with no vLLM engine, EP, or custom MoE kernel."""

    def __init__(
        self,
        directory: Path,
        device: torch.device,
        *,
        attention_backend: TorchAttentionBackend = "sdpa_expanded",
        global_attention_backend: TorchGlobalAttentionBackend = "auto",
        global_attention_gqa: bool = False,
    ) -> None:
        super().__init__()
        if attention_backend not in {
            "sdpa_expanded",
            "sdpa_gqa",
            "flex_expanded",
            "flex_gqa",
        }:
            raise ValueError(f"Unknown attention backend: {attention_backend}")
        global_backends = {"auto", "cudnn", "flash", "efficient", "math"}
        if global_attention_backend not in global_backends:
            raise ValueError(
                f"Unknown global attention backend: {global_attention_backend}"
            )
        self.config = TorchConfig.load(directory)
        weights = load_weights(directory, device)
        self._weight_names: dict[str, str] = {}
        for index, (name, value) in enumerate(weights.items()):
            buffer_name = f"checkpoint_weight_{index}"
            # Own the storage rather than retaining safetensors' mmap-backed weakref;
            # AOTAutograd needs stable buffer storage during graph compilation.
            self.register_buffer(buffer_name, value.clone(), persistent=False)
            self._weight_names[name] = buffer_name
        if _vllm_fused_experts is not None:
            for layer in range(self.config.num_layers):
                prefix = f"model.layers.{layer}.mlp.experts"
                gate = weights[f"{prefix}.gate_proj.weight"]
                up = weights[f"{prefix}.up_proj.weight"]
                self.register_buffer(
                    f"vllm_w1_{layer}",
                    torch.cat((gate, up), dim=1).contiguous(),
                    persistent=False,
                )
        self.device = device
        self.attention_backend = attention_backend
        self.global_attention_backend = global_attention_backend
        self.global_attention_gqa = global_attention_gqa
        self.flex_kernel_options: dict[str, object] = {}
        positions = torch.arange(8192, device=device)
        self.local_mask = (positions[None, :] <= positions[:, None]) & (
            positions[None, :] >= positions[:, None] - (self.config.sliding_window - 1)
        )
        self.local_block_mask: BlockMask | None = None
        if attention_backend in {"flex_expanded", "flex_gqa"}:
            window = self.config.sliding_window

            def local_mask(
                _batch: torch.Tensor,
                _head: torch.Tensor,
                query: torch.Tensor,
                key_value: torch.Tensor,
            ) -> torch.Tensor:
                return (query >= key_value) & (query - key_value < window)

            self.local_block_mask = create_block_mask(
                local_mask,
                B=None,
                H=None,
                Q_LEN=8192,
                KV_LEN=8192,
                device=str(device),
                _compile=True,
            )

    def _weight(self, name: str) -> torch.Tensor:
        return self.get_buffer(self._weight_names[name])

    def _gated_norm(self, prefix: str, x: torch.Tensor) -> torch.Tensor:
        hidden = F.linear(x, self._weight(f"{prefix}.down_proj.weight"))
        gate = torch.sigmoid(
            F.linear(F.silu(hidden), self._weight(f"{prefix}.up_proj.weight"))
        )
        return x * gate.to(x.dtype)

    def _attention(
        self, layer: int, x: torch.Tensor, *, is_global: bool
    ) -> torch.Tensor:
        cfg = self.config
        prefix = f"model.layers.{layer}.self_attn"
        q = F.linear(x, self._weight(f"{prefix}.q_proj.weight"))
        k = F.linear(x, self._weight(f"{prefix}.k_proj.weight"))
        v = F.linear(x, self._weight(f"{prefix}.v_proj.weight"))
        k = short_conv(k, self._weight(f"{prefix}.sconv_k.weight"))
        batch, sequence, _ = q.shape
        q = q.view(batch, sequence, cfg.num_heads, cfg.head_dim).transpose(1, 2)
        k = k.view(batch, sequence, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
        v = v.view(batch, sequence, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
        q = rms_norm(q, None, 1e-6)
        k = rms_norm(k, None, 1e-6)
        if not is_global:
            q = fused_half_rope(q, cfg.rope_theta)
            k = fused_half_rope(k, cfg.rope_theta)
        q = q * cfg.qk_mult
        expand_kv = self.attention_backend in {"sdpa_expanded", "flex_expanded"}
        if is_global and self.global_attention_gqa:
            expand_kv = False
        if expand_kv and cfg.num_heads != cfg.num_kv_heads:
            repeat = cfg.num_heads // cfg.num_kv_heads
            k = k[:, :, None].expand(
                batch, cfg.num_kv_heads, repeat, sequence, cfg.head_dim
            )
            v = v[:, :, None].expand(
                batch, cfg.num_kv_heads, repeat, sequence, cfg.head_dim
            )
            k = k.reshape(batch, cfg.num_heads, sequence, cfg.head_dim)
            v = v.reshape(batch, cfg.num_heads, sequence, cfg.head_dim)
            if is_global:
                # The zero-stride GQA expansion makes SDPA fall back to an unfused
                # FP32 matmul/softmax path. Materializing only the two global-layer
                # K/V views lets the Hopper flash-attention kernel consume them.
                k = k.contiguous()
                v = v.contiguous()
        if self.attention_backend in {"flex_expanded", "flex_gqa"} and not is_global:
            assert self.local_block_mask is not None
            output = flex_attention(
                q,
                k,
                v,
                block_mask=self.local_block_mask,
                enable_gqa=self.attention_backend == "flex_gqa",
                kernel_options=self.flex_kernel_options,
            )
        else:
            backend_map = {
                "cudnn": SDPBackend.CUDNN_ATTENTION,
                "flash": SDPBackend.FLASH_ATTENTION,
                "efficient": SDPBackend.EFFICIENT_ATTENTION,
                "math": SDPBackend.MATH,
            }

            def attention() -> torch.Tensor:
                return F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None if is_global else self.local_mask,
                    is_causal=is_global,
                    enable_gqa=(
                        self.attention_backend == "sdpa_gqa"
                        or (is_global and self.global_attention_gqa)
                    ),
                )

            if is_global and self.global_attention_backend != "auto":
                with sdpa_kernel(backend_map[self.global_attention_backend]):
                    output = attention()
            else:
                output = attention()
        dot = torch.sum(output * v, dim=-1, keepdim=True)
        norm = torch.sum(v * v, dim=-1, keepdim=True)
        output = output - (dot / (norm + 1e-6)) * v
        gate = 2 * torch.sigmoid(
            F.linear(x, self._weight(f"{prefix}.attn_gate.weight"))
        )
        output = output * gate.transpose(1, 2).unsqueeze(-1).to(output.dtype)
        output = output.transpose(1, 2).reshape(batch, sequence, cfg.hidden_dim)
        return F.linear(output, self._weight(f"{prefix}.o_proj.weight"))

    def _moe(
        self,
        layer: int,
        x: torch.Tensor,
        *,
        backend: TorchMoeBackend,
    ) -> torch.Tensor:
        cfg = self.config
        prefix = f"model.layers.{layer}.mlp"
        flat = x.reshape(-1, cfg.hidden_dim)
        # The Hero router dot has the hidden dtype, then casts to FP32 before
        # bias, top-k and sigmoid. The accumulation order can change routing.
        router = F.linear(
            flat, self._weight(f"{prefix}.router.weight").to(flat.dtype)
        ).float()
        biased = router + self._weight(f"{prefix}.router.bias").float()
        selected = torch.topk(
            biased,
            k=cfg.experts_per_token,
            dim=-1,
        ).indices
        combine = torch.sigmoid(torch.gather(router, 1, selected))
        combine = combine * (2.5 / (combine.sum(dim=-1, keepdim=True) + 1e-9))
        routed = F.linear(flat, self._weight(f"{prefix}.latent_down_proj.weight"))
        routed = rms_norm(
            routed,
            self._weight(f"{prefix}.latent_norm.weight"),
            cfg.layer_norm_eps,
        )
        output = torch.zeros_like(routed)
        gate_weights = self._weight(f"{prefix}.experts.gate_proj.weight")
        up_weights = self._weight(f"{prefix}.experts.up_proj.weight")
        down_weights = self._weight(f"{prefix}.experts.down_proj.weight")
        if backend == "vllm":
            if _vllm_fused_experts is None:
                raise RuntimeError("The vLLM fused MoE kernel is unavailable")
            output = _vllm_fused_experts(
                routed.contiguous(),
                self.get_buffer(f"vllm_w1_{layer}"),
                down_weights,
                combine,
                selected.to(torch.int32),
                global_num_experts=cfg.num_experts,
            )
        elif backend == "grouped":
            output = grouped_experts(
                routed,
                selected,
                combine,
                gate_weights,
                up_weights,
                down_weights,
            )
        elif backend == "naive":
            for expert in range(cfg.num_experts):
                rows, slots = torch.where(selected == expert)
                if rows.numel() == 0:
                    continue
                expert_input = routed[rows]
                gate = F.linear(expert_input, gate_weights[expert])
                up = F.linear(expert_input, up_weights[expert])
                expert_output = F.linear(F.silu(gate) * up, down_weights[expert])
                weighted = expert_output * combine[rows, slots, None].to(
                    expert_output.dtype
                )
                output.index_add_(0, rows, weighted)
        else:
            raise ValueError(f"Unknown MoE backend: {backend}")
        output = F.linear(output, self._weight(f"{prefix}.latent_up_proj.weight"))
        return output.reshape_as(x)

    def _shared_experts(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(x)
        for expert in range(self.config.num_shared_experts):
            prefix = f"model.layers.{layer}.shared_experts.{expert}"
            gate = F.linear(x, self._weight(f"{prefix}.gate_proj.weight"))
            up = F.linear(x, self._weight(f"{prefix}.up_proj.weight"))
            output = output + F.linear(
                F.silu(gate) * up, self._weight(f"{prefix}.down_proj.weight")
            )
        return output

    def forward(
        self, token_ids: torch.Tensor, base_ids: list[int], *, variant: TorchVariant
    ) -> torch.Tensor:
        cfg = self.config
        hidden = F.embedding(token_ids, self._weight("model.embed_tokens.weight"))
        hidden = self._gated_norm(
            "model.embed_gated_norm",
            rms_norm(
                hidden, self._weight("model.embed_norm.weight"), cfg.layer_norm_eps
            ),
        )
        for layer in range(cfg.num_layers):
            prefix = f"model.layers.{layer}"
            is_global = (
                layer + 1
            ) % cfg.global_every == 0 or layer == cfg.num_layers - 1
            attn_input = self._gated_norm(
                f"{prefix}.attn_gated_norm",
                rms_norm(
                    hidden,
                    self._weight(f"{prefix}.input_layernorm.weight"),
                    cfg.layer_norm_eps,
                ),
            )
            attention = self._attention(layer, attn_input, is_global=is_global)
            hidden = hidden + short_conv(
                attention, self._weight(f"{prefix}.sconv_attn.weight")
            )
            mlp_input = self._gated_norm(
                f"{prefix}.mlp_gated_norm",
                rms_norm(
                    hidden,
                    self._weight(f"{prefix}.post_attention_layernorm.weight"),
                    cfg.layer_norm_eps,
                ),
            )
            moe_backend: TorchMoeBackend = "naive"
            if variant == "acgt_grouped_moe":
                moe_backend = "grouped"
            elif variant == "acgt_vllm_moe":
                moe_backend = "vllm"
            mlp = self._moe(
                layer, mlp_input, backend=moe_backend
            ) + self._shared_experts(layer, mlp_input)
            hidden = hidden + short_conv(
                mlp, self._weight(f"{prefix}.sconv_mlp.weight")
            )
        hidden = self._gated_norm(
            "model.final_gated_norm",
            rms_norm(hidden, self._weight("model.norm.weight"), cfg.layer_norm_eps),
        )
        head = self._weight("lm_head.weight")
        if variant != "full_head":
            head = head[base_ids]
        logits = F.linear(hidden, head)
        return logits[..., base_ids] if variant == "full_head" else logits
