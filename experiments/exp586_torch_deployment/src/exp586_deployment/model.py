"""BF16 Grug/Hero math, independent of the historical inference benchmark."""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tokenizers import Tokenizer

from exp586_deployment.contract import token_contract
from exp586_deployment.providers import (
    RuntimeConfig,
    load_attention,
    load_fused_experts,
)


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


def resolve_checkpoint(repository: str | Path, revision: str | None) -> Path:
    """Resolve an existing export or download the selected HF snapshot."""
    directory = Path(repository)
    if directory.is_dir():
        return directory
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            str(repository),
            revision=revision,
            allow_patterns=[
                "config.json",
                "tokenizer.json",
                "model.safetensors.index.json",
                "*.safetensors",
            ],
        )
    )


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """Match the training model's FP32 variance and BF16 output."""
    dtype = x.dtype
    value = x.float()
    value = value * torch.rsqrt(torch.mean(value.square(), dim=-1, keepdim=True) + eps)
    if weight is not None:
        value = value * weight.float()
    return value.to(dtype)


def short_conv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Evaluate the lag-zero-first causal depthwise filter with ordinary Torch ops."""
    result = x * weight[0]
    for lag in range(1, min(weight.shape[0], x.shape[1])):
        result = result + F.pad(x[:, :-lag] * weight[lag], (0, 0, lag, 0))
    return result


def depthwise_short_conv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Use the standard conv1d implementation of the same causal filter."""
    return F.conv1d(
        F.pad(x.transpose(1, 2), (weight.shape[0] - 1, 0)),
        weight.transpose(0, 1).flip(-1).unsqueeze(1),
        groups=x.shape[-1],
    ).transpose(1, 2)


def validate_vocab(vocab: str | None) -> None:
    """A retained alphabet has distinct trained DNA symbols; None keeps all rows."""
    if vocab is not None and (
        not isinstance(vocab, str)
        or not vocab
        or set(vocab) - set("ACGTN")
        or len(set(vocab)) != len(vocab)
    ):
        raise ValueError("vocab must contain distinct uppercase A/C/G/T/N, or be None")


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
    gate = F.grouped_mm(inputs, gate_weights.transpose(1, 2), offs=offsets)
    up = F.grouped_mm(inputs, up_weights.transpose(1, 2), offs=offsets)
    outputs = F.grouped_mm(
        F.silu(gate) * up,
        down_weights.transpose(1, 2),
        offs=offsets,
    )
    weights = combine.reshape(-1)[order, None].to(outputs.dtype)
    result = torch.zeros_like(routed)
    result.index_add_(0, sorted_tokens, outputs * weights)
    return result


class MarinDNAMoE(torch.nn.Module):
    """DNA MoE with explicit precision, vocabulary and implementation choices.

    Attention and routed-expert providers are independently selected.
    Selecting an unavailable provider always raises; there is no fallback.
    Encoding and compilation belong to the caller, outside this actual module.
    """

    def __init__(
        self,
        directory: str | Path,
        device: str | torch.device = "cuda",
        *,
        dtype: torch.dtype = torch.bfloat16,
        input_vocab: str | None = "ACGTN",
        output_vocab: str | None = "ACGT",
        attention_provider: str = "flash_attn",
        flash_attention_version: int = 2,
        moe_backend: str = "torch",
        sconv_backend: str = "direct",
        cache_rope: bool = True,
    ) -> None:
        super().__init__()
        if dtype not in {torch.bfloat16, torch.float16}:
            raise ValueError("dtype must be torch.bfloat16 or torch.float16")
        validate_vocab(input_vocab)
        validate_vocab(output_vocab)
        if sconv_backend not in {"direct", "conv1d"}:
            raise ValueError("sconv_backend must be direct or conv1d")
        if not isinstance(cache_rope, bool):
            raise TypeError("cache_rope must be a bool")
        runtime = RuntimeConfig(
            attention_provider, flash_attention_version, moe_backend
        )
        if dtype == torch.float16 and runtime.moe_backend == "torch":
            raise ValueError(
                "The native Torch MoE backend requires bfloat16 with the pinned CUDA 13.2 build"
            )
        directory, device = Path(directory), torch.device(device)
        self.config = TorchConfig.load(directory)
        self.runtime = runtime
        self.dtype = dtype
        self.input_vocab, self.output_vocab = input_vocab, output_vocab
        self.cache_rope = cache_rope
        self.short_conv = (
            short_conv if sconv_backend == "direct" else depthwise_short_conv
        )
        self.sconv_backend = sconv_backend
        self.attend = load_attention(
            runtime.attention_provider, runtime.flash_attention_version
        )
        self.fused_experts = (
            load_fused_experts() if runtime.moe_backend == "vllm" else None
        )
        contract = token_contract(
            Tokenizer.from_file(str(directory / "tokenizer.json"))
        )
        weights = load_weights(directory, device)
        self._weight_names: dict[str, str] = {}
        for i, (name, value) in enumerate(weights.items()):
            if name == "model.embed_tokens.weight" and input_vocab is not None:
                value = value[[contract.input_ids[base] for base in input_vocab]]
            if name == "lm_head.weight" and output_vocab is not None:
                value = value[[contract.input_ids[base] for base in output_vocab]]
            value = value.to(dtype=dtype)
            buffer = f"checkpoint_weight_{i}"
            self.register_buffer(buffer, value.clone(), persistent=False)
            self._weight_names[name] = buffer
        if runtime.moe_backend == "vllm":
            for layer in range(self.config.num_layers):
                prefix = f"model.layers.{layer}.mlp.experts"
                packed = torch.cat(
                    (
                        self._weight(f"{prefix}.gate_proj.weight"),
                        self._weight(f"{prefix}.up_proj.weight"),
                    ),
                    dim=1,
                ).contiguous()
                self.register_buffer(f"vllm_w1_{layer}", packed, persistent=False)
        if cache_rope:
            self.prepare_rope(8192, device)
        self.eval()

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
        attention_provider: str = "flash_attn",
        flash_attention_version: int = 2,
        moe_backend: str = "torch",
        sconv_backend: str = "direct",
        cache_rope: bool = True,
    ) -> "MarinDNAMoE":
        """Load a local export or fetch a pinned Hugging Face snapshot."""
        return cls(
            resolve_checkpoint(repository, revision),
            device,
            dtype=dtype,
            input_vocab=input_vocab,
            output_vocab=output_vocab,
            attention_provider=attention_provider,
            flash_attention_version=flash_attention_version,
            moe_backend=moe_backend,
            sconv_backend=sconv_backend,
            cache_rope=cache_rope,
        )

    def prepare_rope(self, sequence_length: int, device: torch.device) -> None:
        """Prepare ordinary rotary caches outside compiled execution for a request shape."""
        rotary_dim = self.config.head_dim // 2
        half = rotary_dim // 2
        positions = torch.arange(sequence_length, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (
            self.config.rope_theta
            ** (torch.arange(half, device=device, dtype=torch.float32) / half)
        )
        angles = positions[:, None] * inv_freq[None, :]
        self.register_buffer(
            "rope_cos", torch.cos(angles).repeat_interleave(2, dim=-1), persistent=False
        )
        self.register_buffer(
            "rope_sin",
            torch.stack((-torch.sin(angles), torch.sin(angles)), dim=-1).reshape(
                sequence_length, rotary_dim
            ),
            persistent=False,
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
        k = self.short_conv(k, self._weight(f"{prefix}.sconv_k.weight"))
        batch, sequence, _ = q.shape
        q = q.view(batch, sequence, cfg.num_heads, cfg.head_dim).transpose(1, 2)
        k = k.view(batch, sequence, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
        v = v.view(batch, sequence, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
        q = rms_norm(q, None, 1e-6)
        k = rms_norm(k, None, 1e-6)
        if not is_global:
            q = self._precomputed_half_rope(q)
            k = self._precomputed_half_rope(k)
        q = q * cfg.qk_mult
        q_packed = q.transpose(1, 2).reshape(
            batch * sequence, cfg.num_heads, cfg.head_dim
        )
        k_packed = k.transpose(1, 2).reshape(
            batch * sequence, cfg.num_kv_heads, cfg.head_dim
        )
        v_packed = v.transpose(1, 2).reshape(
            batch * sequence, cfg.num_kv_heads, cfg.head_dim
        )
        cu_seqlens = torch.arange(
            0, (batch + 1) * sequence, sequence, device=q.device, dtype=torch.int32
        )
        output = self.attend(
            q_packed,
            k_packed,
            v_packed,
            cu_seqlens,
            sequence,
            -1 if is_global else cfg.sliding_window - 1,
        )
        output = output.reshape(batch, sequence, cfg.num_heads, cfg.head_dim).transpose(
            1, 2
        )
        dot = torch.sum(output * v, dim=-1, keepdim=True)
        norm = torch.sum(v * v, dim=-1, keepdim=True)
        output = output - (dot / (norm + 1e-6)) * v
        gate = 2 * torch.sigmoid(
            F.linear(x, self._weight(f"{prefix}.attn_gate.weight"))
        )
        output = output * gate.transpose(1, 2).unsqueeze(-1).to(output.dtype)
        output = output.transpose(1, 2).reshape(batch, sequence, cfg.hidden_dim)
        return F.linear(output, self._weight(f"{prefix}.o_proj.weight"))

    def _precomputed_half_rope(self, x: torch.Tensor) -> torch.Tensor:
        rotary_dim = x.shape[-1] // 2
        half = rotary_dim // 2
        rotated = x[..., :rotary_dim]
        flipped = (
            rotated.reshape(*rotated.shape[:-1], half, 2).flip(-1).reshape_as(rotated)
        )
        if self.cache_rope:
            cos, sin = self.rope_cos[: x.shape[-2]], self.rope_sin[: x.shape[-2]]
        else:
            positions = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)
            inv_freq = 1.0 / (
                self.config.rope_theta
                ** (torch.arange(half, device=x.device, dtype=torch.float32) / half)
            )
            angles = positions[:, None] * inv_freq[None, :]
            cos = torch.cos(angles).repeat_interleave(2, dim=-1)
            sin = torch.stack((-torch.sin(angles), torch.sin(angles)), dim=-1).reshape(
                x.shape[-2], rotary_dim
            )
        rotated = (cos[None, None] * rotated + sin[None, None] * flipped).to(x.dtype)
        return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)

    def _moe(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        prefix = f"model.layers.{layer}.mlp"
        flat = x.reshape(-1, cfg.hidden_dim)
        router = F.linear(
            flat, self._weight(f"{prefix}.router.weight").to(flat.dtype)
        ).float()
        biased = router + self._weight(f"{prefix}.router.bias").float()
        selected = self._select_experts(biased)
        combine = torch.sigmoid(torch.gather(router, 1, selected))
        combine = combine * (2.5 / (combine.sum(dim=-1, keepdim=True) + 1e-9))
        routed = F.linear(flat, self._weight(f"{prefix}.latent_down_proj.weight"))
        routed = rms_norm(
            routed, self._weight(f"{prefix}.latent_norm.weight"), cfg.layer_norm_eps
        )
        if self.fused_experts is not None:
            output = self.fused_experts(
                routed.contiguous(),
                self.get_buffer(f"vllm_w1_{layer}"),
                self._weight(f"{prefix}.experts.down_proj.weight"),
                combine,
                selected.to(torch.int32),
                global_num_experts=cfg.num_experts,
                quant_config=None,
            )
        else:
            output = grouped_experts(
                routed,
                selected,
                combine,
                self._weight(f"{prefix}.experts.gate_proj.weight"),
                self._weight(f"{prefix}.experts.up_proj.weight"),
                self._weight(f"{prefix}.experts.down_proj.weight"),
            )
        output = F.linear(output, self._weight(f"{prefix}.latent_up_proj.weight"))
        return output.reshape_as(x)

    def _select_experts(self, biased: torch.Tensor) -> torch.Tensor:
        return torch.topk(biased, k=self.config.experts_per_token, dim=-1).indices

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
        self,
        token_ids: torch.Tensor,
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
            hidden = hidden + self.short_conv(
                attention,
                self._weight(f"{prefix}.sconv_attn.weight"),
            )
            mlp_input = self._gated_norm(
                f"{prefix}.mlp_gated_norm",
                rms_norm(
                    hidden,
                    self._weight(f"{prefix}.post_attention_layernorm.weight"),
                    cfg.layer_norm_eps,
                ),
            )
            mlp = self._moe(layer, mlp_input) + self._shared_experts(layer, mlp_input)
            hidden = hidden + self.short_conv(
                mlp,
                self._weight(f"{prefix}.sconv_mlp.weight"),
            )
        hidden = self._gated_norm(
            "model.final_gated_norm",
            rms_norm(hidden, self._weight("model.norm.weight"), cfg.layer_norm_eps),
        )
        return F.linear(hidden, self._weight("lm_head.weight"))
