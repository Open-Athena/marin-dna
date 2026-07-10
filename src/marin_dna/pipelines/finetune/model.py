"""Siamese LoRA classifier over a frozen causal gLM's last hidden state (#369).

The shared LoRA-adapted decoder pools each of the ref/alt windows (entire-window mean,
fp32, BOS excluded), forms ``concat_ref_delta = [pool_ref, pool_alt−pool_ref]``, and a
linear head maps it to a logit. Only the LoRA adapters and the head are trainable; the
backbone is frozen. Mirrors the #341 frozen-probe representation exactly at LoRA Δ=0.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from marin_dna.data.transforms import _get_special_token_counts

# Qwen3/Llama-style attention projections (the default, minimal-capacity target set).
ATTENTION_MODULES: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_MODULES: tuple[str, ...] = ("gate_proj", "up_proj", "down_proj")


class SiameseLoRAClassifier(nn.Module):
    """Shared LoRA backbone → pooled ref/alt embeddings → ``concat_ref_delta`` → logit."""

    def __init__(
        self,
        backbone,  # PeftModel wrapping the causal LM (kept for save/adapter state)
        decoder: nn.Module,  # the LoRA-injected decoder (returns last_hidden_state)
        hidden_size: int,
        pool_lo: int,
        pool_hi: int,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.pool_lo = pool_lo
        self.pool_hi = pool_hi
        # Head consumes concat_ref_delta ([pool_ref, pool_alt-pool_ref]) = 2*hidden.
        # fp32 (its input is the fp32-pooled embedding) for a stable, probe-comparable head.
        self.head = nn.Linear(2 * hidden_size, 1).float()

    def pool(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Entire-window mean-pool of the last hidden state, fp32 (BOS excluded).

        Accepts a stacked ``[K*B, L]`` batch so the siamese ref/alt (and the 4 eval
        strands) go through the backbone in **one** forward — halving/quartering the
        kernel launches vs one call per allele.
        """
        h = self.decoder(input_ids=input_ids).last_hidden_state  # [K*B, L, D]
        return h[:, self.pool_lo : self.pool_hi].float().mean(dim=1)  # [K*B, D]

    def _head(self, pool_ref: torch.Tensor, pool_alt: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([pool_ref, pool_alt - pool_ref], dim=-1)  # concat_ref_delta
        return self.head(feat.float()).squeeze(-1)  # [B] logit

    def forward(self, ref_ids: torch.Tensor, alt_ids: torch.Tensor) -> torch.Tensor:
        """Single-strand logit (training path — one strand per example, RC-augmented)."""
        b = ref_ids.shape[0]
        pooled = self.pool(torch.cat([ref_ids, alt_ids], dim=0))  # one forward of 2B
        return self._head(pooled[:b], pooled[b:])

    def logit_rc_avg(
        self,
        ref_fwd: torch.Tensor,
        alt_fwd: torch.Tensor,
        ref_rc: torch.Tensor,
        alt_rc: torch.Tensor,
    ) -> torch.Tensor:
        """FWD+RC pooled-average logit (eval path — matches the probe/pipeline scoring)."""
        b = ref_fwd.shape[0]
        pooled = self.pool(torch.cat([ref_fwd, alt_fwd, ref_rc, alt_rc], dim=0))  # one 4B fwd
        pool_ref = 0.5 * (pooled[:b] + pooled[2 * b : 3 * b])
        pool_alt = 0.5 * (pooled[b : 2 * b] + pooled[3 * b : 4 * b])
        return self._head(pool_ref, pool_alt)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())


def build_model(
    checkpoint_path: str,
    *,
    window_size: int = 255,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: tuple[str, ...] = ATTENTION_MODULES,
    top_k_layers: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
    gradient_checkpointing: bool = False,
) -> tuple[SiameseLoRAClassifier, AutoTokenizer]:
    """Load an HF checkpoint, inject LoRA, and wrap it in the siamese classifier.

    ``top_k_layers`` restricts LoRA to the *last* K decoder layers (a capacity/reg lever
    that also targets the degraded readout, #341) — ``None`` adapts every layer.
    ``gradient_checkpointing`` trades compute for activation memory so bigger rungs (476M+)
    fit on a 24 GB A10G without an A100.
    """
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, trust_remote_code=True, torch_dtype=dtype
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # PEFT: frozen base needs grad-requiring inputs
    n_prefix, _ = _get_special_token_counts(tokenizer)
    hidden = model.config.hidden_size
    n_layers = model.config.num_hidden_layers

    layers_to_transform = None
    if top_k_layers is not None:
        assert 0 < top_k_layers <= n_layers, (top_k_layers, n_layers)
        layers_to_transform = list(range(n_layers - top_k_layers, n_layers))
    lconf = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=list(target_modules),
        layers_to_transform=layers_to_transform,
        layers_pattern="layers" if layers_to_transform is not None else None,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lconf)
    # The LoRA adapters are injected in place into the causal LM's Linear submodules,
    # including those inside the decoder — so calling the decoder directly applies them
    # (and skips the unused lm_head). `.base_model` on the causal LM is the decoder that
    # returns `last_hidden_state` (the same surface scoring.py hooks).
    decoder = peft_model.get_base_model().base_model
    clf = SiameseLoRAClassifier(peft_model, decoder, hidden, n_prefix, n_prefix + window_size)
    return clf, tokenizer
