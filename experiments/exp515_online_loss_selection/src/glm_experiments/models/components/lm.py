"""Language-model objectives and the issue #515 Hugging Face CLM adapter."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from glm_experiments.models.components.selection import SelectorMode, TokenSelector


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return a mean over mask entries or a differentiable zero."""

    denominator = mask.sum().clamp_min(1)
    return values.masked_select(mask).sum() / denominator


class LM(nn.Module):
    """Base language model with the vendored weighted-token objective."""

    def __init__(
        self,
        embedder: nn.Module,
        encoder: nn.Module,
        layer_norm: nn.Module,
        decoder: nn.Module,
    ) -> None:
        super().__init__()
        self.embedder = embedder
        self.encoder = encoder
        self.layer_norm = layer_norm
        self.decoder = decoder

    def get_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute logits from token IDs."""

        del attention_mask
        x = self.embedder(input_ids.long())
        x = self.encoder(x)
        x = self.layer_norm(x)
        return self.decoder(x)

    def prepare_for_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        soft_masked: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare objective-specific tensors for the vendored loss."""

        raise NotImplementedError("Subclasses must implement prepare_for_loss")

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        soft_masked: torch.Tensor,
        soft_masked_weight: float,
    ) -> dict[str, torch.Tensor]:
        """Compute the original weighted cross-entropy variants."""

        loss_per_token = F.cross_entropy(logits, labels, reduction="none")
        weight_full = torch.ones_like(loss_per_token)
        weight_non_soft_masked = (~soft_masked).float()
        weight_training = torch.where(soft_masked, soft_masked_weight, 1.0)

        def normalize_and_sum(loss: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            weight_sum = weight.sum()
            if float(weight_sum.detach().cpu()) > 0:
                return (loss * weight / weight_sum).sum()
            return loss.sum() * 0.0

        return {
            "loss": normalize_and_sum(loss_per_token, weight_training),
            "loss_full": normalize_and_sum(loss_per_token, weight_full),
            "loss_non_soft_masked": normalize_and_sum(
                loss_per_token,
                weight_non_soft_masked,
            ),
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        soft_masked: torch.Tensor,
        soft_masked_weight: float,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run the vendored weighted-token objective."""

        logits = self.get_logits(input_ids, attention_mask=attention_mask)
        logits, labels, soft_masked = self.prepare_for_loss(logits, labels, soft_masked)
        return self.compute_loss(logits, labels, soft_masked, soft_masked_weight)


class GeneralMaskedLM(LM):
    """Base class for the vendored masked and diffusion language models."""

    def prepare_for_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        soft_masked: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Filter to labels that participate in the masked objective."""

        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1).long()
        soft_masked = soft_masked.view(-1)
        mask = labels != -100
        return logits[mask], labels[mask], soft_masked[mask]


class MLM(GeneralMaskedLM):
    """Masked language model."""


class DLM(GeneralMaskedLM):
    """Diffusion language model."""


class CLM(LM):
    """Causal LM with optional per-sequence nonrepeat target selection."""

    def __init__(
        self,
        embedder: nn.Module,
        encoder: nn.Module,
        layer_norm: nn.Module,
        decoder: nn.Module,
        *,
        selector_enabled: bool = False,
        selector_mode: SelectorMode = "uniform",
        selector_ratio: float = 1.0,
        selector_seed: int = 42,
    ) -> None:
        super().__init__(embedder, encoder, layer_norm, decoder)
        if not getattr(encoder, "is_causal", True):
            raise ValueError(
                "CLM requires causal encoder; "
                f"{type(encoder).__name__}.is_causal={encoder.is_causal}"
            )
        self.selector_enabled = selector_enabled
        self.selector = TokenSelector(
            mode=selector_mode,
            ratio=selector_ratio,
            seed=selector_seed,
        )

    def prepare_for_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        soft_masked: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Flatten causal next-token predictions for the legacy objective."""

        logits = logits[:, :-1].reshape(-1, logits.size(-1))
        labels = labels[:, 1:].reshape(-1).long()
        soft_masked = soft_masked[:, 1:].reshape(-1)
        return logits, labels, soft_masked

    def _selector_loss(
        self,
        *,
        logits: torch.Tensor,
        labels: torch.Tensor,
        soft_masked: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Apply issue #515 selection after causal alignment and CE."""

        aligned_logits = logits[:, :-1]
        aligned_labels = labels[:, 1:].long()
        aligned_soft_masked = soft_masked[:, 1:].bool()
        valid = aligned_labels != -100
        if attention_mask is not None:
            valid = valid & attention_mask[:, 1:].bool()
        loss_per_token = F.cross_entropy(
            aligned_logits.transpose(1, 2),
            aligned_labels,
            reduction="none",
            ignore_index=-100,
        )
        eligible = valid & ~aligned_soft_masked
        selected = self.selector(loss_per_token, eligible)
        unselected = eligible & ~selected
        thresholds = self._selection_thresholds(loss_per_token, eligible, selected)

        return {
            "loss": _masked_mean(loss_per_token, selected),
            "loss_full": _masked_mean(loss_per_token, valid),
            "loss_non_soft_masked": _masked_mean(loss_per_token, eligible),
            "loss_selected": _masked_mean(loss_per_token, selected),
            "loss_unselected": _masked_mean(loss_per_token, unselected),
            "loss_per_token": loss_per_token,
            "aligned_labels": aligned_labels,
            "eligible_mask": eligible,
            "selected_mask": selected,
            "unselected_mask": unselected,
            "selection_thresholds": thresholds,
            "eligible_count": eligible.sum(),
            "selected_count": selected.sum(),
            "unselected_count": unselected.sum(),
        }

    def _selection_thresholds(
        self,
        losses: torch.Tensor,
        eligible: torch.Tensor,
        selected: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-sequence lower/upper selected-loss bounds."""

        bounds = torch.full(
            (losses.shape[0], 2),
            torch.nan,
            device=losses.device,
            dtype=losses.dtype,
        )
        if self.selector.mode in {"uniform", "random"}:
            return bounds
        detached = losses.detach()
        selected_losses = detached.masked_fill(~(selected & eligible), float("inf"))
        bounds[:, 0] = selected_losses.amin(dim=1)
        bounds[:, 1] = detached.masked_fill(~(selected & eligible), float("-inf")).amax(
            dim=1
        )
        bounds.masked_fill_(~selected.any(dim=1, keepdim=True), torch.nan)
        return bounds

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        soft_masked: torch.Tensor,
        soft_masked_weight: float,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the legacy or selector-aware CLM objective."""

        logits = self.get_logits(input_ids, attention_mask=attention_mask)
        if not self.selector_enabled:
            logits, labels, soft_masked = self.prepare_for_loss(
                logits,
                labels,
                soft_masked,
            )
            return self.compute_loss(
                logits,
                labels,
                soft_masked,
                soft_masked_weight,
            )
        if soft_masked_weight != 0.0:
            raise ValueError(
                "selector-enabled CLM requires soft_masked_weight=0.0 so repeat "
                "positions are excluded from both ranking and training"
            )
        return self._selector_loss(
            logits=logits,
            labels=labels,
            soft_masked=soft_masked,
            attention_mask=attention_mask,
        )


class HFCLM(CLM):
    """CLM interface backed by a Hugging Face causal-language-model checkpoint."""

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        *,
        revision: str | None = None,
        torch_dtype: str | torch.dtype | None = None,
        attention_implementation: str = "sdpa",
        selector_enabled: bool = True,
        selector_mode: SelectorMode = "uniform",
        selector_ratio: float = 1.0,
        selector_seed: int = 42,
    ) -> None:
        nn.Module.__init__(self)
        from transformers import AutoModelForCausalLM

        dtype = self._resolve_dtype(torch_dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            revision=revision,
            torch_dtype=dtype,
            attn_implementation=attention_implementation,
        )
        self.model.config.use_cache = False
        self.selector_enabled = selector_enabled
        self.selector = TokenSelector(
            mode=selector_mode,
            ratio=selector_ratio,
            seed=selector_seed,
        )

    @staticmethod
    def _resolve_dtype(value: str | torch.dtype | None) -> torch.dtype | None:
        if value is None or isinstance(value, torch.dtype):
            return value
        mapping: dict[str, torch.dtype] = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        try:
            return mapping[value]
        except KeyError as error:
            raise ValueError(f"unsupported torch_dtype {value!r}") from error

    def get_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return raw logits without invoking a second model forward."""

        output: Any = self.model(
            input_ids=input_ids.long(),
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return output.logits
