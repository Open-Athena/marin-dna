"""Lightning training module for ChromBPNet (one-hot or gLM-embedding input).

Replicates the vendored ChromBPNet training step — a weighted sum of the
base-resolution **profile** multinomial NLL and the **counts** MSE — but as our
own ``LightningModule`` so we control the optimizer (a separate, smaller LR group
for the LM when fine-tuning) and the logger (W&B). The profile loss reuses the
vendored ``multinomial_nll`` so the objective is bit-identical to ChromBPNet.

Batch contract (matches the vendored ``DataModule``): ``onehot_seq`` ``[B,4,L]``
and ``profile`` ``[B,out]`` (observed per-bp counts over the central window).
``alpha`` (counts weight) is set to ``median_count/10`` by the driver (the
ChromBPNet scale-balancing heuristic); ``beta`` (profile weight) is 1.
"""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.nn.functional as F

from marin_dna.pipelines.chrombpnet_eval._vendor.chrombpnet.model_wrappers import (
    multinomial_nll,
)


def _pearson(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm() + 1e-8)


class ChromBPNetLit(L.LightningModule):
    """Train a ChromBPNet-style model with the counts+profile loss.

    ``model`` is either the vendored one-hot ``ChromBPNet`` or our
    :class:`GLMChromBPNet`. With ``finetune=True`` and a ``GLMChromBPNet``, the LM
    (``model.embedder.model``) gets its own ``lr_lm`` param group; otherwise a
    single ``lr_head`` group covers the trainable (head) parameters.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        alpha: float = 1.0,
        beta: float = 1.0,
        lr_head: float = 1e-3,
        lr_lm: float = 1e-5,
        finetune: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.lr_head = lr_head
        self.lr_lm = lr_lm
        self.finetune = finetune
        self._val_pred: list[torch.Tensor] = []
        self._val_true: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)

    def _step(self, batch: dict, mode: str) -> torch.Tensor:
        true_profile = batch["profile"]
        true_counts = torch.log1p(true_profile.sum(dim=-1))
        y_profile, y_count = self(batch["onehot_seq"])
        y_count = y_count.squeeze(-1)
        profile_loss = multinomial_nll(y_profile, true_profile)
        count_loss = F.mse_loss(y_count, true_counts)
        loss = self.beta * profile_loss + self.alpha * count_loss
        self.log_dict(
            {
                f"{mode}_loss": loss,
                f"{mode}_profile_loss": profile_loss,
                f"{mode}_count_loss": count_loss,
            },
            on_step=(
                mode == "train"
            ),  # live per-step train curve; val aggregates per-epoch
            on_epoch=True,
            prog_bar=(mode == "train"),
            sync_dist=True,
        )
        if mode == "val":
            self._val_pred.append(y_count.detach())
            self._val_true.append(true_counts.detach())
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def on_validation_epoch_end(self) -> None:
        if self._val_pred:
            pred = torch.cat(self._val_pred)
            true = torch.cat(self._val_true)
            vp = (
                _pearson(pred, true)
                if pred.numel() > 1
                else torch.zeros((), device=pred.device)
            )
            # val_count_pearson is the early-stopping / checkpoint metric (ChromBPNet convention).
            self.log("val_count_pearson", vp, prog_bar=True, sync_dist=True)
        self._val_pred.clear()
        self._val_true.clear()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        if self.finetune and hasattr(self.model, "embedder"):
            model_any: Any = self.model
            lm = model_any.embedder.model  # the HF gLM (see GLMChromBPNet)
            lm_params = [p for p in lm.parameters() if p.requires_grad]
            lm_ids = {id(p) for p in lm.parameters()}
            head_params = [
                p
                for p in self.model.parameters()
                if p.requires_grad and id(p) not in lm_ids
            ]
            groups = [
                {"params": head_params, "lr": self.lr_head},
                {"params": lm_params, "lr": self.lr_lm},
            ]
        else:
            groups = [
                {
                    "params": [p for p in self.model.parameters() if p.requires_grad],
                    "lr": self.lr_head,
                }
            ]
        return torch.optim.Adam(groups, eps=1e-7)
