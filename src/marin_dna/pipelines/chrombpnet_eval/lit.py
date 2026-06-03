"""Lightning training module for the one-hot ChromBPNet baseline (issue #241).

Replicates the vendored ChromBPNet training step — a weighted sum of the
base-resolution **profile** multinomial NLL and the **counts** MSE — as our own
``LightningModule`` so we own the optimizer and the logger (W&B) and can add the
instrumentation we want while validating the pipeline:

- **counts Pearson + Spearman** on the validation set, computed once per
  validation epoch over the pooled predictions (scipy, matching the vendored
  ``counts_metrics``). ``val_count_pearson`` is the early-stopping / checkpoint
  monitor (the ChromBPNet convention); ``val_count_spearman`` is logged
  alongside it. Official ChromBPNet logs only Pearson in its train loop — the
  Spearman is the extra signal we surface here.
- the **gradient L2 norm** per optimizer step (a basic exploding/vanishing-grad
  health signal), which official ChromBPNet does not log.

The profile loss reuses the vendored ``multinomial_nll`` so the objective is
bit-identical to ChromBPNet's. ``model`` only has to satisfy the ChromBPNet
forward contract (``onehot[B,4,L] -> (profile[B,out], counts[B,1])``), so the
gLM-embedding arm (#243) can reuse this module unchanged; its separate LR group
for the LM is layered on there, not here.

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
from scipy.stats import pearsonr, spearmanr

from marin_dna.pipelines.chrombpnet_eval._vendor.chrombpnet.model_wrappers import (
    multinomial_nll,
)


class ChromBPNetLit(L.LightningModule):
    """Train a ChromBPNet-style model with the counts+profile loss.

    Args:
        model: a module obeying the ChromBPNet forward contract — ``forward(
            onehot[B,4,L]) -> (profile_logits[B,out], log_counts[B,1])``. For
            #241 this is the vendored one-hot ``ChromBPNet`` (built by
            :func:`marin_dna.pipelines.chrombpnet_eval.onehot.build_onehot_chrombpnet`).
        alpha: counts-loss weight (the driver sets ``median_count/10``).
        beta: profile-loss weight (1.0).
        lr: Adam learning rate. Default ``1e-3`` matches official one-hot
            ChromBPNet (the embedding arm drops to ``1e-4``).
        warmup_steps: linear LR warmup over this many optimizer steps (0 = off).
            Ramps from ``0.01*lr`` to ``lr`` then holds constant — tames the
            early large-gradient step that can diverge to NaN (#247). Takes
            precedence over ``lr_scheduler``.
        lr_scheduler: ``None`` (constant LR, official one-hot default) or
            ``"plateau"`` (``ReduceLROnPlateau`` on ``val_count_pearson`` —
            ChromBPNet's commented-out config). Ignored when ``warmup_steps>0``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        alpha: float = 1.0,
        beta: float = 1.0,
        lr: float = 1e-3,
        warmup_steps: int = 0,
        lr_scheduler: str | None = None,
    ) -> None:
        super().__init__()
        assert lr_scheduler in (None, "plateau"), (
            f"lr_scheduler must be None or 'plateau', got {lr_scheduler!r}"
        )
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.lr_scheduler = lr_scheduler
        self._val_pred: list[torch.Tensor] = []
        self._val_true: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)

    def _step(self, batch: dict, mode: str) -> torch.Tensor:
        true_profile = batch["profile"]
        true_counts = torch.log1p(true_profile.sum(dim=-1))
        y_profile, y_count = self(batch["onehot_seq"])
        y_count = y_count.squeeze(-1)
        # Compute the loss in fp32 even under a bf16 autocast: the multinomial
        # NLL and the exp/log count-combine are bf16-unstable (→ NaN). Disabling
        # autocast here keeps a ``--precision bf16-mixed`` run (this arm is
        # GPU-compute-bound, so bf16 can speed it up) from NaN-ing on the loss.
        with torch.autocast(device_type=y_count.device.type, enabled=False):
            profile_loss = multinomial_nll(y_profile.float(), true_profile.float())
            count_loss = F.mse_loss(y_count.float(), true_counts.float())
            loss = self.beta * profile_loss + self.alpha * count_loss
        self.log_dict(
            {
                f"{mode}_loss": loss,
                f"{mode}_profile_loss": profile_loss,
                f"{mode}_count_loss": count_loss,
            },
            on_step=(mode == "train"),  # live per-step train curve; val per-epoch
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
        if not self._val_pred:
            return
        # Pool over the whole val set, then correlate predicted vs observed log
        # counts. Single-device exact; with >1 device `sync_dist` averages the
        # per-rank scalars (we train on 1 GPU, so this is exact for our runs).
        pred = torch.cat(self._val_pred).float().cpu().numpy().ravel()
        true = torch.cat(self._val_true).float().cpu().numpy().ravel()
        if pred.size > 1 and pred.std() > 0 and true.std() > 0:
            pearson = float(pearsonr(pred, true).statistic)
            spearman = float(spearmanr(pred, true).statistic)
        else:
            # Too few / constant predictions (e.g. an early smoke val pass) —
            # correlation is undefined; log 0 so early-stopping has a number.
            pearson = spearman = 0.0
        self.log("val_count_pearson", pearson, prog_bar=True, sync_dist=True)
        self.log("val_count_spearman", spearman, prog_bar=True, sync_dist=True)
        self._val_pred.clear()
        self._val_true.clear()

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        # Total L2 norm of the gradients (post-unscale, pre-clip) over trainable
        # params — frozen-bias params have grad None and are skipped.
        norms = [
            p.grad.detach().norm(2) for p in self.parameters() if p.grad is not None
        ]
        total = (
            torch.norm(torch.stack(norms), 2)
            if norms
            else torch.zeros((), device=self.device)
        )
        self.log("grad_norm", total, on_step=True, on_epoch=False, prog_bar=True)

    def configure_optimizers(self) -> Any:
        opt = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad], lr=self.lr, eps=1e-7
        )
        # Warmup takes precedence: a step-wise linear ramp from 0.01*lr to lr over
        # warmup_steps, then constant (LinearLR holds at end_factor) — keeps the
        # first (huge-gradient) steps tiny so they can't diverge.
        if self.warmup_steps > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=0.01, end_factor=1.0, total_iters=self.warmup_steps
            )
            return {
                "optimizer": opt,
                "lr_scheduler": {"scheduler": warmup, "interval": "step"},
            }
        if self.lr_scheduler == "plateau":
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="max", factor=0.4, patience=3, min_lr=1e-8
            )
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": sched,
                    "monitor": "val_count_pearson",
                    "interval": "epoch",
                },
            }
        return opt
