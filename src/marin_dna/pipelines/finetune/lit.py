"""PyTorch-Lightning module for LoRA missense fine-tuning (#369).

Wraps the siamese LoRA classifier. The training step RC-augments (one random strand per
example); validation scores the val chromosome with the full FWD+RC average once per
epoch (the overfitting instrument + early-stop/checkpoint signal); test scoring (full
FWD+RC) is done once, on the best checkpoint, via ``predict``.
"""

from __future__ import annotations

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup

from marin_dna.pipelines.evals.metrics import per_chrom_weighted_ap
from marin_dna.pipelines.finetune.model import SiameseLoRAClassifier


class LitVariantFinetune(L.LightningModule):
    """LoRA fine-tuning LightningModule over the siamese classifier."""

    def __init__(
        self,
        clf: SiameseLoRAClassifier,
        *,
        label: np.ndarray,
        chrom: np.ndarray,
        lora_lr: float = 1e-4,
        head_lr: float = 1e-3,
        weight_decay: float = 0.0,
        head_weight_decay: float = 0.0,
        warmup_frac: float = 0.1,
        pos_weight: float | None = None,
        compile_model: bool = False,
    ) -> None:
        super().__init__()
        self.clf = torch.compile(clf) if compile_model else clf
        self._raw_clf = clf  # un-compiled handle for param groups / num_trainable
        self.label = np.asarray(label)
        self.chrom = np.asarray(chrom, dtype=str)
        self.lora_lr = lora_lr
        self.head_lr = head_lr
        self.weight_decay = weight_decay
        self.head_weight_decay = head_weight_decay
        self.warmup_frac = warmup_frac
        self.register_buffer(
            "pos_weight",
            torch.tensor(pos_weight) if pos_weight else None,
            persistent=False,
        )
        self._val_scores: list[np.ndarray] = []
        self._val_rows: list[np.ndarray] = []

    # ---- training: one random strand per example (RC augmentation) --------------
    def training_step(self, batch, _idx):
        ref_fwd, alt_fwd, ref_rc, alt_rc, y, _j = batch
        use_rc = torch.rand(len(y), device=self.device) < 0.5
        ref = torch.where(use_rc[:, None], ref_rc, ref_fwd)
        alt = torch.where(use_rc[:, None], alt_rc, alt_fwd)
        loss = F.binary_cross_entropy_with_logits(
            self.clf(ref, alt).float(), y.float(), pos_weight=self.pos_weight
        )
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    # ---- validation: full FWD+RC on the val chromosome --------------------------
    def validation_step(self, batch, _idx):
        ref_fwd, alt_fwd, ref_rc, alt_rc, y, j = batch
        logit = self.clf.logit_rc_avg(ref_fwd, alt_fwd, ref_rc, alt_rc).float()
        loss = F.binary_cross_entropy_with_logits(logit, y.float(), pos_weight=self.pos_weight)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self._val_scores.append(logit.detach().cpu().numpy())
        self._val_rows.append(j.detach().cpu().numpy())
        return loss

    def on_validation_epoch_end(self):
        rows = np.concatenate(self._val_rows)
        scores = np.concatenate(self._val_scores)
        ap = per_chrom_weighted_ap(self.label[rows], scores, self.chrom[rows])
        self.log("val_auprc", float(ap), prog_bar=True)
        self._val_scores.clear()
        self._val_rows.clear()

    # ---- test/predict: full FWD+RC on the best checkpoint -----------------------
    def predict_step(self, batch, _idx):
        ref_fwd, alt_fwd, ref_rc, alt_rc, _y, j = batch
        logit = self.clf.logit_rc_avg(ref_fwd, alt_fwd, ref_rc, alt_rc).float()
        return logit.detach().cpu().numpy(), j.detach().cpu().numpy()

    def configure_optimizers(self):
        head_ids = {id(p) for p in self._raw_clf.head.parameters()}
        head = [p for p in self._raw_clf.head.parameters() if p.requires_grad]
        lora = [
            p for p in self._raw_clf.parameters()
            if p.requires_grad and id(p) not in head_ids
        ]
        opt = torch.optim.AdamW(
            [
                {"params": lora, "lr": self.lora_lr, "weight_decay": self.weight_decay},
                {"params": head, "lr": self.head_lr, "weight_decay": self.head_weight_decay},
            ]
        )
        total = int(self.trainer.estimated_stepping_batches)
        sched = get_cosine_schedule_with_warmup(
            opt, int(self.warmup_frac * total), max(total, 1)
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
