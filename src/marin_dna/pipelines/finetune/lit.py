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
        schedule: str = "cosine",  # "cosine" | "wsd" (warmup-stable-decay)
        pos_weight: float | None = None,
        compile_model: bool = False,
    ) -> None:
        super().__init__()
        self.clf = torch.compile(clf) if compile_model else clf
        # Un-compiled handle for param groups — registered via object.__setattr__ so it is
        # NOT a second registered submodule (else params/state_dict duplicate, and under
        # compile the `_orig_mod` vs raw key names collide on the best-state load).
        object.__setattr__(self, "_raw_clf", clf)
        self.label = np.asarray(label)
        self.chrom = np.asarray(chrom, dtype=str)
        self.lora_lr = lora_lr
        self.head_lr = head_lr
        self.weight_decay = weight_decay
        self.head_weight_decay = head_weight_decay
        self.warmup_frac = warmup_frac
        self.schedule = schedule
        self.register_buffer(
            "pos_weight",
            torch.tensor(pos_weight) if pos_weight else None,
            persistent=False,
        )
        self._val_scores: list[np.ndarray] = []
        self._val_rows: list[np.ndarray] = []
        # best-val trainable-state snapshot + per-epoch trajectory, tracked in-module so
        # they use the FRESH val_auprc + the weights that produced it (callbacks run
        # before the module logs, giving a one-epoch lag — issue #369).
        self._history: list[dict] = []
        self._best_val: float = -float("inf")
        self._best_epoch: int = 0
        self._best_state: dict[str, torch.Tensor] | None = None

    def on_train_start(self):
        t = self.trainer
        nb = t.num_training_batches  # micro-batches / epoch
        acc = t.accumulate_grad_batches
        print(
            f"[steps] micro-batches/epoch={nb} accum={acc} "
            f"optimizer-steps/epoch={-(-nb // acc)} "
            f"total-optimizer-steps(estimated_stepping_batches)={t.estimated_stepping_batches} "
            f"max_epochs={t.max_epochs}",
            flush=True,
        )

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
        ap = float(per_chrom_weighted_ap(self.label[rows], scores, self.chrom[rows]))
        self.log("val_auprc", ap, prog_bar=True)
        self._val_scores.clear()
        self._val_rows.clear()
        m = self.trainer.callback_metrics
        self._history.append({
            "epoch": int(self.current_epoch),
            "train_loss": float(m["train_loss"]) if "train_loss" in m else float("nan"),
            "val_loss": float(m["val_loss"]) if "val_loss" in m else float("nan"),
            "val_auprc": ap,
        })
        # Snapshot the trainable (LoRA+head) params that just produced this val AUPRC.
        if np.isfinite(ap) and ap > self._best_val:
            self._best_val = ap
            self._best_epoch = int(self.current_epoch)
            self._best_state = {
                n: p.detach().cpu().clone()
                for n, p in self.named_parameters()
                if p.requires_grad
            }

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
        total = max(int(self.trainer.estimated_stepping_batches), 1)
        warmup = int(self.warmup_frac * total)
        if self.schedule == "wsd":  # warmup -> stable -> decay (last 20%)
            from transformers import get_wsd_schedule

            decay = max(1, int(0.2 * total))
            sched = get_wsd_schedule(opt, warmup, max(0, total - warmup - decay), decay)
        else:
            sched = get_cosine_schedule_with_warmup(opt, warmup, total)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
