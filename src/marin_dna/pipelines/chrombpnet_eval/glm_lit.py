"""WSD/AdamW Lightning module + step-cadence QTL callback for the #243 frozen-gLM
ChromBPNet arm.

**Duplicated (grug) from the #259 branch** (`lit.py` + `qtl_eval.py`): this is the
experimental M2 embedding arm and may never merge, so it does **not** share the
merged one-hot trainer. The recipe is the #259 recommended one — AdamW(wd) + WSD
over a fixed step budget, ``mse_log`` count loss + multinomial-NLL profile, **no
validation loop / no early-stop / no EMA**. The eval target (caQTL/dsQTL Pearson)
is logged on a global-step cadence so we can watch the **whole trajectory**, not
just the final value.

The QTL scoring core (``QTLSpec`` / ``score_log2fc`` / ``build_qtl_specs``) is
reused from the merged ``qtl_eval.py`` — it calls ``model(onehot)`` and works
unchanged with the embedding model.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, cast

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr

from marin_dna.pipelines.chrombpnet_eval._vendor.chrombpnet.model_wrappers import (
    multinomial_nll,
)
from marin_dna.pipelines.chrombpnet_eval.qtl_eval import QTLSpec, score_log2fc


def wsd_lr_lambda(
    total_steps: int, warmup_frac: float, decay_frac: float
) -> Callable[[int], float]:
    """Warmup-Stable-Decay LR multiplier as a function of optimizer step (#259).

    A linear **warmup** over the first ``warmup_frac`` of ``total_steps`` (~0 → 1),
    a constant **stable** phase at 1.0, then a linear **decay** 1 → 0 over the final
    ``decay_frac``. Pure function of the step (unit-testable without a Trainer); for
    ``torch.optim.lr_scheduler.LambdaLR``, which multiplies the base LR by it.
    """
    assert total_steps > 0, total_steps
    assert 0.0 <= warmup_frac < 1.0 and 0.0 <= decay_frac < 1.0, (
        warmup_frac,
        decay_frac,
    )
    assert warmup_frac + decay_frac <= 1.0, (warmup_frac, decay_frac)
    total = float(total_steps)
    warmup = warmup_frac * total
    decay = decay_frac * total
    decay_start = total - decay

    def factor(step: int) -> float:
        if warmup > 0 and step < warmup:
            return min(1.0, (step + 1) / warmup)  # linear ramp ~0 → 1
        if step < decay_start:
            return 1.0  # stable
        if decay > 0:
            return max(0.0, (total - step) / decay)  # linear 1 → 0
        return 1.0

    return factor


class GLMChromBPNetLit(L.LightningModule):
    """Train the frozen-gLM ChromBPNet head with the #259 count+profile loss.

    No validation loop (#259, all-chromosome fixed-budget protocol): the eval
    target is the caQTL/dsQTL Pearson logged by :class:`GLMQTLStepCallback`; the
    in-training health signals are the train losses and the per-step gradient norm.

    Args:
        model: a module obeying the ChromBPNet contract — ``forward(onehot[B,4,L])
            -> (profile_logits[B,out], log_counts[B,1])`` (here
            :class:`~marin_dna.pipelines.chrombpnet_eval.glm_embed.GLMSamePadChromBPNet`).
            Only its trainable params (the LayerNorm + conv tower; the gLM is
            frozen) are optimized.
        alpha: counts-loss weight (the driver sets ``median_count/10``).
        beta: profile-loss weight (1.0).
        lr: learning rate (head-only; the frozen arm can keep ChromBPNet's 1e-3).
        optimizer: ``"adam"`` or ``"adamw"`` (decoupled weight decay; #259 default
            for this arm with ``weight_decay=0.01``).
        weight_decay: AdamW weight decay.
        warmup_steps: constant-LR linear warmup steps (only when ``lr_scheduler``
            is ``None``).
        lr_scheduler: ``None`` (constant LR) or ``"wsd"`` (Warmup-Stable-Decay over
            the fixed step budget, using ``warmup_frac``/``decay_frac``).
        warmup_frac / decay_frac: WSD warmup / decay fractions of the budget.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        alpha: float = 1.0,
        beta: float = 1.0,
        lr: float = 1e-3,
        optimizer: str = "adamw",
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        lr_scheduler: str | None = "wsd",
        warmup_frac: float = 0.01,
        decay_frac: float = 0.2,
    ) -> None:
        super().__init__()
        assert lr_scheduler in (None, "wsd"), lr_scheduler
        assert optimizer in ("adam", "adamw"), optimizer
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.lr_scheduler = lr_scheduler
        self.warmup_frac = warmup_frac
        self.decay_frac = decay_frac

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        true_profile = batch["profile"]
        true_total = true_profile.sum(dim=-1)  # raw total reads over the window
        y_profile, y_count = self(batch["onehot_seq"])
        y_count = y_count.squeeze(-1)  # predicted log-count (log-rate)
        # Compute the loss in fp32 even under a bf16 autocast: the multinomial NLL
        # and log1p are bf16-unstable (→ NaN). The conv tower still runs bf16 under
        # the outer autocast, so --precision bf16-mixed speeds it up without NaN-ing.
        with torch.autocast(device_type=y_count.device.type, enabled=False):
            count_loss = F.mse_loss(y_count.float(), torch.log1p(true_total).float())
            profile_loss = multinomial_nll(y_profile.float(), true_profile.float())
            loss = self.beta * profile_loss + self.alpha * count_loss
        self.log_dict(
            {
                "train_loss": loss,
                "train_profile_loss": profile_loss,
                "train_count_loss": count_loss,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        # Total L2 norm of the gradients (post-unscale, pre-clip) over trainable
        # params — frozen gLM params have grad None and are skipped.
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
        opt_cls = torch.optim.AdamW if self.optimizer == "adamw" else torch.optim.Adam
        opt = opt_cls(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.lr,
            eps=1e-7,
            weight_decay=self.weight_decay,
        )
        if self.lr_scheduler == "wsd":
            total_f = self.trainer.estimated_stepping_batches
            assert math.isfinite(total_f) and total_f > 0, (
                f"WSD needs a finite positive step budget (set --max-steps); got "
                f"estimated_stepping_batches={total_f}"
            )
            sched = torch.optim.lr_scheduler.LambdaLR(
                opt,
                lr_lambda=wsd_lr_lambda(
                    int(total_f), self.warmup_frac, self.decay_frac
                ),
            )
            return {
                "optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step"},
            }
        if self.warmup_steps > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=0.01, end_factor=1.0, total_iters=self.warmup_steps
            )
            return {
                "optimizer": opt,
                "lr_scheduler": {"scheduler": warmup, "interval": "step"},
            }
        return opt


def signed_pearson(scores: np.ndarray, effect: np.ndarray) -> float:
    """Signed Pearson of ``scores`` vs ``effect``; ``0.0`` if degenerate (constant
    input, < 2 points, or a non-finite result from a diverged model). Never NaN."""
    if len(scores) > 1 and np.std(scores) > 0 and np.std(effect) > 0:
        pearson = float(pearsonr(scores, effect).statistic)
        if np.isfinite(pearson):
            return pearson
    return 0.0


class GLMQTLStepCallback(L.Callback):
    """Log the eval target — signed Pearson of predicted log2FC vs observed
    ``effect`` over the QTL positives, per dataset plus their mean
    ``qtl_avg_pearson`` — on a **training-step cadence** (#259).

    There is no validation loop, so this fires from ``on_train_batch_end`` every
    ``every_n_steps`` optimizer steps and again on the final step (capturing the
    post-WSD-decay endpoint). Metrics go straight to ``trainer.logger`` at the
    current global step — exactly once per eval — rather than via ``pl_module.log``,
    which from a per-batch hook re-emits the held value at every flush and spams the
    curve. Single-device only (we train on 1 GPU).
    """

    def __init__(
        self,
        specs: Sequence[QTLSpec],
        *,
        batch_size: int = 256,
        every_n_steps: int = 500,
    ) -> None:
        super().__init__()
        assert every_n_steps > 0, every_n_steps
        self.specs = list(specs)
        self.batch_size = batch_size
        self.every_n_steps = every_n_steps

    def _log_qtl(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        metrics: dict[str, float] = {}
        pearsons: list[float] = []
        for spec in self.specs:
            scores = score_log2fc(
                cast(torch.nn.Module, pl_module.model),
                spec.ref_oh,
                spec.alt_oh,
                batch_size=self.batch_size,
                device=pl_module.device,
            )
            p = signed_pearson(scores, spec.effect)
            metrics[f"qtl_{spec.name}_pearson"] = p
            pearsons.append(p)
        if not pearsons:
            return
        metrics["qtl_avg_pearson"] = float(np.mean(pearsons))
        if trainer.logger is not None:
            trainer.logger.log_metrics(metrics, step=trainer.global_step)

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        step = trainer.global_step  # post-optimizer-step count
        on_cadence = step > 0 and step % self.every_n_steps == 0
        is_last = trainer.max_steps > 0 and step >= trainer.max_steps
        if on_cadence or is_last:
            self._log_qtl(trainer, pl_module)
