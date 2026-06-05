"""Lightning training module for the one-hot ChromBPNet baseline (#241, #259).

Replicates the vendored ChromBPNet training step — a weighted sum of the
base-resolution **profile** multinomial NLL and the **counts** MSE — as our own
``LightningModule`` so we own the optimizer and the logger (W&B).

There is **no accessibility validation loop** (#259): the all-chromosome
fixed-budget protocol trains on every chromosome, so there is no held-out
accessibility split to validate on, and we do not select on an accessibility
metric. The **eval target** — the caQTL/dsQTL variant-effect Pearson — is logged
on a step cadence by :class:`~marin_dna.pipelines.chrombpnet_eval.qtl_eval.QTLEvalCallback`,
not here. In-training health signals are the **train losses** and the per-step
**gradient L2 norm** (an exploding/vanishing-grad check official ChromBPNet does
not log).

The profile loss reuses the vendored ``multinomial_nll`` so the objective is
bit-identical to ChromBPNet's. ``model`` only has to satisfy the ChromBPNet
forward contract (``onehot[B,4,L] -> (profile[B,out], counts[B,1])``), so the
gLM-embedding arm (#243) can reuse this module; its separate LR group for the LM
is layered on there, not here.

Batch contract (matches the vendored ``DataModule``): ``onehot_seq`` ``[B,4,L]``
and ``profile`` ``[B,out]`` (observed per-bp counts over the central window).
``alpha`` (counts weight) is set to ``median_count/10`` by the driver (the
ChromBPNet scale-balancing heuristic); ``beta`` (profile weight) is 1.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import lightning as L
import torch
import torch.nn.functional as F

from marin_dna.pipelines.chrombpnet_eval._vendor.chrombpnet.model_wrappers import (
    multinomial_nll,
)


def wsd_lr_lambda(
    total_steps: int, warmup_frac: float, decay_frac: float
) -> Callable[[int], float]:
    """Warmup-Stable-Decay LR multiplier as a function of optimizer step (#259).

    Returns a callable for ``torch.optim.lr_scheduler.LambdaLR`` (which multiplies
    the base LR by it): a linear **warmup** over the first ``warmup_frac`` of
    ``total_steps`` (ramping ~0 -> 1), a constant **stable** phase at 1.0, then a
    linear **decay** 1 -> 0 over the final ``decay_frac``. Written as a pure
    function of the step so the schedule shape is unit-testable without a Trainer.
    The stable phase is what lets us fix the budget late — decay from any
    stable-phase checkpoint — rather than committing N up front.
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
            # linear ramp ~0 -> 1; clamp guards sub-step warmup (warmup < 1)
            return min(1.0, (step + 1) / warmup)
        if step < decay_start:
            return 1.0  # stable
        if decay > 0:
            return max(0.0, (total - step) / decay)  # linear 1 -> 0
        return 1.0

    return factor


class ChromBPNetLit(L.LightningModule):
    """Train a ChromBPNet-style model with the counts+profile loss (no val loop).

    Args:
        model: a module obeying the ChromBPNet forward contract — ``forward(
            onehot[B,4,L]) -> (profile_logits[B,out], log_counts[B,1])``. For the
            one-hot arm this is the vendored ``ChromBPNet`` (built by
            :func:`marin_dna.pipelines.chrombpnet_eval.onehot.build_onehot_chrombpnet`).
        alpha: counts-loss weight. The driver calibrates it per ``count_loss`` form
            so the count-head gradient scale matches across forms (``mse_log``:
            ``median_count/10``; ``poisson``: ``0.1``) — see ``count_loss``.
        beta: profile-loss weight (1.0).
        count_loss: count-head loss form (#259 — the "key" loss-function axis).
            ``"mse_log"`` (default, official ChromBPNet) = MSE between the predicted
            log-count and ``log1p(total)``. ``"poisson"`` = Poisson NLL of the raw
            total under rate ``exp(y_count)`` (``log_input=True``). The count head
            stays a scalar log-rate either way, so the QTL log2FC score is
            unchanged. Poisson's gradient wrt ``y_count`` is the count-space
            residual ``exp(y)-total`` (scale ``~median_count``) vs mse-log's
            ``~O(1)``; the driver's ``0.1`` poisson weight (= ``(median/10)/median``)
            matches the two at the median so this isolates loss *form* from *weight*.
        lr: learning rate. Default ``1e-3`` matches official one-hot ChromBPNet
            (the embedding arm drops to ``1e-4``).
        optimizer: ``"adam"`` (default, official ChromBPNet) or ``"adamw"``
            (decoupled weight decay, #259 knob).
        weight_decay: L2 (``adam``) / decoupled (``adamw``) weight decay; default 0.
        warmup_steps: linear LR warmup over this many optimizer steps (0 = off);
            tames the early large-gradient step that can diverge to NaN (#247).
            Used only when ``lr_scheduler`` is ``None`` (constant LR); ``"wsd"``
            owns its own warmup via ``warmup_frac``.
        lr_scheduler: ``None`` (constant LR, optionally with ``warmup_steps``) or
            ``"wsd"`` (#259 — Warmup-Stable-Decay over the fixed step budget,
            using ``warmup_frac``/``decay_frac``).
        warmup_frac / decay_frac: WSD warmup and decay fractions of the total step
            budget (only used when ``lr_scheduler="wsd"``). ``warmup_frac`` is a
            small ``0.01`` — these are short supervised runs, and the early NaN
            spike (#247) is tamed by ~100 warmup steps + grad-clip, not a long
            LLM-pretraining-style 10% warmup.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        alpha: float = 1.0,
        beta: float = 1.0,
        count_loss: str = "mse_log",
        lr: float = 1e-3,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
        lr_scheduler: str | None = None,
        warmup_frac: float = 0.01,
        decay_frac: float = 0.2,
    ) -> None:
        super().__init__()
        assert lr_scheduler in (None, "wsd"), (
            f"lr_scheduler must be None or 'wsd', got {lr_scheduler!r}"
        )
        assert optimizer in ("adam", "adamw"), (
            f"optimizer must be 'adam' or 'adamw', got {optimizer!r}"
        )
        assert count_loss in ("mse_log", "poisson"), (
            f"count_loss must be 'mse_log' or 'poisson', got {count_loss!r}"
        )
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.count_loss = count_loss
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
        # Compute the loss in fp32 even under a bf16 autocast: the multinomial
        # NLL and the exp/log count-combine are bf16-unstable (→ NaN). Disabling
        # autocast here keeps a ``--precision bf16-mixed`` run (this arm is
        # GPU-compute-bound, so bf16 can speed it up) from NaN-ing on the loss.
        with torch.autocast(device_type=y_count.device.type, enabled=False):
            # Count-head loss form (#259). Both treat y_count as a log-rate, so the
            # QTL log2FC score (qtl_eval.score_log2fc) is unchanged; the driver
            # picks alpha per form to match the count-head gradient scale.
            if self.count_loss == "poisson":
                # Poisson NLL: exp(y_count) - total*y_count (full=False drops the
                # log(total!) Stirling term, constant in y_count).
                count_loss = F.poisson_nll_loss(
                    y_count.float(),
                    true_total.float(),
                    log_input=True,
                    full=False,
                    reduction="mean",
                )
            else:  # mse_log (official ChromBPNet): MSE on log1p(total)
                count_loss = F.mse_loss(
                    y_count.float(), torch.log1p(true_total).float()
                )
            # beta=0 -> count-only (#259): skip the profile NLL entirely. The QTL
            # score uses only the count head, so this tests whether the profile
            # objective is needed at all. Otherwise the vendored profile+count loss.
            if self.beta:
                profile_loss = multinomial_nll(y_profile.float(), true_profile.float())
                loss = self.beta * profile_loss + self.alpha * count_loss
            else:
                profile_loss = count_loss.new_zeros(())
                loss = self.alpha * count_loss
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
        opt_cls = torch.optim.AdamW if self.optimizer == "adamw" else torch.optim.Adam
        opt = opt_cls(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.lr,
            eps=1e-7,
            weight_decay=self.weight_decay,
        )
        # WSD (#259): warmup -> stable -> linear decay to 0 over the fixed step
        # budget. Needs a known horizon, so it composes with a fixed budget.
        if self.lr_scheduler == "wsd":
            total_f = self.trainer.estimated_stepping_batches
            assert math.isfinite(total_f) and total_f > 0, (
                f"WSD needs a finite positive step budget (set --max-steps or "
                f"--max-epochs); got estimated_stepping_batches={total_f}"
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
        # Constant LR with an optional step-wise linear warmup (0.01*lr -> lr over
        # warmup_steps, then held) — keeps the first (huge-gradient) steps tiny.
        if self.warmup_steps > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=0.01, end_factor=1.0, total_iters=self.warmup_steps
            )
            return {
                "optimizer": opt,
                "lr_scheduler": {"scheduler": warmup, "interval": "step"},
            }
        return opt
