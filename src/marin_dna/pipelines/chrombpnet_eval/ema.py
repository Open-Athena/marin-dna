"""Exponential moving average of model weights (#259 pre-registered knob).

A standard EMA of the trainable parameters, updated every optimizer step. The QTL
eval scores the **EMA** weights (via :meth:`EMA.average_parameters`, swapped in
around scoring and restored after), so the logged metric reflects what an
EMA checkpoint would deploy — without disturbing the live training weights.

EMA operates on the ``LightningModule`` (parameter names like ``model.iconv.weight``);
those tensors are shared with ``pl_module.model``, so swapping them is visible to
``score_log2fc(pl_module.model, ...)``. EMA covers *parameters* only, not buffers —
fine here because the winning two-head tower has no BatchNorm running stats (and
the per-base variant uses GroupNorm, which also has none).
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import lightning as L
import torch


class EMA:
    """Exponential moving average of a module's trainable parameters."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        assert 0.0 < decay < 1.0, decay
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @contextlib.contextmanager
    def average_parameters(self, model: torch.nn.Module) -> Iterator[None]:
        """Swap the EMA weights into ``model`` for the duration of the block,
        restoring the live weights on exit."""
        backup = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if n in self.shadow
        }
        try:
            for n, p in model.named_parameters():
                if n in self.shadow:
                    p.data.copy_(self.shadow[n])
            yield
        finally:
            for n, p in model.named_parameters():
                if n in backup:
                    p.data.copy_(backup[n])


class EMACallback(L.Callback):
    """Maintain an :class:`EMA` of the LightningModule's weights — update each
    optimizer step. Add this **before** the QTLEvalCallback so the EMA is current
    when the eval reads it (callbacks fire in list order)."""

    def __init__(self, decay: float = 0.999) -> None:
        super().__init__()
        self.decay = decay
        self.ema: EMA | None = None

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self.ema is None:
            self.ema = EMA(pl_module, self.decay)

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.ema is not None:
            self.ema.update(pl_module)
