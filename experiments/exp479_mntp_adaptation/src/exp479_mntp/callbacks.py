"""Runtime, budget, and data-contract callbacks for exp479."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lightning as L
import torch

from exp479_mntp.config import BUDGET_USD, LAMBDA_GH200_PRICE_PER_HOUR_USD, SEQUENCE_LENGTH


class RuntimeMetricsCallback(L.Callback):
    """Record throughput, peak CUDA memory, and wall time for one arm."""

    def __init__(self, output_path: Path, batch_size: int) -> None:
        self.output_path = output_path
        self.batch_size = batch_size
        self.started_at: float | None = None
        self.started_at_utc: str | None = None
        self.starting_global_step: int | None = None

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        del pl_module
        self.started_at = time.monotonic()
        self.started_at_utc = datetime.now(UTC).isoformat()
        self.starting_global_step = int(trainer.global_step)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        del pl_module
        if self.started_at is None or self.starting_global_step is None:
            return
        elapsed = time.monotonic() - self.started_at
        global_step = int(trainer.global_step)
        executed_steps = global_step - self.starting_global_step
        payload = {
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": elapsed,
            "starting_global_step": self.starting_global_step,
            "global_step": global_step,
            "executed_steps": executed_steps,
            "batch_size": self.batch_size,
            "model_tokens": executed_steps * self.batch_size * SEQUENCE_LENGTH,
            "model_tokens_per_second": (
                executed_steps * self.batch_size * SEQUENCE_LENGTH / elapsed
                if elapsed > 0
                else None
            ),
            "peak_cuda_allocated_bytes": (
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
            ),
            "peak_cuda_reserved_bytes": (
                int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None
            ),
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class BudgetGuardCallback(L.Callback):
    """Abort before the registered cloud budget can be exhausted."""

    def __init__(
        self,
        *,
        instance_start_unix: float | None,
        budget_usd: float = BUDGET_USD,
        price_per_hour_usd: float = LAMBDA_GH200_PRICE_PER_HOUR_USD,
        reserve_usd: float = 2.0,
        prior_cost_usd: float = 0.0,
    ) -> None:
        self.instance_start_unix = instance_start_unix
        self.budget_usd = budget_usd
        self.price_per_hour_usd = price_per_hour_usd
        self.reserve_usd = reserve_usd
        self.prior_cost_usd = prior_cost_usd

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del pl_module, outputs, batch, batch_idx
        if self.instance_start_unix is None:
            return
        elapsed_hours = (time.time() - self.instance_start_unix) / 3600
        accrued = self.prior_cost_usd + elapsed_hours * self.price_per_hour_usd
        if accrued >= self.budget_usd - self.reserve_usd:
            trainer.should_stop = True
            raise RuntimeError(
                f"exp479 budget guard stopped at projected accrued charge ${accrued:.2f}; "
                f"cap=${self.budget_usd:.2f}, reserve=${self.reserve_usd:.2f}"
            )
