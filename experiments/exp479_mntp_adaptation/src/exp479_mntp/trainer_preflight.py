"""One-step memory gate through the exact Lightning training closure."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lightning as L
import torch

from exp479_mntp.config import NUCLEOTIDE_LENGTH
from exp479_mntp.datamodule import ExperimentDataModule
from exp479_mntp.modeling import load_model_bundle
from exp479_mntp.module import AdaptationModule
from exp479_mntp.preflight import enable_training_determinism


class FirstStepMemoryCallback(L.Callback):
    """Measure the complete first optimizer step, including Lightning hooks."""

    def __init__(self) -> None:
        self.started: float | None = None
        self.measurement: dict[str, float | int] | None = None
        self.allocated_before_step = 0
        self.reserved_before_step = 0

    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del trainer, pl_module, batch, batch_idx
        torch.cuda.reset_peak_memory_stats()
        self.allocated_before_step = int(torch.cuda.memory_allocated())
        self.reserved_before_step = int(torch.cuda.memory_reserved())
        self.started = time.perf_counter()

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del trainer, pl_module, outputs, batch, batch_idx
        if self.started is None:
            raise RuntimeError("first-step memory timer did not start")
        torch.cuda.synchronize()
        total = int(torch.cuda.get_device_properties(0).total_memory)
        peak = int(torch.cuda.max_memory_allocated())
        self.measurement = {
            "allocated_before_step_bytes": self.allocated_before_step,
            "reserved_before_step_bytes": self.reserved_before_step,
            "peak_allocated_bytes": peak,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "total_memory_bytes": total,
            "headroom_fraction": 1.0 - peak / total,
            "seconds_per_step": time.perf_counter() - self.started,
        }


def _write_probe_plan(path: Path, rows: int) -> None:
    sequence = ("ACGTacgt" * ((NUCLEOTIDE_LENGTH + 7) // 8))[:NUCLEOTIDE_LENGTH]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample_id in range(rows):
            row = {"sample_id": sample_id, "component": "trainer_probe", "sequence": sequence}
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def _cuda_state() -> dict[str, int]:
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
    }


def run_trainer_preflight(*, batch_size: int, output_path: Path) -> dict[str, Any]:
    """Run one transferred-MNTP optimizer step through the production Trainer."""

    if batch_size <= 0:
        raise ValueError(f"batch size must be positive, got {batch_size}")
    if not torch.cuda.is_available():
        raise RuntimeError("Lightning memory preflight requires CUDA")
    enable_training_determinism()
    L.seed_everything(0, workers=True)
    torch.set_float32_matmul_precision("high")

    probe_dir = output_path.parent / f"lightning-probe-batch-{batch_size}"
    train_plan = probe_dir / "train.jsonl"
    validation_plan = probe_dir / "validation.jsonl"
    _write_probe_plan(train_plan, batch_size)
    _write_probe_plan(validation_plan, 1)

    bundle = load_model_bundle(
        initialization="transferred",
        add_mask=True,
        attention_implementation="sdpa",
    )
    bundle.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    bundle.model.config.use_cache = False
    module = AdaptationModule(model=bundle.model, arm="transferred_mntp", batch_size=batch_size)
    data = ExperimentDataModule(
        train_plan=train_plan,
        validation_plan=validation_plan,
        tokenizer=bundle.tokenizer,
        objective="mntp",
        canonical_token_ids=bundle.canonical_token_ids,
        mask_token_id=bundle.mask_token_id,
        batch_size=batch_size,
        seed=0,
        num_workers=0,
    )
    memory = FirstStepMemoryCallback()
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        max_steps=1,
        max_epochs=-1,
        accumulate_grad_batches=1,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        logger=False,
        callbacks=[memory],
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        deterministic=True,
        default_root_dir=probe_dir,
    )
    started_at = datetime.now(UTC).isoformat()
    try:
        trainer.fit(module, datamodule=data)
    except torch.OutOfMemoryError as error:
        result: dict[str, Any] = {
            "status": "oom",
            "batch_size": batch_size,
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "error": str(error),
            "cuda": _cuda_state(),
        }
    else:
        if trainer.global_step != 1 or memory.measurement is None:
            raise RuntimeError("Lightning preflight did not complete exactly one measured step")
        result = {
            "status": (
                "passed"
                if float(memory.measurement["headroom_fraction"]) >= 0.10
                else "insufficient_headroom"
            ),
            "batch_size": batch_size,
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "memory_and_throughput": memory.measurement,
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
