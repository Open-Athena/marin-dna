"""Train and export one registered exp479 arm."""

from __future__ import annotations

import json
import os
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from exp479_mntp.callbacks import BudgetGuardCallback, RuntimeMetricsCallback
from exp479_mntp.config import (
    CHECKPOINT_INTERVAL,
    EXPERIMENT_TAGS,
    TRAIN_STEPS,
    WANDB_GROUP,
    WANDB_PROJECT,
)
from exp479_mntp.datamodule import ExperimentDataModule
from exp479_mntp.modeling import load_model_bundle
from exp479_mntp.module import AdaptationModule, Arm
from exp479_mntp.probes import ContextProbeCallback
from exp479_mntp.publishing import CheckpointUploadCallback, upload_final_arm


def _instance_start_unix() -> float | None:
    raw = os.getenv("EXP479_INSTANCE_START_UNIX")
    return None if raw is None else float(raw)


def _prior_cost_usd() -> float:
    return float(os.getenv("EXP479_PRIOR_COST_USD", "0"))


def finish_wandb_run(logger: WandbLogger) -> None:
    """Close the process-global W&B run before starting another arm."""

    logger.experiment.finish(exit_code=0)


def train_arm(
    *,
    arm: Arm,
    batch_size: int,
    train_plan: Path,
    validation_plan: Path,
    output_dir: Path,
    seed: int,
    num_workers: int,
    resume_from: Path | None,
    offline_wandb: bool,
    accelerator: str,
    precision: str,
    hf_repo_id: str | None = None,
    checkpoint_upload_steps: tuple[int, ...] | None = None,
) -> None:
    """Run one 1,000-step arm and export its cooled Hugging Face model."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    L.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("high")

    initialization = "scratch" if arm == "scratch_mntp" else "transferred"
    is_mntp = arm != "clm_continuation"
    bundle = load_model_bundle(
        initialization=initialization,
        add_mask=is_mntp,
        attention_implementation="sdpa",
    )
    bundle.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    bundle.model.config.use_cache = False

    module = AdaptationModule(model=bundle.model, arm=arm, batch_size=batch_size)
    data = ExperimentDataModule(
        train_plan=train_plan,
        validation_plan=validation_plan,
        tokenizer=bundle.tokenizer,
        objective="mntp" if is_mntp else "clm",
        canonical_token_ids=bundle.canonical_token_ids,
        mask_token_id=bundle.mask_token_id,
        batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
    )

    run_name = f"dna-exp479-{arm}-seed{seed}"
    logger = WandbLogger(
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        name=run_name,
        tags=[*EXPERIMENT_TAGS, arm],
        save_dir=str(output_dir),
        offline=offline_wandb,
        log_model=False,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="step-{step:04d}",
        every_n_train_steps=CHECKPOINT_INTERVAL,
        save_top_k=-1,
        save_last=True,
        save_weights_only=False,
        enable_version_counter=False,
        auto_insert_metric_name=False,
    )
    callbacks = [
        checkpoint_callback,
        RuntimeMetricsCallback(output_dir / "runtime.json", batch_size),
        BudgetGuardCallback(
            instance_start_unix=_instance_start_unix(),
            prior_cost_usd=_prior_cost_usd(),
        ),
    ]
    callbacks.append(
        ContextProbeCallback(
            validation_plan=validation_plan,
            tokenizer=bundle.tokenizer,
            mask_token_id=bundle.mask_token_id,
            canonical_ids=bundle.canonical_token_ids,
        )
    )
    if hf_repo_id is not None:
        callbacks.append(
            CheckpointUploadCallback(
                checkpoint_dir=output_dir / "checkpoints",
                repo_id=hf_repo_id,
                arm=arm,
                upload_steps=checkpoint_upload_steps,
            )
        )
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=1,
        precision=precision,
        max_steps=TRAIN_STEPS,
        max_epochs=-1,
        accumulate_grad_batches=1,
        val_check_interval=CHECKPOINT_INTERVAL,
        check_val_every_n_epoch=None,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        deterministic=True,
        default_root_dir=str(output_dir),
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=True,
    )
    trainer.fit(
        module, datamodule=data, ckpt_path=None if resume_from is None else str(resume_from)
    )
    if trainer.global_step != TRAIN_STEPS:
        raise RuntimeError(f"{arm} stopped at step {trainer.global_step}, expected {TRAIN_STEPS}")

    bundle.model.config.is_causal = module.attention_mode == "causal"
    export_dir = output_dir / "hf" / f"step-{TRAIN_STEPS}"
    bundle.model.save_pretrained(export_dir, safe_serialization=True)
    bundle.tokenizer.save_pretrained(export_dir)
    manifest = {
        "arm": arm,
        "run_name": run_name,
        "seed": seed,
        "batch_size": batch_size,
        "global_step": int(trainer.global_step),
        "attention_mode": module.attention_mode,
        "input_output_tied": bundle.input_output_tied,
        "optimizer": module.optimizer_values.to_dict(),
        "train_plan": str(train_plan),
        "validation_plan": str(validation_plan),
        "final_lightning_checkpoint": checkpoint_callback.last_model_path,
        "hf_export": str(export_dir),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if hf_repo_id is not None:
        upload_final_arm(output_dir=output_dir, repo_id=hf_repo_id, arm=arm)
    finish_wandb_run(logger)
