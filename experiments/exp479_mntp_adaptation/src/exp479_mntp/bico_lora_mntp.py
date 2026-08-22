"""Rank-16 BICO LoRA with a maximum no-accumulation GH200 batch."""

from __future__ import annotations

import gc
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import pandas as pd
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
from peft import PeftModel, get_peft_model_state_dict

from exp479_mntp.bico_attention_diagnostic import (
    excluded_selected_key_mask,
    install_reflected_future_rope,
)
from exp479_mntp.callbacks import BudgetGuardCallback, RuntimeMetricsCallback
from exp479_mntp.causal_longrun import (
    _artifact_record,
    _write_retention_manifest,
    plot_longrun_stability,
)
from exp479_mntp.config import (
    BUDGET_USD,
    EXPERIMENT_TAGS,
    MODEL_ID,
    MODEL_REVISION,
    SEQUENCE_LENGTH,
    WANDB_PROJECT,
)
from exp479_mntp.data import SequencePlanDataset, plan_sha256
from exp479_mntp.datamodule import ExperimentDataModule
from exp479_mntp.lora_mntp import (
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_RANK,
    LORA_TARGET_MODULES,
    LoraMntpModule,
    RetainedLoraTrajectoryCallback,
    _evaluate_preserving_mode,
    _trajectory_tables,
    build_lora_bundle,
    plot_lora_trajectory,
)
from exp479_mntp.modeling import ModelBundle, model_logits
from exp479_mntp.paired_nucleotide_gate import (
    EXPECTED_VALIDATION_PLAN_SHA256,
    evaluate_readout,
)
from exp479_mntp.publishing import assert_budget_reserve, write_cost_estimate

BICO_LORA_MASK_PROBABILITY = 0.15
BICO_LORA_LEARNING_RATE = 1e-5
BICO_LORA_STANDARD_LEARNING_RATE = 5e-5
BICO_LORA_MAX_INSTANCE_HOURS = 3.3
BICO_LORA_MEMORY_HEADROOM = 0.10
BICO_LORA_BUDGET_RESERVE_USD = 2.0
BICO_LORA_EVALUATION_RESERVE_HOURS = 0.25
BICO_LORA_RUN_NAME = "dna-exp479-bico-lora-r16-pad15-lr1e-5-wsd1000-seed0"
BICO_LORA_WANDB_GROUP = "dna-exp479-bico-lora-information-gate"
BICO_LORA_MODEL_PREFIX = "dna-exp479-bico-lora-r16-pad15"
BICO_LORA_EVALUATION_ARTIFACT = "dna-exp479-bico-lora-r16-information-gate"
BICO_LORA_STANDARD_RUN_NAME = "dna-exp479-bico-lora-r16-pad15-lr5e-5-wsd1000-seed0"
BICO_LORA_STANDARD_MODEL_PREFIX = "dna-exp479-bico-lora-r16-pad15-lr5e-5"
BICO_LORA_STANDARD_EVALUATION_ARTIFACT = "dna-exp479-bico-lora-r16-lr5e-5-information-gate"


@dataclass(frozen=True)
class BicoLoraConfig:
    """The registered BICO LoRA configuration at one selected physical batch."""

    batch_size: int
    rank: int = LORA_RANK
    alpha: int = LORA_ALPHA
    dropout: float = LORA_DROPOUT
    mask_probability: float = BICO_LORA_MASK_PROBABILITY
    learning_rate: float = BICO_LORA_LEARNING_RATE
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    cooldown_start_step: int = 800
    train_steps: int = 1_000
    accumulation_steps: int = 1
    attention_anneal_steps: int = 1

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("BICO LoRA batch size must be positive")
        if self.accumulation_steps != 1:
            raise ValueError("BICO LoRA must not use gradient accumulation")
        if self.rank <= 0 or self.alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if not 0 < self.mask_probability <= 1:
            raise ValueError("mask probability must be in (0, 1]")
        if self.learning_rate <= 0:
            raise ValueError("learning rate must be positive")

    @property
    def microbatch_size(self) -> int:
        """Return the physical batch consumed by every optimizer step."""

        return self.batch_size

    def to_dict(self) -> dict[str, object]:
        """Return the complete serializable training configuration."""

        return asdict(self) | {
            "microbatch_size": self.batch_size,
            "target_modules": list(LORA_TARGET_MODULES),
            "attention_schedule": "full_bico_from_step_0",
            "mask_token": "[PAD]",
            "masked_key_attention": "excluded_at_every_layer",
        }


def excluded_mntp_key_mask(
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    pad_token_id: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build full attention while excluding every shifted MNTP target key."""

    if attention_mask.ndim != 2 or labels.shape != attention_mask.shape:
        raise ValueError("attention mask and labels must share [batch, sequence] shape")
    if input_ids.shape != attention_mask.shape:
        raise ValueError("input IDs must match the attention-mask shape")
    selected_outputs = labels != -100
    if torch.any(selected_outputs[:, -1]):
        raise ValueError("last output position cannot map to a shifted input target")
    excluded_keys = torch.zeros_like(selected_outputs)
    excluded_keys[:, 1:] = selected_outputs[:, :-1]
    if not torch.all(input_ids[excluded_keys] == pad_token_id):
        raise RuntimeError("an excluded MNTP target key is not the registered PAD token")
    key_allowed = torch.logical_and(attention_mask.to(torch.bool), ~excluded_keys)
    batch_size, sequence_length = attention_mask.shape
    allowed = key_allowed[:, None, None, :].expand(
        batch_size,
        1,
        sequence_length,
        sequence_length,
    )
    additive = torch.zeros(allowed.shape, dtype=dtype, device=attention_mask.device)
    return additive.masked_fill(~allowed, torch.finfo(dtype).min)


def build_bico_lora_bundle(config: BicoLoraConfig) -> tuple[ModelBundle, int]:
    """Build rank-16 LoRA while selecting the existing PAD token for masking."""

    bundle, trainable_count = build_lora_bundle(config)  # type: ignore[arg-type]
    pad_token_id = bundle.tokenizer.pad_token_id
    if pad_token_id is None or int(pad_token_id) < 0:
        raise RuntimeError("source tokenizer lacks the BICO PAD token")
    configured_pad_token_id = bundle.model.config.pad_token_id
    if configured_pad_token_id is None or int(configured_pad_token_id) != int(pad_token_id):
        raise RuntimeError("source model and tokenizer disagree on the BICO PAD token")
    if int(pad_token_id) in bundle.canonical_token_ids:
        raise RuntimeError("the BICO PAD token aliases a canonical nucleotide")
    install_reflected_future_rope(bundle.model)
    return (
        ModelBundle(
            model=bundle.model,
            tokenizer=bundle.tokenizer,
            canonical_token_ids=bundle.canonical_token_ids,
            mask_token_id=int(pad_token_id),
            input_output_tied=bundle.input_output_tied,
        ),
        trainable_count,
    )


class BicoLoraModule(LoraMntpModule):
    """Train only LoRA matrices with reflected future RoPE and excluded PAD keys."""

    def __init__(self, *, model: PeftModel, config: BicoLoraConfig, seed: int = 0) -> None:
        super().__init__(model=model, config=config, seed=seed)  # type: ignore[arg-type]
        self.bico_config = config
        self.supervised_masked_targets = 0

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        if self.bico_config.accumulation_steps != 1:
            raise RuntimeError("BICO LoRA accumulation changed after configuration")
        pad_token_id = self.model.config.pad_token_id
        if pad_token_id is None:
            raise RuntimeError("BICO LoRA model lacks a PAD token ID")
        attention_mask = excluded_mntp_key_mask(
            batch["attention_mask"],
            batch["labels"],
            batch["input_ids"],
            pad_token_id=int(pad_token_id),
            dtype=torch.bfloat16,
        )
        self._latest_attention_future_edge_probability = 1.0
        install_reflected_future_rope(self.model)
        return model_logits(
            self.model,
            input_ids=batch["input_ids"],
            attention_mask=attention_mask,
            attention_mode="full",
        )

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        self.supervised_masked_targets += int((batch["labels"] != -100).sum())
        return super().training_step(batch, batch_idx)


def evaluate_bico_readout(
    bundle: ModelBundle,
    *,
    validation_plan: Path,
    batch_size: int,
    readout: str,
) -> pd.DataFrame:
    """Evaluate one excluded-PAD, reflected-RoPE full-attention readout."""

    def exclude_selected_key(
        token_mask: torch.Tensor,
        sample_ids: torch.Tensor,
        output_positions: torch.Tensor,
    ) -> torch.Tensor:
        del sample_ids
        return excluded_selected_key_mask(
            token_mask,
            output_positions,
            dtype=torch.bfloat16,
        )

    bundle.model.to(device="cuda")
    was_training = bundle.model.training
    install_reflected_future_rope(bundle.model)
    scores = evaluate_readout(
        bundle,
        validation_plan=validation_plan,
        batch_size=batch_size,
        readout=readout,
        attention_mode="full",
        attention_mask_transform=exclude_selected_key,
    )
    bundle.model.train(was_training)
    return scores


class RetainedBicoLoraTrajectoryCallback(RetainedLoraTrajectoryCallback):
    """Retain adapter milestones and evaluate the registered BICO readout."""

    def __init__(self, *, model_prefix: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model_prefix = model_prefix

    def _evaluate_and_retain(self, step: int) -> None:
        if step in self.saved:
            return
        assert_budget_reserve()
        scores = evaluate_bico_readout(
            self.bundle,
            validation_plan=self.validation_plan,
            batch_size=self.evaluation_batch_size,
            readout=f"lora_full_step{step:04d}",
        )
        scores["optimizer_step"] = step
        self.score_frames.append(scores)
        self.run.log(
            {
                "bico_lora_gate/step": step,
                "bico_lora_gate/full_nucleotide_ce": float(scores["nucleotide_ce"].mean()),
                "bico_lora_gate/full_nucleotide_accuracy": float(
                    scores["nucleotide_correct"].mean()
                ),
            }
        )
        adapter_dir = self.output_dir / "adapters" / f"step-{step:04d}"
        self.bundle.model.save_pretrained(adapter_dir, safe_serialization=True)
        artifact = wandb.Artifact(
            f"{self.model_prefix}-step-{step:04d}",
            type="model",
            metadata={
                "optimizer_step": step,
                "format": "peft_adapter",
                "base_model": MODEL_ID,
                "base_revision": MODEL_REVISION,
                "mask_token": "[PAD]",
                "attention": "BICO reflected future RoPE",
            },
        )
        artifact.add_dir(str(adapter_dir), name="adapter")
        logged = self.run.log_artifact(artifact, aliases=[f"step-{step:04d}"])
        logged.wait()
        self.retained.append(_artifact_record(logged, kind="peft_adapter", step=step))
        self.saved.add(step)


def _build_training_objects(
    *,
    batch_size: int,
    train_plan: Path,
    validation_plan: Path,
    seed: int,
    num_workers: int,
    learning_rate: float = BICO_LORA_LEARNING_RATE,
) -> tuple[BicoLoraConfig, ModelBundle, int, BicoLoraModule, ExperimentDataModule]:
    config = BicoLoraConfig(batch_size=batch_size, learning_rate=learning_rate)
    bundle, trainable_count = build_bico_lora_bundle(config)
    if not isinstance(bundle.model, PeftModel):
        raise TypeError("BICO LoRA builder did not return a PEFT model")
    bundle.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    bundle.model.enable_input_require_grads()
    bundle.model.config.use_cache = False
    module = BicoLoraModule(model=bundle.model, config=config, seed=seed)
    data = ExperimentDataModule(
        train_plan=train_plan,
        validation_plan=validation_plan,
        tokenizer=bundle.tokenizer,
        objective="mntp",
        canonical_token_ids=bundle.canonical_token_ids,
        mask_token_id=bundle.mask_token_id,
        batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
        fixed_mask_probability=config.mask_probability,
    )
    return config, bundle, trainable_count, module, data


def run_bico_lora_preflight(
    *,
    batch_size: int,
    train_plan: Path,
    validation_plan: Path,
    output_path: Path,
    seed: int,
    learning_rate: float = BICO_LORA_LEARNING_RATE,
) -> dict[str, object]:
    """Exercise two exact optimizer steps and enforce memory and budget headroom."""

    if not torch.cuda.is_available():
        raise RuntimeError("BICO LoRA preflight requires a CUDA GPU")
    if plan_sha256(validation_plan) != EXPECTED_VALIDATION_PLAN_SHA256:
        raise RuntimeError("BICO LoRA preflight validation plan differs from the fixed gate")
    dataset = SequencePlanDataset(train_plan)
    if len(dataset) < 2 * batch_size:
        raise ValueError("preflight plan lacks two complete candidate batches")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    payload: dict[str, object] = {
        "batch_size": batch_size,
        "accumulation_steps": 1,
        "learning_rate": learning_rate,
    }
    trainer: L.Trainer | None = None
    data: ExperimentDataModule | None = None
    module: BicoLoraModule | None = None
    bundle: ModelBundle | None = None
    try:
        L.seed_everything(seed, workers=True)
        torch.set_float32_matmul_precision("high")
        config, bundle, trainable_count, module, data = _build_training_objects(
            batch_size=batch_size,
            train_plan=train_plan,
            validation_plan=validation_plan,
            seed=seed,
            num_workers=0,
            learning_rate=learning_rate,
        )
        torch.cuda.reset_peak_memory_stats()
        trainer = L.Trainer(
            accelerator="gpu",
            devices=1,
            precision="bf16-mixed",
            max_steps=2,
            max_epochs=-1,
            accumulate_grad_batches=1,
            limit_val_batches=0,
            num_sanity_val_steps=0,
            deterministic=True,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        step_started = time.monotonic()
        trainer.fit(module, datamodule=data)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - step_started
        total_memory = int(torch.cuda.get_device_properties(0).total_memory)
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
        headroom = (total_memory - peak_reserved) / total_memory
        seconds_per_step = elapsed / 2
        instance_start = float(os.getenv("EXP479_INSTANCE_START_UNIX", str(started)))
        prior_cost = float(os.getenv("EXP479_PRIOR_COST_USD", "0"))
        price = float(os.getenv("EXP479_INSTANCE_PRICE_PER_HOUR_USD", "2.29"))
        accrued = prior_cost + (time.time() - instance_start) / 3600 * price
        projected_training_hours = seconds_per_step * config.train_steps / 3600
        projected_total = (
            accrued + (projected_training_hours + BICO_LORA_EVALUATION_RESERVE_HOURS) * price
        )
        trace = module.gradient_norm_trace
        finite = (
            int(trainer.global_step) == 2
            and len(trace) == 2
            and all(
                math.isfinite(float(row["train_loss"]))
                and math.isfinite(float(row["pre_clip_gradient_norm"]))
                for row in trace
            )
        )
        memory_passed = headroom >= BICO_LORA_MEMORY_HEADROOM
        budget_passed = projected_total < BUDGET_USD - BICO_LORA_BUDGET_RESERVE_USD
        passed = finite and memory_passed and budget_passed
        payload |= {
            "status": "passed" if passed else "rejected",
            "trainable_parameters": trainable_count,
            "optimizer_steps": int(trainer.global_step),
            "finite_loss_and_gradients": finite,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "total_memory_bytes": total_memory,
            "headroom_fraction": headroom,
            "memory_headroom_required": BICO_LORA_MEMORY_HEADROOM,
            "seconds_per_step": seconds_per_step,
            "projected_training_hours": projected_training_hours,
            "projected_total_cost_usd": projected_total,
            "budget_guard_total_usd": BUDGET_USD - BICO_LORA_BUDGET_RESERVE_USD,
            "budget_passed": budget_passed,
        }
    except Exception as error:  # noqa: BLE001 - every rejected batch needs a JSON record.
        payload |= {
            "status": "oom" if isinstance(error, torch.OutOfMemoryError) else "failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    finally:
        payload["elapsed_seconds"] = time.time() - started
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        del trainer, data, module, bundle
        gc.collect()
        torch.cuda.empty_cache()
    return payload


def _assert_training_plan(train_plan: Path, validation_plan: Path, batch_size: int) -> None:
    if plan_sha256(validation_plan) != EXPECTED_VALIDATION_PLAN_SHA256:
        raise RuntimeError("BICO LoRA validation plan differs from the fixed gate")
    rows = len(SequencePlanDataset(train_plan))
    expected = batch_size * 1_000
    if rows != expected:
        raise RuntimeError(f"BICO LoRA train plan has {rows} rows, expected {expected}")


def run_bico_lora_mntp(
    *,
    artifact_dir: Path,
    output_dir: Path,
    train_plan: Path,
    validation_plan: Path,
    preflight_dir: Path,
    batch_size: int,
    seed: int,
    num_workers: int,
    evaluation_batch_size: int,
    n_bootstrap: int,
    learning_rate: float = BICO_LORA_LEARNING_RATE,
    run_name: str = BICO_LORA_RUN_NAME,
    model_prefix: str = BICO_LORA_MODEL_PREFIX,
    evaluation_artifact: str = BICO_LORA_EVALUATION_ARTIFACT,
) -> None:
    """Train the selected no-accumulation BICO LoRA and apply the paired gate."""

    if not torch.cuda.is_available():
        raise RuntimeError("BICO LoRA training requires one CUDA GPU")
    if evaluation_batch_size <= 0 or n_bootstrap <= 0:
        raise ValueError("evaluation batch size and bootstrap count must be positive")
    _assert_training_plan(train_plan, validation_plan, batch_size)
    selected_preflight = preflight_dir / f"batch-{batch_size}.json"
    if not selected_preflight.exists():
        raise RuntimeError("selected BICO LoRA batch lacks a preflight record")
    preflight_payload = json.loads(selected_preflight.read_text(encoding="utf-8"))
    if preflight_payload.get("status") != "passed":
        raise RuntimeError("selected BICO LoRA batch did not pass preflight")
    if int(preflight_payload.get("batch_size", -1)) != batch_size:
        raise RuntimeError("selected BICO LoRA preflight records a different batch")
    if float(preflight_payload.get("learning_rate", float("nan"))) != learning_rate:
        raise RuntimeError("selected BICO LoRA preflight records a different learning rate")
    price = float(os.getenv("EXP479_INSTANCE_PRICE_PER_HOUR_USD", "2.29"))
    prior_cost = float(os.getenv("EXP479_PRIOR_COST_USD", "0"))
    if prior_cost + BICO_LORA_MAX_INSTANCE_HOURS * price >= BUDGET_USD:
        raise RuntimeError("BICO LoRA projection reaches the issue budget cap")
    assert_budget_reserve()
    output_dir.mkdir(parents=True, exist_ok=True)
    budget_path = output_dir / "prelaunch-budget.json"
    budget_path.write_text(
        json.dumps(
            {
                "prior_cost_usd": prior_cost,
                "maximum_instance_hours": BICO_LORA_MAX_INSTANCE_HOURS,
                "price_per_hour_usd": price,
                "projected_total_usd": prior_cost + BICO_LORA_MAX_INSTANCE_HOURS * price,
                "budget_cap_usd": BUDGET_USD,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    L.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("high")
    config, bundle, trainable_count, module, data = _build_training_objects(
        batch_size=batch_size,
        train_plan=train_plan,
        validation_plan=validation_plan,
        seed=seed,
        num_workers=num_workers,
        learning_rate=learning_rate,
    )
    logger = WandbLogger(
        project=WANDB_PROJECT,
        group=BICO_LORA_WANDB_GROUP,
        name=run_name,
        tags=[*EXPERIMENT_TAGS, "lora", "rank-16", "bico", "pad-mask", "no-accumulation"],
        save_dir=str(output_dir),
        log_model=False,
    )
    logger.log_hyperparams(
        config.to_dict()
        | {
            "base_model": MODEL_ID,
            "base_revision": MODEL_REVISION,
            "trainable_parameters": trainable_count,
            "model_tokens": config.train_steps * batch_size * SEQUENCE_LENGTH,
        }
    )
    run = logger.experiment
    trajectory = RetainedBicoLoraTrajectoryCallback(
        bundle=bundle,
        validation_plan=validation_plan,
        evaluation_batch_size=evaluation_batch_size,
        output_dir=output_dir,
        run=run,
        model_prefix=model_prefix,
    )
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        max_steps=config.train_steps,
        max_epochs=-1,
        accumulate_grad_batches=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        deterministic=True,
        default_root_dir=str(output_dir),
        logger=logger,
        callbacks=[
            trajectory,
            RuntimeMetricsCallback(output_dir / "runtime.json", batch_size),
            BudgetGuardCallback(
                instance_start_unix=(
                    None
                    if os.getenv("EXP479_INSTANCE_START_UNIX") is None
                    else float(os.environ["EXP479_INSTANCE_START_UNIX"])
                ),
                prior_cost_usd=prior_cost,
                price_per_hour_usd=price,
                reserve_usd=BICO_LORA_BUDGET_RESERVE_USD,
            ),
        ],
        enable_checkpointing=False,
    )

    try:
        trainer.fit(module, datamodule=data)
        if trainer.global_step != config.train_steps:
            raise RuntimeError(
                f"BICO LoRA stopped at step {trainer.global_step}, expected {config.train_steps}"
            )
        scores, summary, comparisons, gate = _trajectory_tables(
            trajectory,
            n_bootstrap=n_bootstrap,
        )
        with bundle.model.disable_adapter():
            final_disabled = _evaluate_preserving_mode(
                bundle,
                validation_plan=validation_plan,
                batch_size=evaluation_batch_size,
                readout="source_causal_adapter_disabled_step1000",
                attention_mode="causal",
            )
        if trajectory.source_scores is None:
            raise RuntimeError("BICO LoRA source preservation check lacks step-0 scores")
        columns = (
            "sample_id",
            "target_nucleotide_index",
            "nucleotide_ce",
            "nucleotide_correct",
            "full_vocab_ce",
            "full_vocab_correct",
        )
        source_preserved = all(
            np.array_equal(
                trajectory.source_scores[column].to_numpy(),
                final_disabled[column].to_numpy(),
            )
            for column in columns
        )
        if not source_preserved:
            raise RuntimeError("disabled-adapter causal readout changed despite frozen base")

        scores_path = output_dir / "paired-nucleotide-scores.csv"
        summary_path = output_dir / "paired-nucleotide-summary.csv"
        comparisons_path = output_dir / "paired-nucleotide-comparisons.csv"
        gate_path = output_dir / "paired-nucleotide-gate.json"
        scores.to_csv(scores_path, index=False)
        summary.to_csv(summary_path, index=False)
        comparisons.to_csv(comparisons_path, index=False)
        gate_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
        trajectory_figure = output_dir / "figures" / "paired-nucleotide-trajectory"
        plot_lora_trajectory(summary, comparisons, trajectory_figure)

        trace = pd.DataFrame(module.gradient_norm_trace)
        if trace["step"].astype(int).tolist() != list(range(config.train_steps)):
            raise RuntimeError("BICO LoRA gradient trace omits an optimizer step")
        trace_path = output_dir / "gradient-norm-trace.csv"
        trace.to_csv(trace_path, index=False)
        stability_figure = output_dir / "figures" / "training-stability"
        plot_longrun_stability(
            trace,
            stability_figure,
            title="Rank-16 BICO LoRA MNTP training stability",
        )

        checkpoint_path = output_dir / "checkpoints" / "step-1000-adapter-optimizer.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        scheduler = trainer.lr_scheduler_configs[0].scheduler
        torch.save(
            {
                "global_step": int(trainer.global_step),
                "adapter_state_dict": {
                    name: tensor.detach().cpu()
                    for name, tensor in get_peft_model_state_dict(bundle.model).items()
                },
                "optimizer_state_dict": trainer.optimizers[0].state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
                "train_plan_sha256": plan_sha256(train_plan),
                "validation_plan_sha256": plan_sha256(validation_plan),
                "next_training_sample_id": config.train_steps * batch_size,
                "config": config.to_dict(),
            },
            checkpoint_path,
        )
        checkpoint_artifact = wandb.Artifact(
            f"{model_prefix}-step-1000-optimizer",
            type="model",
            metadata={
                "optimizer_step": 1_000,
                "format": "adapter_optimizer_rng",
                "contains_optimizer_state": True,
                "base_weights_included": False,
            },
        )
        checkpoint_artifact.add_file(str(checkpoint_path))
        logged_checkpoint = run.log_artifact(
            checkpoint_artifact,
            aliases=["step-1000-optimizer"],
        )
        logged_checkpoint.wait()
        trajectory.retained.append(
            _artifact_record(logged_checkpoint, kind="adapter_optimizer_checkpoint", step=1_000)
        )
        retention_path = output_dir / "retention-manifest.json"
        _write_retention_manifest(retention_path, trajectory.retained)
        cost_path = write_cost_estimate(artifact_dir=artifact_dir)
        manifest = {
            "status": "completed",
            "run_name": run_name,
            "wandb_url": run.get_url(),
            "base_model": MODEL_ID,
            "base_revision": MODEL_REVISION,
            "base_frozen": True,
            "disabled_adapter_causal_bit_exact": source_preserved,
            "trainable_parameters": trainable_count,
            "configuration": config.to_dict(),
            "selected_preflight": preflight_payload,
            "physical_batch_size": batch_size,
            "accumulation_steps": 1,
            "sequences": config.train_steps * batch_size,
            "model_tokens": config.train_steps * batch_size * SEQUENCE_LENGTH,
            "supervised_masked_targets": module.supervised_masked_targets,
            "training_objective": "sequence-balanced repeat-weighted fixed-15%-mask MNTP",
            "attention_training": "BICO reflected future RoPE with masked PAD keys excluded",
            "train_plan_sha256": plan_sha256(train_plan),
            "validation_plan_sha256": plan_sha256(validation_plan),
            "paired_target_count": 640,
            "gate": gate,
            "checkpoint_retention": "adapter-only and optimizer-bearing W&B artifacts",
            "checkpoint_deletion": "not performed",
            "hugging_face_upload": "not performed",
            "vep_evaluation": "not performed",
            "nucleotide_dependency": "not performed",
            "knowledge_base_update": "not performed",
        }
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        run.log(
            {
                "bico_lora_gate/summary": wandb.Table(dataframe=summary),
                "bico_lora_gate/comparisons": wandb.Table(dataframe=comparisons),
                "bico_lora_gate/trajectory": wandb.Image(
                    str(trajectory_figure.with_suffix(".png"))
                ),
                "bico_lora_gate/stability": wandb.Image(str(stability_figure.with_suffix(".png"))),
            }
        )
        run.summary["bico_lora_gate/passed"] = bool(gate["passed"])
        run.summary["bico_lora_gate/source_causal_preserved"] = source_preserved
        run.summary["bico_lora_gate/physical_batch_size"] = batch_size
        run.summary["bico_lora_gate/model_tokens"] = manifest["model_tokens"]
        run.summary["bico_lora_gate/supervised_masked_targets"] = module.supervised_masked_targets
        result_artifact = wandb.Artifact(evaluation_artifact, type="evaluation")
        result_artifact.add_dir(str(preflight_dir), name="batch-preflights")
        for path in (
            scores_path,
            summary_path,
            comparisons_path,
            gate_path,
            trace_path,
            output_dir / "runtime.json",
            budget_path,
            retention_path,
            manifest_path,
            cost_path,
            trajectory_figure.with_suffix(".svg"),
            stability_figure.with_suffix(".svg"),
        ):
            result_artifact.add_file(str(path))
        logged_result = run.log_artifact(
            result_artifact,
            aliases=["paired-gate", "step-1000", "bico"],
        )
        logged_result.wait()
        run.finish(exit_code=0)
    except BaseException:
        run.finish(exit_code=1)
        raise
    finally:
        del trainer, data, module, bundle
        gc.collect()
        torch.cuda.empty_cache()
