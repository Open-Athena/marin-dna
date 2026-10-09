"""Six d768 DNA trials using the native Hero EP step and exact token clocks."""

import argparse
import dataclasses
import gc
import hashlib
import json
import logging
import math
import os
import time
from collections.abc import Callable
from enum import Enum
from typing import Any, Literal

import equinox as eqx
import fsspec
import jax
import jmp
import numpy as np
from fray.cluster import ResourceConfig
from fray.current_client import current_client
from fray.device_flops import device_flops_for_jax_device
from fray.types import Entrypoint, JobRequest, create_environment
from iris.rpc.proto_display import priority_band_value
from jax.experimental import multihost_utils
from levanter import tracker
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import discover_checkpoint_candidates
from levanter.grug.grug_moe import (
    MOE_DROPPED_ASSIGNMENTS_METRIC,
    MOE_VALID_ASSIGNMENTS_METRIC,
)
from levanter.grug.sharding import compact_grug_mesh
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.training.training import resolve_training_env
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration

from exp586_moe.config import (
    EPOCH_TOKENS,
    GLOBAL_BATCH_SIZE,
    PEAK_CHECKPOINT_UPDATE,
    SOURCE_CHECKPOINT,
    SOURCE_METADATA_DIGEST,
    TOKENS_PER_UPDATE,
    TOTAL_UPDATES,
    TRAINING_TOKENS,
    d768_config,
)
from exp586_moe.corpus import corpus_digest, distributed_batch, open_dataset
from exp586_moe.data import (
    pretrained_dna_tokenizer,
    scratch_tokenizer,
    tokenizer_digest,
)
from exp586_moe.routing_probe import routing_diagnostics
from exp586_moe.schedule import (
    HeroTokenSchedule,
    TokenClock,
    optimizer_config,
    set_token_learning_rates,
)
from exp586_moe.state import (
    fresh_state,
    restore_dna_checkpoint,
    save_dna_checkpoint,
    state_all_finite,
)
from exp586_moe.tracking import wandb_history_offset
from experiments.grug.dispatch import _forwarded_env_vars
from experiments.grug.moe_hero_ep.model import Transformer, apply_qb_betas
from experiments.grug.moe_hero_ep.train import (
    MasterParamMode,
    _apply_hero_ep_runtime_defaults,
    _compute_flops,
    _make_train_step,
    grug_trainer_mesh_config,
)
from experiments.grug.moe_hero_ep.weights import restore_weights

BATCH_SIZE = GLOBAL_BATCH_SIZE
COOLDOWN_UPDATE = PEAK_CHECKPOINT_UPDATE
VALIDATION_EXAMPLES = 512
ARTIFACT_DATE = "2026.10.08"
PERMANENT_CHECKPOINT_COUNT = 8
PERMANENT_CHECKPOINT_UPDATES = frozenset(
    {
        math.ceil(
            TRAINING_TOKENS
            * checkpoint_index
            / (PERMANENT_CHECKPOINT_COUNT - 1)
            / TOKENS_PER_UPDATE
        )
        for checkpoint_index in range(1, PERMANENT_CHECKPOINT_COUNT - 1)
    }
    | {PEAK_CHECKPOINT_UPDATE, TOTAL_UPDATES}
)


def theoretical_flops(device_kind: str, device_count: int) -> tuple[float, float]:
    """Return the standard dense-BF16 peak for one device and the whole mesh."""
    if device_count <= 0:
        raise ValueError("Device count must be positive")
    per_device = device_flops_for_jax_device(device_kind)
    if per_device is None:
        raise ValueError(f"Unknown theoretical FLOPs for {device_kind}")
    return float(per_device), float(per_device * device_count)


def throughput_flop_metrics(
    *,
    flops_per_example: float,
    completed_examples: int,
    batch_size: int,
    elapsed_seconds: float,
    theoretical_flops_total: float,
) -> dict[str, float]:
    """Match Levanter's analytic MFU convention using the durable example clock."""
    if (
        flops_per_example <= 0
        or completed_examples < 0
        or batch_size <= 0
        or elapsed_seconds <= 0
        or theoretical_flops_total <= 0
    ):
        raise ValueError("FLOP metrics require positive rates and a nonnegative clock")
    model_flops_per_second = flops_per_example * batch_size / elapsed_seconds
    return {
        "throughput/gflops_per_second": model_flops_per_second / 1e9,
        "throughput/mfu": model_flops_per_second / theoretical_flops_total * 100.0,
        "throughput/total_gflops": flops_per_example * completed_examples / 1e9,
    }


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    condition: Literal["scratch", "pretrained"]
    lr_multiplier: float
    cluster: str
    nodes: int
    seed: int = 0
    checkpoint_interval_seconds: int = 900
    recovery_checkpoint_delay_updates: int = 25
    moe_backend: Literal["pooled", "fixed", "ring"] | None = None

    def __post_init__(self) -> None:
        multipliers = {
            "pretrained": (0.5, 1.0, 2.0),
            "scratch": (1.0, 3.0, 10.0),
        }
        if (
            self.condition not in multipliers
            or self.lr_multiplier not in multipliers[self.condition]
            or self.seed != 0
        ):
            raise ValueError("Select one of the six authorized trials")
        if self.cluster not in ("cw-rno2a", "cw-us-east-02a") or self.nodes not in (
            1,
            2,
            4,
            8,
        ):
            raise ValueError("Training requires an 8-, 16-, 32- or 64-H100 placement")
        if self.checkpoint_interval_seconds <= 0:
            raise ValueError("Checkpoint interval must be positive")
        if self.recovery_checkpoint_delay_updates <= 0:
            raise ValueError("Recovery checkpoint delay must be positive")
        if self.moe_backend not in (None, "pooled", "fixed", "ring") or (
            self.condition == "pretrained" and self.moe_backend not in (None, "pooled")
        ):
            raise ValueError("Select an approved MoE runtime backend")

    @property
    def context_axis_size(self) -> int:
        return 1

    @property
    def run_id(self) -> str:
        multiplier = {0.5: "0p5", 1.0: "1", 2.0: "2", 3.0: "3", 10.0: "10"}[
            self.lr_multiplier
        ]
        return (
            f"exp586-plantcad2-d768-{self.condition}-lrm{multiplier}-seed{self.seed}-v1"
        )

    @property
    def checkpoint_root(self) -> str:
        return f"s3://marin-us-east-02a/MarinDNA/exp586_plantcad2_moe/checkpoints/{self.run_id}/{ARTIFACT_DATE}/checkpoints"

    @property
    def schedule(self) -> HeroTokenSchedule:
        return HeroTokenSchedule()

    @property
    def scientific_model(self) -> Any:
        """Stable logical model contract shared by transport-compatible resumes."""
        return dataclasses.replace(
            d768_config(self.condition),
            capacity_factor=32.0,
            pooled_transport_capacity_factor=8.0,
            expert_chunks=1,
        )

    @property
    def resolved_moe_backend(self) -> Literal["pooled", "fixed", "ring"]:
        return self.moe_backend or ("ring" if self.condition == "scratch" else "pooled")

    @property
    def model(self) -> Any:
        implementation = {
            "pooled": "fixed_pooled_wave_all_to_all",
            "fixed": "fixed_all_to_all",
            "ring": "ring",
        }[self.resolved_moe_backend]
        return dataclasses.replace(
            self.scientific_model,
            moe_implementation=implementation,
            pooled_transport_capacity_factor=(
                8.0 if self.resolved_moe_backend == "pooled" else None
            ),
        )


def scientific_config(config: TrainingConfig, tokenizer_sha256: str) -> dict[str, Any]:
    """Immutable science and lineage; placement remains operational."""
    return {
        "schema": 1,
        "condition": config.condition,
        "seed": config.seed,
        "batch_size": BATCH_SIZE,
        "model": dataclasses.asdict(config.scientific_model),
        "optimizer": dataclasses.asdict(
            optimizer_config(
                reference_tokens_per_update=TOKENS_PER_UPDATE,
                multiplier=config.lr_multiplier,
            )
        ),
        "schedule": dataclasses.asdict(config.schedule),
        "optimizer_reference_tokens": TRAINING_TOKENS,
        "source_checkpoint": SOURCE_CHECKPOINT
        if config.condition == "pretrained"
        else None,
        "source_metadata_sha256": SOURCE_METADATA_DIGEST
        if config.condition == "pretrained"
        else None,
        "tokenizer_sha256": tokenizer_sha256,
        "corpus_sha256": corpus_digest(),
        "data_order": "exp472-block256-window512-feistel-mixture2048-seed0",
        "augmentation": "exp472-occurrence-bernoulli-472",
        "precision": "native-bf16-fp32-pinned-host-master",
        "z_loss_weight": 1e-4,
    }


def config_json(value: dict[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda x: x.name if isinstance(x, Enum) else str(x),
    )


def checkpoint_due(
    update: int,
    initial_update: int,
    seconds_since_save: float,
    interval_seconds: int,
    peak_update: int = COOLDOWN_UPDATE,
    recovery_delay_updates: int = 25,
) -> bool:
    """Bound replay after every restore as well as during uninterrupted training."""
    return (
        is_permanent_checkpoint(update, peak_update)
        or update == initial_update + recovery_delay_updates
        or seconds_since_save >= interval_seconds
    )


def is_permanent_checkpoint(update: int, peak_update: int = COOLDOWN_UPDATE) -> bool:
    """Retain the LR peak, six horizon milestones, and the final state."""
    return update == peak_update or update in PERMANENT_CHECKPOINT_UPDATES


def primary_json(action: Callable[[], Any]) -> Any:
    """Coordinate primary-only I/O, including failures, across the GPU gang."""
    blob = b""
    if jax.process_index() == 0:
        try:
            result = {"value": action()}
        except Exception as error:  # noqa: BLE001 -- broadcast failure before raising
            result = {"error": f"{type(error).__name__}: {error}"}
        blob = json.dumps(result).encode()
    length = int(
        multihost_utils.broadcast_one_to_all(np.asarray(len(blob), dtype=np.int32))
    )
    payload = (
        np.frombuffer(blob, dtype=np.uint8).copy()
        if blob
        else np.zeros(length, dtype=np.uint8)
    )
    result = json.loads(bytes(multihost_utils.broadcast_one_to_all(payload)))
    if "error" in result:
        raise RuntimeError(result["error"])
    return result["value"]


def bind_and_discover(
    config: TrainingConfig, binding: dict[str, Any]
) -> tuple[str, str | None]:
    serialized = config_json(binding)
    digest = hashlib.sha256(serialized.encode()).hexdigest()
    path = StoragePath(config.checkpoint_root) / "training.json"
    if path.exists():
        if path.read_text() != serialized:
            raise ValueError("Trial scientific configuration changed")
    else:
        path.write_text(serialized)
        if path.read_text() != serialized:
            raise ValueError("Scientific configuration readback failed")
    candidates = discover_checkpoint_candidates(config.checkpoint_root)
    for candidate in candidates:
        if candidate.metadata.get("scientific_config_sha256") != digest:
            raise ValueError("Checkpoint scientific configuration mismatch")
    if candidates and candidates[-1].step >= config.schedule.peak_update:
        peaks = [c for c in candidates if c.step == config.schedule.peak_update]
        if len(peaks) != 1 or peaks[0].metadata.get("is_temporary") is not False:
            raise ValueError("Permanent peak checkpoint is missing")
    return digest, candidates[-1].path if candidates else None


def prepare_checkpoint(config: TrainingConfig, clock: TokenClock) -> str:
    """Clear an interrupted save before its destination is reused.

    Call through primary_json with exactly one active dispatch for this trial.
    Levanter publishes metadata.json only after every shard commits. Partial
    arrays can retain incompatible chunk shapes after a GPU gang resize.
    """
    check_clock(clock)
    checkpoint = f"{config.checkpoint_root}/step-{clock.updates}"
    fs, path = fsspec.core.url_to_fs(checkpoint)
    fs.invalidate_cache(path)
    if fs.exists(path + "/metadata.json"):
        raise ValueError("Refusing to overwrite a committed checkpoint")
    if fs.exists(path):
        logging.getLogger(__name__).warning(
            "Removing incomplete checkpoint before retry: %s", checkpoint
        )
        fs.rm(path, recursive=True)
        fs.invalidate_cache(path)
        if fs.exists(path):
            raise RuntimeError("Incomplete checkpoint cleanup failed")
    return checkpoint


def commit_and_prune(
    config: TrainingConfig, checkpoint: str, digest: str, clock: TokenClock
) -> None:
    """Publish only a committed checkpoint; retain two temporary recovery saves."""
    metadata = json.loads((StoragePath(checkpoint) / "metadata.json").read_text())
    if (
        metadata.get("exp586_clock") != dataclasses.asdict(clock)
        or metadata.get("scientific_config_sha256") != digest
    ):
        raise ValueError("Checkpoint commit readback mismatch")
    if checkpoint != f"{config.checkpoint_root}/step-{clock.updates}":
        raise ValueError("Checkpoint is outside the active trial")
    pointer = StoragePath(config.checkpoint_root) / "latest.json"
    payload = json.dumps(
        {"checkpoint": checkpoint, "metadata": metadata}, sort_keys=True
    )
    pointer.write_text(payload)
    if pointer.read_text() != payload:
        raise ValueError("Latest checkpoint pointer readback failed")
    temporary = [
        c
        for c in discover_checkpoint_candidates(config.checkpoint_root)
        if c.metadata.get("is_temporary")
        and c.metadata.get("scientific_config_sha256") == digest
    ]
    for candidate in temporary[:-2]:
        if (
            not candidate.path.startswith(config.checkpoint_root + "/step-")
            or candidate.step >= clock.updates
        ):
            raise ValueError("Unsafe checkpoint retention candidate")
        fs, path = fsspec.core.url_to_fs(candidate.path)
        fs.rm(path, recursive=True)


def check_clock(clock: TokenClock) -> None:
    if (
        not 0 <= clock.updates <= TOTAL_UPDATES
        or clock.examples != clock.updates * BATCH_SIZE
        or clock.input_tokens != clock.updates * TOKENS_PER_UPDATE
        or clock.loss_targets != clock.examples * 8191
    ):
        raise ValueError("Checkpoint data cursor and token clocks disagree")


class ValidationSample:
    """A fixed, evenly spaced sample across all validation Parquet shards."""

    def __init__(self) -> None:
        self.dataset = open_dataset("validation")

    def get_batch(self, indices: list[int]) -> Any:
        return self.dataset.get_batch(
            [i * len(self.dataset) // VALIDATION_EXAMPLES for i in indices]
        )


@eqx.filter_jit
def validation_step(params: Transformer, pending: jax.Array, batch: Any) -> Any:
    mp = jmp.get_policy("params=bfloat16,compute=bfloat16,output=bfloat16")
    return mp.cast_to_compute(apply_qb_betas(params, pending)).next_token_loss(
        batch.tokens,
        batch.loss_weight,
        mask=batch.attn_mask,
        reduction="mean",
        logsumexp_weight=None,
        return_router_metrics=True,
    )


def evaluate(
    state: Any, dataset: ValidationSample, tokenizer: Any, mesh: jax.sharding.Mesh
) -> dict[str, float]:
    losses, dropped, valid = [], 0, 0
    for start in range(0, VALIDATION_EXAMPLES, BATCH_SIZE):
        batch = distributed_batch(
            dataset, tokenizer, mesh, start, BATCH_SIZE, augment=False
        )
        loss, metrics = validation_step(state.params, state.pending_qb_betas, batch)
        losses.append(float(loss))
        dropped += int(
            np.asarray(metrics[MOE_DROPPED_ASSIGNMENTS_METRIC]).astype(np.int64).sum()
        )
        valid += int(
            np.asarray(metrics[MOE_VALID_ASSIGNMENTS_METRIC]).astype(np.int64).sum()
        )
    if not all(math.isfinite(x) for x in losses):
        raise ValueError("Nonfinite held-out loss")
    return {
        "eval/loss": float(np.mean(losses)),
        "eval/drop_fraction": dropped / valid,
        "eval/examples": VALIDATION_EXAMPLES,
    }


def _run_training(config: TrainingConfig) -> None:
    mp = jmp.get_policy("params=bfloat16,compute=bfloat16,output=bfloat16")
    trainer = TrainerConfig(
        id=config.run_id,
        seed=config.seed,
        train_batch_size=BATCH_SIZE,
        num_train_steps=TOTAL_UPDATES,
        require_accelerator=True,
        mp=mp,
        use_explicit_mesh_axes=True,
        mesh=grug_trainer_mesh_config(config.context_axis_size),
        watch=WatchConfig(interval=0, watch_targets=[]),
        tracker=WandbConfig(
            entity="eric-czech",
            project="marin",
            name=config.run_id,
            group="exp586-plantcad2-moe-sweep",
            tags=["dna-exp586", "H100", config.cluster, config.condition],
            save_code=False,
            background=False,
            replicate_path=config.checkpoint_root,
        ),
    )
    trainer.initialize()
    assert jax.device_count() == 8 * config.nodes
    assert all("H100" in d.device_kind for d in jax.devices())
    device_kind = jax.devices()[0].device_kind
    flops_per_example, flops_summary = _compute_flops(model_config=config.model)
    flops_per_device, flops_total = theoretical_flops(device_kind, jax.device_count())
    log_offset = wandb_history_offset()
    tracker.log_configuration(config)
    tracker.log_summary(
        {
            "training/phase": "loading",
            "training/moe_runtime_backend": config.model.moe_implementation,
            "gpu_type": "H100",
            "nodes": config.nodes,
            **flops_summary,
            "throughput/flops_per_example": flops_per_example,
            "throughput/device_kind": device_kind,
            "throughput/theoretical_flops_per_device": flops_per_device,
            "throughput/theoretical_flops": flops_total,
        }
    )
    tokenizer = (
        scratch_tokenizer()
        if config.condition == "scratch"
        else pretrained_dna_tokenizer()
    )
    tokenizer_sha256 = tokenizer_digest(tokenizer)
    binding = scientific_config(config, tokenizer_sha256)
    digest, checkpoint = primary_json(lambda: bind_and_discover(config, binding))
    tracker.log_hyperparameters(
        {
            **binding,
            "scientific_config_sha256": digest,
            "reference_real_tokens_per_update": TOKENS_PER_UPDATE,
        }
    )
    dataset, validation = open_dataset("train", config.seed), ValidationSample()
    opt_config = optimizer_config(
        reference_tokens_per_update=TOKENS_PER_UPDATE, multiplier=config.lr_multiplier
    )
    optimizer = opt_config.build(TOTAL_UPDATES)
    schedule = config.schedule
    mesh = training_mesh(config)
    with jax.set_mesh(mesh):
        weights = (
            restore_weights(
                SOURCE_CHECKPOINT, SOURCE_METADATA_DIGEST, config.model, mesh
            )
            if config.condition == "pretrained" and checkpoint is None
            else eqx.filter_jit(Transformer.init)(
                config.model, key=jax.random.key(config.seed)
            )
        )
        state = eqx.filter_jit(lambda w: fresh_state(w, optimizer, mp, offload=True))(
            weights
        )
        del weights
        clock = TokenClock()
        if checkpoint is not None:
            state, clock = restore_dna_checkpoint(
                state,
                checkpoint,
                mesh,
                data_seed=config.seed,
                tokenizer_sha256=tokenizer_sha256,
            )
            gc.collect()
        check_clock(clock)
        initial_update = clock.updates
        tracker.log_summary(
            {
                "training/resumed_checkpoint": checkpoint,
                "training/resumed_input_tokens": clock.input_tokens,
                "training/schedule_revision": schedule.revision,
                "training/target_input_tokens": TRAINING_TOKENS,
                "training/peak_update": schedule.peak_update,
                "training/phase": "compiling",
            }
        )
        if initial_update >= schedule.peak_update:
            tracker.log_summary(
                {
                    "training/peak_checkpoint": (
                        f"{config.checkpoint_root}/step-{schedule.peak_update}"
                    )
                }
            )
        if initial_update == schedule.peak_update:
            primary_json(lambda: commit_and_prune(config, checkpoint, digest, clock))
            tracker.log_summary({"training/peak_checkpoint": checkpoint})
            tracker.log(
                evaluate(state, validation, tokenizer, mesh), step=log_offset + 1
            )
            log_offset += 1
        train_step = _make_train_step(
            optimizer,
            mp,
            z_loss_weight=1e-4,
            ema_beta=None,
            offload_opt_state=True,
            master_param_mode=MasterParamMode.FP32_PINNED_HOST,
            watch_config=None,
        )
        last_save = time.monotonic()
        while clock.input_tokens < TRAINING_TOKENS:
            start = time.monotonic()
            batch = distributed_batch(
                dataset, tokenizer, mesh, clock.examples, BATCH_SIZE, augment=True
            )
            loading_time = time.monotonic() - start
            factor = schedule.factor(clock.input_tokens)
            state = dataclasses.replace(
                state,
                opt_state=set_token_learning_rates(
                    state.opt_state, opt_config, schedule, clock
                ),
            )
            state, metrics, watch = train_step(state, batch)
            jax.block_until_ready((state, metrics, watch))
            clock = clock.advance(
                input_tokens=TOKENS_PER_UPDATE,
                loss_targets=BATCH_SIZE * 8191,
                examples=BATCH_SIZE,
            )
            elapsed = time.monotonic() - start
            values = routing_diagnostics(
                metrics, batch_size=BATCH_SIZE, model_config=config.model
            )
            values.update({key: float(value) for key, value in (watch or {}).items()})
            if any(not math.isfinite(float(value)) for value in values.values()):
                tracker.log_summary(
                    {
                        "training/phase": "failed",
                        "training/error": "Nonfinite training metrics",
                    }
                )
                raise ValueError("Nonfinite loss, routing, gradient or update metrics")
            values.update(
                {
                    "run_progress": min(clock.input_tokens / TRAINING_TOKENS, 0.999999),
                    "train/step": clock.updates,
                    "train/input_tokens": clock.input_tokens,
                    "train/loss_targets": clock.loss_targets,
                    "train/examples": clock.examples,
                    "train/epochs": clock.input_tokens / EPOCH_TOKENS,
                    "train/schedule_revision": schedule.revision,
                    "train/muon_lr": opt_config.learning_rate * factor,
                    "train/adam_lr": opt_config.adam_lr * factor,
                    "throughput/step_seconds": elapsed,
                    "throughput/loading_seconds": loading_time,
                    "throughput/tokens_per_second": TOKENS_PER_UPDATE / elapsed,
                }
            )
            values.update(
                throughput_flop_metrics(
                    flops_per_example=flops_per_example,
                    completed_examples=clock.examples,
                    batch_size=BATCH_SIZE,
                    elapsed_seconds=elapsed,
                    theoretical_flops_total=flops_total,
                )
            )
            for key in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit"):
                memory = jax.local_devices()[0].memory_stats() or {}
                if key in memory:
                    values[f"memory/{key}"] = memory[key]
            history_step = log_offset + clock.updates - initial_update
            tracker.log(values, step=history_step)
            tracker.log_summary({"training/phase": "training"})
            permanent = is_permanent_checkpoint(
                clock.updates, peak_update=schedule.peak_update
            )
            due = int(
                multihost_utils.broadcast_one_to_all(
                    np.asarray(
                        checkpoint_due(
                            clock.updates,
                            initial_update,
                            time.monotonic() - last_save,
                            config.checkpoint_interval_seconds,
                            peak_update=schedule.peak_update,
                            recovery_delay_updates=config.recovery_checkpoint_delay_updates,
                        ),
                        dtype=np.int32,
                    )
                )
            )
            if due:
                save_start = time.monotonic()
                if not bool(state_all_finite(state)):
                    raise ValueError("Nonfinite checkpoint state")
                tracker.log_summary({"training/phase": "checkpointing"})
                checkpoint = primary_json(
                    lambda clock=clock: prepare_checkpoint(config, clock)
                )
                save_dna_checkpoint(
                    state,
                    clock,
                    checkpoint,
                    data_seed=config.seed,
                    tokenizer_sha256=tokenizer_sha256,
                    scientific_config_sha256=digest,
                    runtime_moe_implementation=config.model.moe_implementation,
                    is_temporary=not permanent,
                )
                primary_json(
                    lambda checkpoint=checkpoint, clock=clock: commit_and_prune(
                        config, checkpoint, digest, clock
                    )
                )
                last_save = time.monotonic()
                tracker.log_summary(
                    {
                        "training/latest_checkpoint": checkpoint,
                        "training/checkpoint_input_tokens": clock.input_tokens,
                        "training/checkpoint_step": clock.updates,
                        "training/checkpoint_seconds": last_save - save_start,
                    }
                )
                if permanent:
                    tracker.log_summary(
                        {"training/latest_permanent_checkpoint": checkpoint}
                    )
                if clock.updates == schedule.peak_update:
                    tracker.log_summary({"training/peak_checkpoint": checkpoint})
            if clock.updates == 25 or clock.updates % 2000 == 0 or permanent:
                tracker.log_summary({"training/phase": "evaluating"})
                tracker.log(
                    evaluate(state, validation, tokenizer, mesh), step=history_step
                )
        if initial_update == TOTAL_UPDATES:
            primary_json(lambda: commit_and_prune(config, checkpoint, digest, clock))
            tracker.log(
                evaluate(state, validation, tokenizer, mesh), step=log_offset + 1
            )
        tracker.log(
            {"run_progress": 1.0}, step=log_offset + clock.updates - initial_update + 2
        )
        tracker.log_summary(
            {"training/phase": "complete", "training/final_checkpoint": checkpoint}
        )
    tracker.current_tracker().finish()


def training_mesh(config: TrainingConfig) -> jax.sharding.Mesh:
    """Keep EP8 inside each node and replicate the fixed global batch."""
    mesh = compact_grug_mesh(
        expert_axis_size=8,
        replica_axis_size=config.nodes // config.context_axis_size,
        context_axis_size=config.context_axis_size,
    )
    for devices in mesh.devices.reshape(config.nodes, 8):
        if len({d.process_index // 8 for d in devices}) != 1:
            raise ValueError("EP group crosses Iris node boundaries")
    return mesh


def run_training(config: TrainingConfig) -> None:
    try:
        _run_training(config)
    except Exception as error:
        if jax.process_index() == 0:
            try:
                tracker.log_summary(
                    {
                        "training/phase": "failed",
                        "training/error": f"{type(error).__name__}: {error}",
                    }
                )
                tracker.get_tracker("wandb").run.finish(exit_code=1)
            except Exception:
                logging.getLogger(__name__).exception(
                    "Could not publish the training failure to W&B"
                )
        raise


def dispatch(config: TrainingConfig) -> None:
    _apply_hero_ep_runtime_defaults(
        inline_watch_enabled=False,
        moe_implementation=config.model.moe_implementation,
        remat_mode="recompute_all",
        processes_per_task=8,
    )
    resources = ResourceConfig.with_gpu(
        "H100", count=8, replicas=config.nodes, cpu=32, ram="600g", disk="900g"
    )
    env = resolve_training_env(
        {
            **_forwarded_env_vars(),
            "MARIN_PREFIX": "s3://marin-us-east-02a/marin",
            "WANDB_API_KEY": os.environ["WANDB_API_KEY"],
            "WANDB_ENTITY": "eric-czech",
            "WANDB_PROJECT": "marin",
            "EXP586_CHECKPOINT_DEBUG": os.environ.get("EXP586_CHECKPOINT_DEBUG", "0"),
        },
        resources,
    )
    job = current_client().submit(
        JobRequest(
            name=f"train-{config.run_id}",
            entrypoint=Entrypoint.from_callable(run_training, args=[config]),
            resources=resources,
            processes_per_task=8,
            environment=create_environment(env_vars=env, extras=["gpu"]),
            priority=priority_band_value("batch"),
            timeout=Duration.from_hours(14 * 24),
            max_retries_failure=0,
            max_task_failures=0,
        )
    )
    job.wait(raise_on_failure=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=["scratch", "pretrained"], required=True)
    parser.add_argument(
        "--lr-multiplier",
        type=float,
        choices=[0.5, 1.0, 2.0, 3.0, 10.0],
        required=True,
    )
    parser.add_argument(
        "--cluster", choices=["cw-rno2a", "cw-us-east-02a"], required=True
    )
    parser.add_argument("--nodes", type=int, choices=[1, 2, 4, 8], required=True)
    parser.add_argument("--checkpoint-interval-seconds", type=int, default=900)
    parser.add_argument("--recovery-checkpoint-delay-updates", type=int, default=25)
    parser.add_argument(
        "--moe-backend", choices=["pooled", "fixed", "ring"], default=None
    )
    dispatch(TrainingConfig(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
