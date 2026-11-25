"""Prove that the combined runtime preserves exp586's scientific state."""

import argparse
import dataclasses
import hashlib
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
from fray.cluster import ResourceConfig
from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, create_environment
from iris.rpc.proto_display import priority_band_value
from levanter import tracker
from levanter.callbacks.watch import WatchConfig
from levanter.grug.grug_moe import (
    MOE_DROPPED_ASSIGNMENTS_METRIC,
    MOE_RECEIVER_DROPPED_ASSIGNMENTS_METRIC,
    MOE_SENDER_DROPPED_ASSIGNMENTS_METRIC,
    MOE_SKIPPED_PADDING_ASSIGNMENTS_METRIC,
    MOE_VALID_ASSIGNMENTS_METRIC,
)
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.training.training import resolve_training_env
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration

from exp586_moe.config import GLOBAL_BATCH_SIZE, SEQ_LEN, TOKENS_PER_UPDATE
from exp586_moe.corpus import distributed_batch, open_dataset
from exp586_moe.data import pretrained_dna_tokenizer, tokenizer_digest
from exp586_moe.schedule import optimizer_config, set_token_learning_rates
from exp586_moe.state import (
    fresh_state,
    restore_dna_checkpoint,
    save_dna_checkpoint,
    state_error_by_group,
    state_max_abs_error,
    validate_restored_training_state,
)
from exp586_moe.train import (
    TrainingConfig,
    config_json,
    scientific_config,
    training_mesh,
)
from experiments.grug.dispatch import _forwarded_env_vars
from experiments.grug.moe_hero_ep.model import Transformer
from experiments.grug.moe_hero_ep.train import (
    MasterParamMode,
    _apply_hero_ep_runtime_defaults,
    _drop_metrics,
    _make_train_step,
    grug_trainer_mesh_config,
)

SOURCE_CHECKPOINT = (
    "s3://marin-us-east-02a/MarinDNA/exp586_plantcad2_moe/checkpoints/"
    "exp586-plantcad2-d768-pretrained-lrm1-seed0-1t-v1/2026.10.20/"
    "checkpoints/step-693582"
)
SOURCE_METADATA_SHA256 = (
    "ad4827f1160660c264712a99cfea4e6a9df9f699e3bc4d405a745bf86d2ddfe5"
)
SOURCE_SCIENTIFIC_CONFIG_SHA256 = (
    "6a5273b56f94f5da995a882b44208678dbbb8baf8871a4c3ccc37c286f222899"
)


@dataclass(frozen=True)
class EquivalenceConfig:
    cluster: str
    validation_id: str
    updates: int = 100
    nodes: int = 8

    def __post_init__(self) -> None:
        if self.cluster not in ("cw-rno2a", "cw-us-east-02a"):
            raise ValueError("Equivalence validation requires an approved H100 cluster")
        if self.nodes != 8:
            raise ValueError("Equivalence must use the production eight-node topology")
        if not 10 <= self.updates <= 200:
            raise ValueError("Equivalence validation requires 10 through 200 updates")
        if not self.validation_id or not self.validation_id.replace("-", "").isalnum():
            raise ValueError(
                "Validation ID must contain only letters, digits, and hyphens"
            )

    @property
    def run_id(self) -> str:
        return (
            "exp586-plantcad2-d768-pretrained-lrm1-seed0-1t-v1-"
            f"equivalence-{self.validation_id}"
        )

    @property
    def checkpoint_root(self) -> str:
        return (
            "s3://marin-us-east-02a/tmp/ttl=7d/MarinDNA/"
            f"exp586_plantcad2_moe/{self.run_id}/checkpoints"
        )


@eqx.filter_jit
def _tree_max_abs_error(left: Any, right: Any) -> jax.Array:
    pairs = [
        (a, b)
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
        if eqx.is_array(a) and eqx.is_array(b)
    ]
    if not pairs:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return jnp.max(
        jnp.stack(
            [
                jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32)))
                for a, b in pairs
            ]
        )
    )


def _memory_kinds(tree: Any) -> set[str]:
    return {
        str(getattr(leaf.sharding, "memory_kind", None))
        for leaf in jax.tree.leaves(tree)
        if isinstance(leaf, jax.Array)
    }


def _memory_kind_bytes(tree: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for leaf in jax.tree.leaves(tree):
        if not isinstance(leaf, jax.Array):
            continue
        kind = str(getattr(leaf.sharding, "memory_kind", None))
        result[kind] = result.get(kind, 0) + leaf.size * leaf.dtype.itemsize
    return result


def _assert_same_value_layout(left: Any, right: Any) -> None:
    left_leaves = [x for x in jax.tree.leaves(left) if isinstance(x, jax.Array)]
    right_leaves = [x for x in jax.tree.leaves(right) if isinstance(x, jax.Array)]
    if len(left_leaves) != len(right_leaves):
        raise ValueError("Runtime modes have different state structures")
    for a, b in zip(left_leaves, right_leaves, strict=True):
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError("Runtime modes have different state shapes or dtypes")
        a_sharding = a.sharding
        b_sharding = b.sharding
        if getattr(a_sharding, "spec", None) != getattr(b_sharding, "spec", None):
            raise ValueError("Runtime modes have different partitioning")


def _prefetched_batches(
    dataset: Any,
    tokenizer: Any,
    mesh: jax.sharding.Mesh,
    start: int,
    batch_size: int,
    count: int,
):
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            distributed_batch,
            dataset,
            tokenizer,
            mesh,
            start,
            batch_size,
            augment=True,
        )
        for index in range(count):
            batch = future.result()
            if index + 1 < count:
                future = pool.submit(
                    distributed_batch,
                    dataset,
                    tokenizer,
                    mesh,
                    start + (index + 1) * batch_size,
                    batch_size,
                    augment=True,
                )
            yield batch


def _errors(left: Any, right: Any) -> dict[str, float]:
    return {
        key: float(value) for key, value in state_error_by_group(left, right).items()
    }


def _max_error(left: Any, right: Any) -> float:
    return float(state_max_abs_error(left, right))


def _drop_fraction(metrics: dict[str, Any], training: TrainingConfig) -> float:
    return float(
        _drop_metrics(
            metrics[MOE_DROPPED_ASSIGNMENTS_METRIC],
            metrics[MOE_SENDER_DROPPED_ASSIGNMENTS_METRIC],
            metrics[MOE_RECEIVER_DROPPED_ASSIGNMENTS_METRIC],
            metrics[MOE_SKIPPED_PADDING_ASSIGNMENTS_METRIC],
            metrics[MOE_VALID_ASSIGNMENTS_METRIC],
            batch_size=GLOBAL_BATCH_SIZE,
            sequence_length=SEQ_LEN,
            top_k=training.model.num_experts_per_token,
            num_layers=training.model.num_layers,
        )["moe/drop_fraction"]
    )


def run_validation(config: EquivalenceConfig) -> None:
    training = TrainingConfig(
        "pretrained", 1.0, config.cluster, config.nodes, stage="t1"
    )
    mp = jmp.get_policy("params=bfloat16,compute=bfloat16,output=bfloat16")
    trainer = TrainerConfig(
        id=config.run_id,
        seed=0,
        train_batch_size=GLOBAL_BATCH_SIZE,
        num_train_steps=training.target_updates,
        require_accelerator=True,
        mp=mp,
        use_explicit_mesh_axes=True,
        mesh=grug_trainer_mesh_config(training.context_axis_size),
        watch=WatchConfig(interval=0, watch_targets=[]),
        tracker=WandbConfig(
            entity="eric-czech",
            project="marin",
            name=config.run_id,
            group="exp586-plantcad2-moe-validation",
            tags=[
                "dna-exp586",
                "H100",
                config.cluster,
                "scientific-equivalence",
            ],
            save_code=False,
            background=False,
            replicate_path=config.checkpoint_root,
        ),
    )
    trainer.initialize()
    if jax.device_count() != 64 or any(
        "H100" not in device.device_kind for device in jax.devices()
    ):
        raise ValueError("Equivalence validation requires exactly 64 H100s")
    tracker.log_configuration(config)
    tracker.log_summary(
        {
            "validation/phase": "restoring",
            "validation/source_checkpoint": SOURCE_CHECKPOINT,
            "validation/control_state_mode": "fp32_pinned_host",
            "validation/candidate_state_mode": "fp32_device",
            "validation/prefetch_depth": 1,
            "validation/nodes": config.nodes,
        }
    )

    sync_tokenizer = pretrained_dna_tokenizer()
    prefetch_tokenizer = pretrained_dna_tokenizer()
    tokenizer_sha256 = tokenizer_digest(sync_tokenizer)
    if tokenizer_digest(prefetch_tokenizer) != tokenizer_sha256:
        raise ValueError("Synchronous and prefetched tokenizers differ")
    digest = hashlib.sha256(
        config_json(scientific_config(training, tokenizer_sha256)).encode()
    ).hexdigest()
    if digest != SOURCE_SCIENTIFIC_CONFIG_SHA256:
        raise ValueError("Validation scientific configuration differs from production")

    opt_config = optimizer_config(
        reference_tokens_per_update=TOKENS_PER_UPDATE,
        training_tokens=training.optimizer_reference_tokens,
        multiplier=1.0,
    )
    optimizer = opt_config.build(training.target_updates)
    mesh = training_mesh(training)
    with jax.set_mesh(mesh):
        weights = eqx.filter_jit(Transformer.init)(
            training.model, key=jax.random.key(0)
        )
        pinned_exemplar = eqx.filter_jit(
            lambda value: fresh_state(
                value, optimizer, mp, offload=True, retain_master=True
            )
        )(weights)
        device_exemplar = eqx.filter_jit(
            lambda value: fresh_state(
                value, optimizer, mp, offload=False, retain_master=True
            )
        )(weights)
        del weights

        control_a, source_clock = restore_dna_checkpoint(
            pinned_exemplar,
            SOURCE_CHECKPOINT,
            mesh,
            data_seed=0,
            tokenizer_sha256=tokenizer_sha256,
            scientific_config_sha256=digest,
            metadata_sha256=SOURCE_METADATA_SHA256,
        )
        control_b, control_b_clock = restore_dna_checkpoint(
            pinned_exemplar,
            SOURCE_CHECKPOINT,
            mesh,
            data_seed=0,
            tokenizer_sha256=tokenizer_sha256,
            scientific_config_sha256=digest,
            metadata_sha256=SOURCE_METADATA_SHA256,
        )
        candidate, candidate_clock = restore_dna_checkpoint(
            device_exemplar,
            SOURCE_CHECKPOINT,
            mesh,
            data_seed=0,
            tokenizer_sha256=tokenizer_sha256,
            scientific_config_sha256=digest,
            metadata_sha256=SOURCE_METADATA_SHA256,
        )
        if source_clock != control_b_clock or source_clock != candidate_clock:
            raise ValueError("Runtime modes restored different token/data clocks")
        for state in (control_a, control_b, candidate):
            validate_restored_training_state(state, source_clock)
        _assert_same_value_layout(control_a, candidate)
        pinned_master_kinds = _memory_kinds(control_a.master_params)
        pinned_optimizer_kinds = _memory_kinds(control_a.opt_state)
        device_master_kinds = _memory_kinds(candidate.master_params)
        device_optimizer_kinds = _memory_kinds(candidate.opt_state)
        pinned_master_bytes = _memory_kind_bytes(control_a.master_params)
        pinned_optimizer_bytes = _memory_kind_bytes(control_a.opt_state)
        device_master_bytes = _memory_kind_bytes(candidate.master_params)
        device_optimizer_bytes = _memory_kind_bytes(candidate.opt_state)
        tracker.log_summary(
            {
                "validation/control_master_memory_kinds": sorted(pinned_master_kinds),
                "validation/control_optimizer_memory_kinds": sorted(
                    pinned_optimizer_kinds
                ),
                "validation/candidate_master_memory_kinds": sorted(device_master_kinds),
                "validation/candidate_optimizer_memory_kinds": sorted(
                    device_optimizer_kinds
                ),
                "validation/control_master_bytes_by_kind": pinned_master_bytes,
                "validation/control_optimizer_bytes_by_kind": pinned_optimizer_bytes,
                "validation/candidate_master_bytes_by_kind": device_master_bytes,
                "validation/candidate_optimizer_bytes_by_kind": device_optimizer_bytes,
            }
        )
        if (
            pinned_master_bytes.get("pinned_host", 0) == 0
            or pinned_optimizer_bytes.get("pinned_host", 0) == 0
        ):
            raise ValueError("Control bulk state is not pinned-host resident")
        if (
            device_master_bytes.get("pinned_host", 0) != 0
            or device_optimizer_bytes.get("pinned_host", 0) != 0
        ):
            raise ValueError("Candidate retains pinned-host training state")
        initial_control_error = _max_error(control_a, control_b)
        initial_candidate_error = _max_error(control_a, candidate)
        if initial_control_error != 0.0 or initial_candidate_error != 0.0:
            raise ValueError("Runtime modes did not restore identical values")

        pinned_step = _make_train_step(
            optimizer,
            mp,
            z_loss_weight=1e-4,
            ema_beta=None,
            offload_opt_state=True,
            master_param_mode=MasterParamMode.FP32_PINNED_HOST,
            watch_config=None,
        )
        device_step = _make_train_step(
            optimizer,
            mp,
            z_loss_weight=1e-4,
            ema_beta=None,
            offload_opt_state=False,
            master_param_mode=MasterParamMode.FP32_DEVICE,
            watch_config=None,
        )
        sync_dataset = open_dataset("train", 0)
        prefetch_dataset = open_dataset("train", 0)
        prefetched = iter(
            _prefetched_batches(
                prefetch_dataset,
                prefetch_tokenizer,
                mesh,
                source_clock.examples,
                GLOBAL_BATCH_SIZE,
                config.updates + 1,
            )
        )
        clock = source_clock
        milestone_updates = {1, 2, 10, 25, 50, config.updates}
        max_batch_error = 0.0
        max_control_metric_error = 0.0
        max_candidate_metric_error = 0.0
        state_results: dict[str, dict[str, float]] = {}
        control_seconds = 0.0
        candidate_seconds = 0.0
        control_loss_squared_error = 0.0
        candidate_loss_squared_error = 0.0
        max_control_loss_error = 0.0
        max_candidate_loss_error = 0.0
        max_control_drop_error = 0.0
        max_candidate_drop_error = 0.0

        tracker.log_summary({"validation/phase": "updating"})
        for relative_update in range(1, config.updates + 1):
            sync_batch = distributed_batch(
                sync_dataset,
                sync_tokenizer,
                mesh,
                clock.examples,
                GLOBAL_BATCH_SIZE,
                augment=True,
            )
            candidate_batch = next(prefetched)
            batch_error = float(_tree_max_abs_error(sync_batch, candidate_batch))
            max_batch_error = max(max_batch_error, batch_error)
            if batch_error != 0.0:
                raise ValueError(
                    f"Prefetch changed the training batch at update {relative_update}"
                )

            control_a = dataclasses.replace(
                control_a,
                opt_state=set_token_learning_rates(
                    control_a.opt_state, opt_config, training.schedule, clock
                ),
            )
            control_b = dataclasses.replace(
                control_b,
                opt_state=set_token_learning_rates(
                    control_b.opt_state, opt_config, training.schedule, clock
                ),
            )
            candidate = dataclasses.replace(
                candidate,
                opt_state=set_token_learning_rates(
                    candidate.opt_state, opt_config, training.schedule, clock
                ),
            )

            started = time.monotonic()
            control_a, control_a_metrics, _ = pinned_step(control_a, sync_batch)
            jax.block_until_ready((control_a, control_a_metrics))
            control_seconds += time.monotonic() - started
            control_b, control_b_metrics, _ = pinned_step(control_b, sync_batch)
            jax.block_until_ready((control_b, control_b_metrics))
            started = time.monotonic()
            candidate, candidate_metrics, _ = device_step(candidate, candidate_batch)
            jax.block_until_ready((candidate, candidate_metrics))
            candidate_seconds += time.monotonic() - started

            control_metric_error = float(
                _tree_max_abs_error(control_a_metrics, control_b_metrics)
            )
            candidate_metric_error = float(
                _tree_max_abs_error(control_a_metrics, candidate_metrics)
            )
            max_control_metric_error = max(
                max_control_metric_error, control_metric_error
            )
            max_candidate_metric_error = max(
                max_candidate_metric_error, candidate_metric_error
            )
            control_a_loss = float(control_a_metrics["train/loss"])
            control_b_loss = float(control_b_metrics["train/loss"])
            candidate_loss = float(candidate_metrics["train/loss"])
            control_loss_error = abs(control_a_loss - control_b_loss)
            candidate_loss_error = abs(control_a_loss - candidate_loss)
            control_loss_squared_error += control_loss_error**2
            candidate_loss_squared_error += candidate_loss_error**2
            max_control_loss_error = max(max_control_loss_error, control_loss_error)
            max_candidate_loss_error = max(
                max_candidate_loss_error, candidate_loss_error
            )
            control_a_drop = _drop_fraction(control_a_metrics, training)
            control_b_drop = _drop_fraction(control_b_metrics, training)
            candidate_drop = _drop_fraction(candidate_metrics, training)
            control_drop_error = abs(control_a_drop - control_b_drop)
            candidate_drop_error = abs(control_a_drop - candidate_drop)
            max_control_drop_error = max(max_control_drop_error, control_drop_error)
            max_candidate_drop_error = max(
                max_candidate_drop_error, candidate_drop_error
            )
            clock = clock.advance(
                input_tokens=TOKENS_PER_UPDATE,
                loss_targets=GLOBAL_BATCH_SIZE * (SEQ_LEN - 1),
                examples=GLOBAL_BATCH_SIZE,
            )
            if relative_update in milestone_updates:
                control_error = _max_error(control_a, control_b)
                candidate_error = _max_error(control_a, candidate)
                state_results[str(relative_update)] = {
                    "control_max_abs": control_error,
                    "candidate_max_abs": candidate_error,
                    "control_metric_max_abs": control_metric_error,
                    "candidate_metric_max_abs": candidate_metric_error,
                    "control_loss_abs": control_loss_error,
                    "candidate_loss_abs": candidate_loss_error,
                    "control_drop_abs": control_drop_error,
                    "candidate_drop_abs": candidate_drop_error,
                }
                tracker.log(
                    {
                        "validation/relative_update": relative_update,
                        "validation/control_state_max_abs": control_error,
                        "validation/candidate_state_max_abs": candidate_error,
                        "validation/control_metric_max_abs": control_metric_error,
                        "validation/candidate_metric_max_abs": candidate_metric_error,
                        "validation/control_loss_abs": control_loss_error,
                        "validation/candidate_loss_abs": candidate_loss_error,
                        "validation/control_drop_abs": control_drop_error,
                        "validation/candidate_drop_abs": candidate_drop_error,
                        "validation/batch_max_abs": batch_error,
                    },
                    step=relative_update,
                )

        control_path = f"{config.checkpoint_root}/control/step-{clock.updates}"
        candidate_path = f"{config.checkpoint_root}/candidate/step-{clock.updates}"
        if StoragePath(control_path).exists() or StoragePath(candidate_path).exists():
            raise ValueError("Isolated validation checkpoint already exists")
        tracker.log_summary({"validation/phase": "checkpointing"})
        save_dna_checkpoint(
            control_a,
            clock,
            control_path,
            data_seed=0,
            tokenizer_sha256=tokenizer_sha256,
            scientific_config_sha256=digest,
            runtime_moe_implementation=training.model.moe_implementation,
            is_temporary=True,
        )
        save_dna_checkpoint(
            candidate,
            clock,
            candidate_path,
            data_seed=0,
            tokenizer_sha256=tokenizer_sha256,
            scientific_config_sha256=digest,
            runtime_moe_implementation=training.model.moe_implementation,
            is_temporary=True,
        )
        candidate_as_pinned, candidate_as_pinned_clock = restore_dna_checkpoint(
            pinned_exemplar,
            candidate_path,
            mesh,
            data_seed=0,
            tokenizer_sha256=tokenizer_sha256,
            scientific_config_sha256=digest,
        )
        control_as_device, control_as_device_clock = restore_dna_checkpoint(
            device_exemplar,
            control_path,
            mesh,
            data_seed=0,
            tokenizer_sha256=tokenizer_sha256,
            scientific_config_sha256=digest,
        )
        if candidate_as_pinned_clock != clock or control_as_device_clock != clock:
            raise ValueError("Cross-mode checkpoint restore changed the clock")
        candidate_roundtrip_error = _max_error(candidate, candidate_as_pinned)
        control_roundtrip_error = _max_error(control_a, control_as_device)
        if candidate_roundtrip_error != 0.0 or control_roundtrip_error != 0.0:
            raise ValueError("Cross-mode checkpoint restore changed state values")

        # The iterator has already queued this batch. Recompute it from the restored
        # clock exactly as a restarted process would, then execute one post-restore step.
        outstanding_batch = next(prefetched)
        resumed_batch = distributed_batch(
            open_dataset("train", 0),
            pretrained_dna_tokenizer(),
            mesh,
            clock.examples,
            GLOBAL_BATCH_SIZE,
            augment=True,
        )
        restart_batch_error = float(
            _tree_max_abs_error(outstanding_batch, resumed_batch)
        )
        if restart_batch_error != 0.0:
            raise ValueError("Restart changed the next prefetched batch")
        candidate_as_pinned = dataclasses.replace(
            candidate_as_pinned,
            opt_state=set_token_learning_rates(
                candidate_as_pinned.opt_state,
                opt_config,
                training.schedule,
                clock,
            ),
        )
        control_as_device = dataclasses.replace(
            control_as_device,
            opt_state=set_token_learning_rates(
                control_as_device.opt_state,
                opt_config,
                training.schedule,
                clock,
            ),
        )
        candidate_as_pinned, pinned_restart_metrics, _ = pinned_step(
            candidate_as_pinned, resumed_batch
        )
        control_as_device, device_restart_metrics, _ = device_step(
            control_as_device, outstanding_batch
        )
        jax.block_until_ready(
            (
                candidate_as_pinned,
                control_as_device,
                pinned_restart_metrics,
                device_restart_metrics,
            )
        )
        restart_state_error = _max_error(candidate_as_pinned, control_as_device)
        restart_metric_error = float(
            _tree_max_abs_error(pinned_restart_metrics, device_restart_metrics)
        )

        final_control_errors = _errors(control_a, control_b)
        final_candidate_errors = _errors(control_a, candidate)
        control_loss_rmse = math.sqrt(control_loss_squared_error / config.updates)
        candidate_loss_rmse = math.sqrt(candidate_loss_squared_error / config.updates)
        tracker.log_summary(
            {
                "validation/phase": "complete",
                "validation/scientific_config_sha256": digest,
                "validation/source_step": source_clock.updates,
                "validation/final_step": clock.updates,
                "validation/max_batch_error": max_batch_error,
                "validation/max_control_metric_error": max_control_metric_error,
                "validation/max_candidate_metric_error": max_candidate_metric_error,
                "validation/control_loss_rmse": control_loss_rmse,
                "validation/candidate_loss_rmse": candidate_loss_rmse,
                "validation/max_control_loss_error": max_control_loss_error,
                "validation/max_candidate_loss_error": max_candidate_loss_error,
                "validation/max_control_drop_error": max_control_drop_error,
                "validation/max_candidate_drop_error": max_candidate_drop_error,
                "validation/control_roundtrip_error": control_roundtrip_error,
                "validation/candidate_roundtrip_error": candidate_roundtrip_error,
                "validation/restart_batch_error": restart_batch_error,
                "validation/restart_state_error": restart_state_error,
                "validation/restart_metric_error": restart_metric_error,
                "validation/control_mean_step_seconds": control_seconds
                / config.updates,
                "validation/candidate_mean_step_seconds": candidate_seconds
                / config.updates,
                "validation/milestones": state_results,
                **{
                    f"validation/final_control_{key}": value
                    for key, value in final_control_errors.items()
                },
                **{
                    f"validation/final_candidate_{key}": value
                    for key, value in final_candidate_errors.items()
                },
            }
        )
        tracker.log({"run_progress": 1.0}, step=config.updates + 1)
    tracker.current_tracker().finish()


def dispatch(config: EquivalenceConfig) -> None:
    _apply_hero_ep_runtime_defaults(
        inline_watch_enabled=False,
        moe_implementation="fixed_pooled_wave_all_to_all",
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
            "EXP586_CHECKPOINT_DEBUG": "1",
        },
        resources,
    )
    job = current_client().submit(
        JobRequest(
            name=f"validate-{config.run_id}",
            entrypoint=Entrypoint.from_callable(run_validation, args=[config]),
            resources=resources,
            processes_per_task=8,
            environment=create_environment(env_vars=env, extras=["gpu"]),
            priority=priority_band_value("batch"),
            timeout=Duration.from_hours(4),
            max_retries_failure=0,
            max_task_failures=0,
        )
    )
    job.wait(raise_on_failure=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cluster", choices=["cw-rno2a", "cw-us-east-02a"], required=True
    )
    parser.add_argument("--nodes", type=int, choices=[8], default=8)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--validation-id", required=True)
    dispatch(EquivalenceConfig(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
