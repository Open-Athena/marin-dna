"""Validate the second exp586 runtime transition on the production topology."""

import argparse
import dataclasses
import gc
import hashlib
import json
import math
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import equinox as eqx
import jax
import jmp
from fray.cluster import ResourceConfig
from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, create_environment
from iris.rpc.proto_display import priority_band_value
from levanter import tracker
from levanter.callbacks.watch import WatchConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.training.training import resolve_training_env
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration

from exp586_moe.config import GLOBAL_BATCH_SIZE, SEQ_LEN, TOKENS_PER_UPDATE
from exp586_moe.corpus import distributed_batch, open_dataset
from exp586_moe.data import pretrained_dna_tokenizer, tokenizer_digest
from exp586_moe.routing_probe import routing_diagnostics
from exp586_moe.schedule import optimizer_config, set_token_learning_rates
from exp586_moe.state import (
    fresh_state,
    restore_dna_checkpoint,
    save_dna_checkpoint,
    state_all_finite,
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
    _make_train_step,
    grug_trainer_mesh_config,
)

SCIENTIFIC_CONFIG_SHA256 = (
    "6a5273b56f94f5da995a882b44208678dbbb8baf8871a4c3ccc37c286f222899"
)


@dataclass(frozen=True)
class ValidationConfig:
    cluster: str
    validation_id: str
    source_checkpoint: str
    source_metadata_sha256: str
    runtime_code_sha256: str
    updates: int = 25
    nodes: int = 8

    def __post_init__(self) -> None:
        if self.cluster not in ("cw-rno2a", "cw-us-east-02a"):
            raise ValueError("Validation requires an approved H100 cluster")
        if self.nodes != 8:
            raise ValueError("Validation requires the production eight-node topology")
        if not 10 <= self.updates <= 50:
            raise ValueError("Validation requires 10 through 50 updates")
        if not self.validation_id.replace("-", "").isalnum():
            raise ValueError("Validation ID must contain letters, digits, and hyphens")
        for value in (self.source_metadata_sha256, self.runtime_code_sha256):
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Validation digests must be lowercase SHA-256")
        if "/checkpoints/step-" not in self.source_checkpoint:
            raise ValueError("Validation source must be an exp586 training checkpoint")

    @property
    def run_id(self) -> str:
        return (
            "exp586-plantcad2-d768-pretrained-lrm1-seed0-1t-v1-"
            f"perf2-fullbatch-{self.validation_id}"
        )

    @property
    def artifact_root(self) -> str:
        return (
            "s3://marin-us-east-02a/tmp/ttl=7d/MarinDNA/"
            f"exp586_plantcad2_moe/{self.run_id}"
        )


def _prefetched_batches(dataset, tokenizer, mesh, start: int, count: int):
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            distributed_batch,
            dataset,
            tokenizer,
            mesh,
            start,
            GLOBAL_BATCH_SIZE,
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
                    start + (index + 1) * GLOBAL_BATCH_SIZE,
                    GLOBAL_BATCH_SIZE,
                    augment=True,
                )
            yield batch


def _runtime_training(config: ValidationConfig) -> TrainingConfig:
    return TrainingConfig(
        "pretrained",
        1.0,
        config.cluster,
        config.nodes,
        stage="t1",
        runtime_state_mode="fp32_device",
        runtime_prefetch=True,
        runtime_transition_id="embedding-expert-kernels-v2-validation",
        runtime_code_sha256=config.runtime_code_sha256,
        runtime_kernel_optimizations=True,
    )


def _new_exemplar(training, optimizer, mp):
    weights = eqx.filter_jit(Transformer.init)(
        training.model, key=jax.random.key(training.seed)
    )
    state = eqx.filter_jit(
        lambda value: fresh_state(
            value, optimizer, mp, offload=False, retain_master=True
        )
    )(weights)
    del weights
    return state


def run_validation(config: ValidationConfig) -> None:
    training = _runtime_training(config)
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
            tags=["dna-exp586", "H100", config.cluster, "perf2-fullbatch"],
            save_code=False,
            background=False,
            replicate_path=config.artifact_root,
        ),
    )
    trainer.initialize()
    if jax.device_count() != 64 or any(
        "H100" not in device.device_kind for device in jax.devices()
    ):
        raise ValueError("Validation requires exactly 64 H100s")

    tokenizer = pretrained_dna_tokenizer()
    tokenizer_sha256 = tokenizer_digest(tokenizer)
    digest = hashlib.sha256(
        config_json(scientific_config(training, tokenizer_sha256)).encode()
    ).hexdigest()
    if digest != SCIENTIFIC_CONFIG_SHA256:
        raise ValueError("Validation scientific configuration differs from production")

    tracker.log_configuration(config)
    tracker.log_summary(
        {
            "validation/phase": "restoring",
            "validation/source_checkpoint": config.source_checkpoint,
            "validation/scientific_config_sha256": digest,
            "validation/runtime_metadata": training.runtime_metadata,
            "validation/model_config": dataclasses.asdict(training.model),
        }
    )
    opt_config = optimizer_config(
        reference_tokens_per_update=TOKENS_PER_UPDATE,
        training_tokens=training.optimizer_reference_tokens,
        multiplier=1.0,
    )
    optimizer = opt_config.build(training.target_updates)
    mesh = training_mesh(training)
    with jax.set_mesh(mesh):
        state = _new_exemplar(training, optimizer, mp)
        state, clock = restore_dna_checkpoint(
            state,
            config.source_checkpoint,
            mesh,
            data_seed=0,
            tokenizer_sha256=tokenizer_sha256,
            scientific_config_sha256=digest,
            metadata_sha256=config.source_metadata_sha256,
        )
        source_summary = validate_restored_training_state(state, clock)
        if not bool(state_all_finite(state)):
            raise ValueError("Restored validation source state is nonfinite")

        train_step = _make_train_step(
            optimizer,
            mp,
            z_loss_weight=1e-4,
            ema_beta=None,
            offload_opt_state=False,
            master_param_mode=MasterParamMode.FP32_DEVICE,
            watch_config=None,
        )
        batches = iter(
            _prefetched_batches(
                open_dataset("train", 0),
                tokenizer,
                mesh,
                clock.examples,
                config.updates + 1,
            )
        )
        step_seconds = []
        losses = []
        tracker.log_summary(
            {
                "validation/phase": "updating",
                "validation/source_step": clock.updates,
                "validation/source_state": source_summary,
            }
        )
        for relative_update in range(1, config.updates + 1):
            batch = next(batches)
            factor = training.schedule.factor(clock.input_tokens)
            state = dataclasses.replace(
                state,
                opt_state=set_token_learning_rates(
                    state.opt_state, opt_config, training.schedule, clock
                ),
            )
            started = time.monotonic()
            state, metrics, _ = train_step(state, batch)
            jax.block_until_ready((state, metrics))
            elapsed = time.monotonic() - started
            values = routing_diagnostics(
                metrics,
                batch_size=GLOBAL_BATCH_SIZE,
                model_config=training.model,
            )
            if any(not math.isfinite(float(value)) for value in values.values()):
                raise ValueError("Validation produced nonfinite metrics")
            if any(
                float(values[key]) != 0.0
                for key in (
                    "moe/drop_fraction",
                    "moe/sender_drop_fraction",
                    "moe/receiver_drop_fraction",
                )
            ):
                raise ValueError("Validation produced MoE capacity drops")
            clock = clock.advance(
                input_tokens=TOKENS_PER_UPDATE,
                loss_targets=GLOBAL_BATCH_SIZE * (SEQ_LEN - 1),
                examples=GLOBAL_BATCH_SIZE,
            )
            loss = float(values["train/loss"])
            step_seconds.append(elapsed)
            losses.append(loss)
            tracker.log(
                {
                    "validation/relative_update": relative_update,
                    "validation/train_step": clock.updates,
                    "validation/input_tokens": clock.input_tokens,
                    "validation/loss": loss,
                    "validation/muon_lr": opt_config.learning_rate * factor,
                    "validation/adam_lr": opt_config.adam_lr * factor,
                    "validation/step_seconds": elapsed,
                    "validation/tokens_per_second": TOKENS_PER_UPDATE / elapsed,
                    **values,
                },
                step=relative_update,
            )

        checkpoint = f"{config.artifact_root}/checkpoints/step-{clock.updates}"
        if StoragePath(checkpoint).exists():
            raise ValueError("Validation checkpoint already exists")
        tracker.log_summary({"validation/phase": "checkpointing"})
        save_dna_checkpoint(
            state,
            clock,
            checkpoint,
            data_seed=0,
            tokenizer_sha256=tokenizer_sha256,
            scientific_config_sha256=digest,
            runtime_moe_implementation=training.model.moe_implementation,
            runtime_metadata=training.runtime_metadata,
            is_temporary=True,
        )
        del state
        gc.collect()

        replay = _new_exemplar(training, optimizer, mp)
        replay, replay_clock = restore_dna_checkpoint(
            replay,
            checkpoint,
            mesh,
            data_seed=0,
            tokenizer_sha256=tokenizer_sha256,
            scientific_config_sha256=digest,
        )
        replay_summary = validate_restored_training_state(replay, replay_clock)
        if replay_clock != clock or not bool(state_all_finite(replay)):
            raise ValueError("Validation checkpoint replay changed the clock or state")
        replay = dataclasses.replace(
            replay,
            opt_state=set_token_learning_rates(
                replay.opt_state, opt_config, training.schedule, replay_clock
            ),
        )
        replay, replay_metrics, _ = train_step(replay, next(batches))
        jax.block_until_ready((replay, replay_metrics))
        replay_values = routing_diagnostics(
            replay_metrics,
            batch_size=GLOBAL_BATCH_SIZE,
            model_config=training.model,
        )
        if any(
            float(replay_values[key]) != 0.0
            for key in (
                "moe/drop_fraction",
                "moe/sender_drop_fraction",
                "moe/receiver_drop_fraction",
            )
        ):
            raise ValueError("Checkpoint replay produced MoE capacity drops")

        steady = step_seconds[5:]
        result = {
            "status": "complete",
            "run_id": config.run_id,
            "source_checkpoint": config.source_checkpoint,
            "source_step": source_summary["step"],
            "final_step": clock.updates,
            "checkpoint": checkpoint,
            "runtime_metadata": training.runtime_metadata,
            "scientific_config_sha256": digest,
            "tokenizer_sha256": tokenizer_sha256,
            "updates": config.updates,
            "mean_step_seconds_after_five": statistics.mean(steady),
            "median_step_seconds_after_five": statistics.median(steady),
            "mean_tokens_per_second_after_five": TOKENS_PER_UPDATE
            / statistics.mean(steady),
            "median_tokens_per_second_after_five": TOKENS_PER_UPDATE
            / statistics.median(steady),
            "first_loss": losses[0],
            "last_loss": losses[-1],
            "max_drop_fraction": 0.0,
            "source_state": source_summary,
            "replay_state": replay_summary,
            "replay_loss": float(replay_values["train/loss"]),
        }
        result_uri = f"{config.artifact_root}/result.json"
        payload = json.dumps(result, sort_keys=True)
        StoragePath(result_uri).write_text(payload)
        if StoragePath(result_uri).read_text() != payload:
            raise ValueError("Validation result readback failed")
        tracker.log_summary(
            {
                "validation/phase": "complete",
                "validation/result_uri": result_uri,
                **{f"validation/{key}": value for key, value in result.items()},
            }
        )
        tracker.log({"run_progress": 1.0}, step=config.updates + 1)
    tracker.current_tracker().finish()


def dispatch(config: ValidationConfig) -> None:
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
            timeout=Duration.from_hours(3),
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
    parser.add_argument("--updates", type=int, default=25)
    parser.add_argument("--validation-id", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--source-metadata-sha256", required=True)
    parser.add_argument("--runtime-code-sha256", required=True)
    dispatch(ValidationConfig(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
