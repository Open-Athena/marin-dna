"""Bounded full-model validation; this entry point cannot start a production trial."""

import argparse
import dataclasses
import os
import time
from collections.abc import Callable
from typing import Literal

import equinox as eqx
import fsspec
import jax
import jmp
import numpy as np
import pyarrow.parquet as pq
from fray.cluster import ResourceConfig
from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, create_environment
from iris.rpc.proto_display import priority_band_value
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter import tracker
from levanter.callbacks.watch import WatchConfig
from levanter.data.text.examples import GrugLmExample
from levanter.grug.attention import AttentionMask
from levanter.grug.grug_moe import (
    MOE_DROPPED_ASSIGNMENTS_METRIC,
    MOE_RECEIVER_DROPPED_ASSIGNMENTS_METRIC,
    MOE_SENDER_DROPPED_ASSIGNMENTS_METRIC,
    MOE_SKIPPED_PADDING_ASSIGNMENTS_METRIC,
    MOE_VALID_ASSIGNMENTS_METRIC,
)
from levanter.grug.sharding import compact_grug_mesh
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.training.training import resolve_training_env
from rigging.timing import Duration

from exp586_moe.config import SOURCE_CHECKPOINT, SOURCE_METADATA_DIGEST, d768_config
from exp586_moe.data import (
    WindowBatch,
    encode_windows,
    pretrained_dna_tokenizer,
    scratch_tokenizer,
    tokenizer_digest,
)
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
    state_error_by_group,
    state_max_abs_error,
)
from exp586_moe.tracking import wandb_history_offset
from experiments.grug.dispatch import _forwarded_env_vars
from experiments.grug.moe_hero_ep.model import Transformer
from experiments.grug.moe_hero_ep.train import (
    GrugTrainState,
    MasterParamMode,
    _apply_hero_ep_runtime_defaults,
    _drop_metrics,
    _make_train_step,
)
from experiments.grug.moe_hero_ep.weights import restore_weights

RAW_FIRST_SHARD = "s3://marin-us-east-02a/MarinDNA/data/plantcad/Angiosperm_65_genomes_8192bp/train/data-00000-of-00044.parquet"


@dataclasses.dataclass(frozen=True)
class SmokeConfig:
    condition: Literal["scratch", "pretrained"]
    run_id: str
    cluster: str
    nodes: int = 1

    def __post_init__(self) -> None:
        if self.condition not in ("scratch", "pretrained") or self.nodes not in (
            1,
            2,
            4,
            8,
            16,
        ):
            raise ValueError("Invalid condition or whole-node H100 shape")
        if self.cluster not in ("cw-us-east-02a", "cw-rno2a"):
            raise ValueError("Unapproved H100 cluster")
        if not self.run_id.startswith("exp586-") or "smoke" not in self.run_id:
            raise ValueError("Validation requires a separate exp586 smoke identity")

    @property
    def checkpoint_root(self) -> str:
        return f"s3://marin-us-east-02a/tmp/ttl=7d/MarinDNA/exp586_plantcad2_moe/{self.run_id}/checkpoints"


def device_batch(batch: WindowBatch, mesh: jax.sharding.Mesh) -> GrugLmExample:
    sharding = NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None))

    def put(array: np.ndarray) -> jax.Array:
        return jax.make_array_from_callback(
            array.shape, sharding, lambda index: array[index]
        )

    return GrugLmExample(
        tokens=put(batch.token_ids),
        loss_weight=put(batch.loss_weights),
        attn_mask=AttentionMask.causal().with_segment_ids(put(batch.segment_ids)),
    )


def run_canary(config: SmokeConfig) -> None:
    mp = jmp.get_policy("params=bfloat16,compute=bfloat16,output=bfloat16")
    batch_size = config.nodes * 8
    trainer = TrainerConfig(
        id=config.run_id,
        seed=0,
        train_batch_size=batch_size,
        num_train_steps=3,
        require_accelerator=True,
        mp=mp,
        use_explicit_mesh_axes=True,
        watch=WatchConfig(interval=0, watch_targets=[]),
        tracker=WandbConfig(
            entity="eric-czech",
            project="marin",
            name=config.run_id,
            group="exp586-plantcad2-moe-validation",
            tags=["dna-exp586", "validation", "H100", config.cluster, config.condition],
            save_code=False,
            background=False,
            replicate_path=config.checkpoint_root,
        ),
    )
    trainer.initialize()
    # A diagnosed validation retry keeps its W&B identity and appends history.
    log_offset = wandb_history_offset()
    assert jax.device_count() == 8 * config.nodes
    assert all("H100" in device.device_kind for device in jax.devices())
    tracker.log_configuration(config)
    tracker.log_summary(
        {"validation/phase": "loading", "gpu_type": "H100", "nodes": config.nodes}
    )

    def reject(message: str) -> None:
        # All ranks reach these numerical checks together. Flush the primary's
        # evidence before any rank exits and Iris terminates its peers.
        tracker.log_summary({"validation/phase": "failed", "validation/error": message})
        tracker.get_tracker("wandb").run.finish(exit_code=1)
        multihost_utils.sync_global_devices("exp586-validation-failure")
        raise ValueError(message)

    with fsspec.open(RAW_FIRST_SHARD, "rb") as source:
        windows = (
            pq.ParquetFile(source)
            .read_row_group(0, columns=["seq"])
            .column("seq")
            .to_pylist()[:batch_size]
        )
    tokenizer = (
        scratch_tokenizer()
        if config.condition == "scratch"
        else pretrained_dna_tokenizer()
    )
    tokenizer_sha256 = tokenizer_digest(tokenizer)
    tracker.log_hyperparameters(
        {"tokenizer_sha256": tokenizer_sha256, "dna_encoding": "character-v1"}
    )
    host_batches = [
        encode_windows(
            tokenizer,
            windows,
            occurrences=list(range(i * batch_size, (i + 1) * batch_size)),
        )
        for i in range(3)
    ]
    # Only the validation horizon is shortened. The production token schedule is tested separately.
    reference_tokens = host_batches[0].input_tokens
    schedule = HeroTokenSchedule(
        total_updates=3,
        warmup_updates=1,
        tokens_per_update=reference_tokens,
        peak_update=2,
    )
    opt_config = optimizer_config(reference_tokens_per_update=reference_tokens)
    optimizer = opt_config.build(3)
    tracker.log_hyperparameters(
        {
            "model": dataclasses.asdict(d768_config(config.condition)),
            "optimizer": dataclasses.asdict(opt_config),
            "validation_schedule": dataclasses.asdict(schedule),
            "reference_real_tokens_per_update": reference_tokens,
            "source_checkpoint": SOURCE_CHECKPOINT
            if config.condition == "pretrained"
            else None,
        }
    )
    mesh = compact_grug_mesh(expert_axis_size=8, replica_axis_size=config.nodes)
    with jax.set_mesh(mesh):
        model_config = d768_config(config.condition)
        if config.condition == "pretrained":
            weights = restore_weights(
                SOURCE_CHECKPOINT, SOURCE_METADATA_DIGEST, model_config, mesh
            )
        else:
            weights = eqx.filter_jit(Transformer.init)(
                model_config, key=jax.random.key(0)
            )
        state = eqx.filter_jit(lambda w: fresh_state(w, optimizer, mp, offload=True))(
            weights
        )
        del weights
        clock = TokenClock()
        train_step = _make_train_step(
            optimizer,
            mp,
            z_loss_weight=1e-4,
            ema_beta=None,
            offload_opt_state=True,
            master_param_mode=MasterParamMode.FP32_PINNED_HOST,
        )
        batches = [device_batch(batch, mesh) for batch in host_batches]

        def step(
            state: GrugTrainState, clock: TokenClock, i: int
        ) -> tuple[GrugTrainState, TokenClock, float, float, dict[str, int | float]]:
            state = dataclasses.replace(
                state,
                opt_state=set_token_learning_rates(
                    state.opt_state, opt_config, schedule, clock
                ),
            )
            started = time.monotonic()
            state, metrics, _ = train_step(state, batches[i])
            jax.block_until_ready(state)
            loss = float(metrics["train/loss"])
            if not np.isfinite(loss):
                raise ValueError(f"Nonfinite loss at update {i}")
            batch = host_batches[i]
            routing = _drop_metrics(
                metrics[MOE_DROPPED_ASSIGNMENTS_METRIC],
                metrics[MOE_SENDER_DROPPED_ASSIGNMENTS_METRIC],
                metrics[MOE_RECEIVER_DROPPED_ASSIGNMENTS_METRIC],
                metrics[MOE_SKIPPED_PADDING_ASSIGNMENTS_METRIC],
                metrics[MOE_VALID_ASSIGNMENTS_METRIC],
                batch_size=batch_size,
                sequence_length=8192,
                top_k=model_config.num_experts_per_token,
                num_layers=model_config.num_layers,
            )
            expected_valid = (
                batch.input_tokens
                * model_config.num_experts_per_token
                * model_config.num_layers
            )
            if routing[MOE_VALID_ASSIGNMENTS_METRIC] != expected_valid:
                raise ValueError(
                    "Router valid assignments differ from the real input-token count"
                )
            clock = clock.advance(
                input_tokens=batch.input_tokens,
                loss_targets=batch.loss_targets,
                examples=batch_size,
            )
            return state, clock, loss, time.monotonic() - started, routing

        state, clock, loss, elapsed, routing = step(state, clock, 0)
        tracker.log(
            {
                "run_progress": 0.2,
                "validation/update": 1,
                "train/loss": loss,
                "validation/step_seconds": elapsed,
                **routing,
            },
            step=log_offset + 1,
        )
        checkpoint = f"{config.checkpoint_root}/step-1"
        save_dna_checkpoint(
            state, clock, checkpoint, data_seed=0, tokenizer_sha256=tokenizer_sha256
        )
        replay, replay_clock = restore_dna_checkpoint(
            state, checkpoint, mesh, data_seed=0, tokenizer_sha256=tokenizer_sha256
        )
        roundtrip_error = float(state_max_abs_error(state, replay))
        tracker.log_summary({"validation/restore_max_abs_error": roundtrip_error})
        if roundtrip_error != 0 or clock != replay_clock:
            reject(
                f"Immediate checkpoint round trip differs: max error {roundtrip_error}"
            )
        for original, restored in zip(
            jax.tree.leaves(state), jax.tree.leaves(replay), strict=True
        ):
            if (
                original.dtype != restored.dtype
                or original.sharding != restored.sharding
            ):
                reject("Checkpoint changed dtype or sharding")
        for i in (1, 2):
            state, clock, loss, elapsed, routing = step(state, clock, i)
            tracker.log(
                {
                    "run_progress": 0.2 + 0.2 * i,
                    "validation/update": i + 1,
                    "train/loss": loss,
                    "validation/step_seconds": elapsed,
                    **routing,
                },
                step=log_offset + i + 1,
            )
        tracker.log_summary({"validation/phase": "resume_replay"})
        for i in (1, 2):
            replay, replay_clock, _, _, _ = step(replay, replay_clock, i)
        if clock != replay_clock:
            raise ValueError("Resumed token/data clock differs")

        error = float(state_max_abs_error(state, replay))
        details = {
            "validation/replay/" + key: float(value)
            for key, value in state_error_by_group(state, replay).items()
        }
        tracker.log_summary({"validation/replay_max_abs_error": error, **details})
        if (
            not np.isfinite(error)
            or not bool(state_all_finite(state))
            or not bool(state_all_finite(replay))
        ):
            reject("Nonfinite replay state or comparison")
        tracker.log_summary({"validation/non_bitwise_updates_accepted": True})
        final_checkpoint = f"{config.checkpoint_root}/step-3"
        save_dna_checkpoint(
            replay,
            replay_clock,
            final_checkpoint,
            data_seed=0,
            tokenizer_sha256=tokenizer_sha256,
        )
        tracker.log_summary(
            {
                "validation/phase": "complete",
                "validation/replay_max_abs_error": error,
                "validation/final_checkpoint": final_checkpoint,
                "validation/input_tokens": clock.input_tokens,
            }
        )
        tracker.log({"run_progress": 1.0}, step=log_offset + 4)
    tracker.current_tracker().finish()


def dispatch(
    config: SmokeConfig,
    *,
    worker: Callable[[SmokeConfig], None] = run_canary,
    moe_implementation: str = "fixed_pooled_wave_all_to_all",
    inline_watch_enabled: bool = False,
    timeout_hours: int = 1,
) -> None:
    if timeout_hours < 1:
        raise ValueError("Dispatch timeout must be at least one hour")
    _apply_hero_ep_runtime_defaults(
        inline_watch_enabled=inline_watch_enabled,
        moe_implementation=moe_implementation,
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
        },
        resources,
    )
    job = current_client().submit(
        JobRequest(
            name=f"train-{config.run_id}",
            entrypoint=Entrypoint.from_callable(worker, args=[config]),
            resources=resources,
            processes_per_task=8,
            environment=create_environment(env_vars=env, extras=["gpu"]),
            # Explicit even on the child: direct invocation must never default to interactive.
            priority=priority_band_value("batch"),
            timeout=Duration.from_hours(timeout_hours),
            max_retries_failure=0,
            max_task_failures=0,
        )
    )
    job.wait(raise_on_failure=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=["scratch", "pretrained"], required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--cluster", choices=["cw-us-east-02a", "cw-rno2a"], required=True
    )
    parser.add_argument("--nodes", type=int, choices=[1, 2, 4, 8], default=1)
    args = parser.parse_args()
    dispatch(SmokeConfig(**vars(args)))


if __name__ == "__main__":
    main()
