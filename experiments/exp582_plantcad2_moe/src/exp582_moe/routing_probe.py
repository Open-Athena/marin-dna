"""Bounded routing diagnostics on unchanged, unpacked 8K genomic windows."""

import argparse
import dataclasses
import gc
import hashlib
import json
import math
import time
from enum import Enum
from typing import Literal

import equinox as eqx
import fsspec
import jax
import jmp
import numpy as np
import pyarrow.parquet as pq
from jax.experimental import multihost_utils
from levanter import tracker
from levanter.callbacks.watch import WatchConfig
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

from exp582_moe.config import SOURCE_CHECKPOINT, SOURCE_METADATA_DIGEST, d1536_config
from exp582_moe.data import (
    WindowBatch,
    encode_windows,
    pretrained_dna_tokenizer,
    scratch_tokenizer,
    tokenizer_digest,
)
from exp582_moe.gpu_smoke import (
    RAW_FIRST_SHARD,
    SmokeConfig,
    device_batch,
    dispatch,
)
from exp582_moe.precision import RouterFloat32Policy
from exp582_moe.replay import three_way_replay
from exp582_moe.routing import damp_qb_updates
from exp582_moe.schedule import TokenClock, optimizer_config
from exp582_moe.state import (
    fresh_state,
    restore_dna_checkpoint,
    save_dna_checkpoint,
    state_all_finite,
)
from exp582_moe.tracking import wandb_history_offset
from experiments.grug.moe_hero_ep.model import GrugModelConfig, Transformer
from experiments.grug.moe_hero_ep.train import (
    MasterParamMode,
    _drop_metrics,
    _make_train_step,
)
from experiments.grug.moe_hero_ep.weights import restore_weights


@dataclasses.dataclass(frozen=True)
class RoutingProbeConfig(SmokeConfig):
    backend: Literal["pooled", "dropless"] = "pooled"
    capacity_factor: float = 1.15
    transport_capacity_factor: float | None = None
    batch_size: int = 8
    steps: int = 200
    warmup_updates: int = 100
    router_fp32: bool = False
    replay_control: bool = False
    resume_replay_checkpoint: str | None = None
    resume_metadata_digest: str | None = None
    resume_reference_uri: str | None = None
    resume_reference_sha256: str | None = None
    qb_update_rate: float = 1.0
    peak_lr_multiplier: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.backend not in ("pooled", "dropless") or self.capacity_factor <= 0:
            raise ValueError("Invalid routing backend or capacity")
        if not math.isfinite(self.capacity_factor):
            raise ValueError("Capacity must be finite")
        if self.transport_capacity_factor is not None and (
            not math.isfinite(self.transport_capacity_factor)
            or self.transport_capacity_factor <= 0
        ):
            raise ValueError("Transport capacity must be finite and positive")
        if not 0 < self.qb_update_rate <= 1:
            raise ValueError("QB update rate must lie in (0, 1]")
        if not math.isfinite(self.peak_lr_multiplier) or self.peak_lr_multiplier < 0:
            raise ValueError("Peak LR multiplier must be finite and nonnegative")
        if self.batch_size < self.nodes * 8 or self.batch_size % (self.nodes * 8):
            raise ValueError("Batch must contain whole data-parallel batches")
        if not 1 <= self.steps <= 1000 or not 1 <= self.warmup_updates <= self.steps:
            raise ValueError("Bounded diagnostics require 1..1000 updates")
        if self.resume_replay_checkpoint is not None:
            prefix = (
                "s3://marin-us-east-02a/tmp/ttl=7d/MarinDNA/exp582_plantcad2_moe/"
                f"exp582-plantcad2-d1536-{self.condition}-"
            )
            if (
                not self.replay_control
                or not self.resume_replay_checkpoint.startswith(prefix)
                or "smoke" not in self.resume_replay_checkpoint
                or not self.resume_replay_checkpoint.endswith(
                    f"/checkpoints/step-{self.steps}"
                )
            ):
                raise ValueError("Replay recovery requires the matching smoke endpoint")
            digest = self.resume_metadata_digest or ""
            if len(digest) != 64 or any(x not in "0123456789abcdef" for x in digest):
                raise ValueError("Replay recovery requires a metadata SHA-256")
            reference = self.resume_reference_uri or ""
            digest = self.resume_reference_sha256 or ""
            if (
                not reference.startswith(prefix.rsplit("/", 1)[0] + "/")
                or len(digest) != 64
                or any(x not in "0123456789abcdef" for x in digest)
            ):
                raise ValueError(
                    "Replay recovery requires a digest-bound configuration reference"
                )
        elif any(
            x is not None
            for x in (
                self.resume_metadata_digest,
                self.resume_reference_uri,
                self.resume_reference_sha256,
            )
        ):
            raise ValueError("Recovery references require a replay checkpoint")

    @property
    def implementation(self) -> str:
        # Existing portable grouped GEMM; no expert-capacity clipping with EP=1.
        return (
            "scatter" if self.backend == "dropless" else "fixed_pooled_wave_all_to_all"
        )

    @property
    def expert_axis_size(self) -> int:
        return 1 if self.backend == "dropless" else 8


def probe_model_config(config: RoutingProbeConfig) -> GrugModelConfig:
    """Change dispatch/storage layout only; preserve model parameters and routing."""
    return dataclasses.replace(
        d1536_config(config.condition),
        moe_implementation=config.implementation,
        capacity_factor=config.capacity_factor,
        pooled_transport_capacity_factor=config.transport_capacity_factor
        if config.transport_capacity_factor is not None
        else config.capacity_factor,
        expert_chunks=1,
    )


def verify_replay_clock(config: RoutingProbeConfig, clock: TokenClock) -> None:
    if (
        clock.updates != config.steps
        or clock.examples != config.steps * config.batch_size
    ):
        raise ValueError(
            "Replay endpoint clock differs from the configured data cursor"
        )


def verify_replay_reference(
    config: RoutingProbeConfig,
    reference: dict,
    model: dict,
    optimizer: dict,
    reference_tokens: int,
) -> None:
    """Bind recovery settings to the original run, including resolved defaults."""
    original = reference["configuration"]
    for key, value in dataclasses.asdict(config).items():
        if key in ("run_id", "replay_control") or key.startswith("resume_"):
            continue
        if key not in original or original[key] != value:
            raise ValueError(f"Replay configuration differs for {key}")
    source = SmokeConfig(
        config.condition, original["run_id"], config.cluster, config.nodes
    )
    if (
        reference["source_run_id"] != original["run_id"]
        or reference["checkpoint"] != config.resume_replay_checkpoint
        or reference["checkpoint"] != f"{source.checkpoint_root}/step-{config.steps}"
        or reference["checkpoint_metadata_digest"] != config.resume_metadata_digest
    ):
        raise ValueError("Replay reference identifies a different checkpoint")
    for key, current in (("model", model), ("optimizer", optimizer)):
        if not recorded_config_equal(original[key], current):
            raise ValueError(f"Replay resolved {key} differs from the source run")
    if original["reference_real_tokens_per_update"] != reference_tokens:
        raise ValueError("Replay LR reference batch differs from the source run")


def recorded_config_equal(recorded: object, current: object) -> bool:
    """Account only for W&B's enum names and JSON float serialization rounding."""
    if isinstance(current, Enum):
        return recorded == current.name
    if isinstance(current, dict):
        return (
            isinstance(recorded, dict)
            and recorded.keys() == current.keys()
            and all(
                recorded_config_equal(recorded[key], value)
                for key, value in current.items()
            )
        )
    if isinstance(current, (list, tuple)):
        return (
            isinstance(recorded, list)
            and len(recorded) == len(current)
            and all(
                recorded_config_equal(x, y)
                for x, y in zip(recorded, current, strict=True)
            )
        )
    if isinstance(current, float):
        return isinstance(recorded, (float, int)) and math.isclose(
            recorded, current, rel_tol=1e-14, abs_tol=0
        )
    return recorded == current


def routing_diagnostics(
    metrics: dict[str, object], *, batch_size: int, model_config: GrugModelConfig
) -> dict[str, float | int]:
    """Log real assignment counts and layer concentration without averaging ranks."""
    result = _drop_metrics(
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
    for key, value in metrics.items():
        if isinstance(value, jax.Array) and value.ndim == 0:
            result[key] = float(value)
    counts = np.asarray(metrics["train/router/routing_counts_per_layer"])
    for layer, row in enumerate(counts):
        prefix = f"train/router/layer_{layer}"
        total = max(float(row.sum()), 1)
        result[f"{prefix}/active_experts"] = int(np.count_nonzero(row))
        result[f"{prefix}/max_to_mean_load"] = float(row.max() * len(row) / total)
        result[f"{prefix}/top32_assignment_fraction"] = float(
            np.sort(row)[-32:].sum() / total
        )
    return result


def run_probe(config: RoutingProbeConfig) -> None:
    mp = jmp.get_policy("params=bfloat16,compute=bfloat16,output=bfloat16")
    if config.router_fp32:
        mp = RouterFloat32Policy(mp.param_dtype, mp.compute_dtype, mp.output_dtype)
    trainer = TrainerConfig(
        id=config.run_id,
        seed=0,
        train_batch_size=config.batch_size,
        num_train_steps=config.steps,
        require_accelerator=True,
        mp=mp,
        use_explicit_mesh_axes=True,
        watch=WatchConfig(interval=0, watch_targets=[]),
        tracker=WandbConfig(
            entity="eric-czech",
            project="marin",
            name=config.run_id,
            group="exp582-plantcad2-moe-routing",
            tags=[
                "dna-exp582",
                "validation",
                "H100",
                config.cluster,
                config.condition,
                config.backend,
            ],
            save_code=False,
            background=False,
            replicate_path=config.checkpoint_root,
        ),
    )
    trainer.initialize()
    assert jax.device_count() == 8 * config.nodes
    assert all("H100" in device.device_kind for device in jax.devices())
    tracker.log_configuration(config)
    tracker.log_summary({"validation/phase": "loading", "gpu_type": "H100"})
    log_offset = wandb_history_offset()

    def reject(message: str) -> None:
        tracker.log_summary({"validation/phase": "failed", "validation/error": message})
        tracker.get_tracker("wandb").run.finish(exit_code=1)
        multihost_utils.sync_global_devices("exp582-routing-failure")
        raise ValueError(message)

    tokenizer = (
        scratch_tokenizer()
        if config.condition == "random"
        else pretrained_dna_tokenizer()
    )
    tokenizer_sha256 = tokenizer_digest(tokenizer)
    tracker.log_hyperparameters(
        {"tokenizer_sha256": tokenizer_sha256, "dna_encoding": "character-v1"}
    )
    with fsspec.open(RAW_FIRST_SHARD, "rb") as source:
        parquet = pq.ParquetFile(source)
        requested_batches = config.steps + (2 if config.replay_control else 0)
        if requested_batches * config.batch_size > parquet.metadata.num_rows:
            reject("Diagnostic requests more distinct windows than its source shard")
        rows = parquet.iter_batches(batch_size=config.batch_size, columns=["seq"])

        def next_batch(update: int) -> WindowBatch:
            windows = next(rows).column("seq").to_pylist()
            if len(windows) != config.batch_size:
                raise ValueError("Diagnostic exhausted its source shard")
            return encode_windows(
                tokenizer,
                windows,
                occurrences=list(
                    range(update * config.batch_size, (update + 1) * config.batch_size)
                ),
            )

        first = next_batch(0)
        opt_config = optimizer_config(reference_tokens_per_update=first.input_tokens)
        optimizer = opt_config.build(config.steps)
        tracker.log_hyperparameters(
            {
                "optimizer": dataclasses.asdict(opt_config),
                "reference_real_tokens_per_update": first.input_tokens,
                "jax_default_matmul_precision": str(
                    jax.config.jax_default_matmul_precision
                ),
                "router_precision": "FP32 weights, biases, logits; existing XLA matmul policy"
                if config.router_fp32
                else "native BF16 parameter and compute policy",
                "diagnostic_lr_schedule": "linear warmup then constant peak; not the production schedule",
                "source_checkpoint": SOURCE_CHECKPOINT
                if config.condition == "pretrained"
                else None,
            }
        )
        mesh = compact_grug_mesh(
            expert_axis_size=config.expert_axis_size, replica_axis_size=config.nodes
        )
        with jax.set_mesh(mesh):
            model_config = probe_model_config(config)
            tracker.log_hyperparameters({"model": dataclasses.asdict(model_config)})
            if config.resume_replay_checkpoint is not None:
                with fsspec.open(config.resume_reference_uri, "rb") as source_reference:
                    reference_blob = source_reference.read()
                if (
                    hashlib.sha256(reference_blob).hexdigest()
                    != config.resume_reference_sha256
                ):
                    reject("Replay configuration reference changed")
                verify_replay_reference(
                    config,
                    json.loads(reference_blob),
                    dataclasses.asdict(model_config),
                    dataclasses.asdict(opt_config),
                    first.input_tokens,
                )
            weights = (
                restore_weights(
                    SOURCE_CHECKPOINT, SOURCE_METADATA_DIGEST, model_config, mesh
                )
                if config.condition == "pretrained"
                and config.resume_replay_checkpoint is None
                else eqx.filter_jit(Transformer.init)(
                    model_config, key=jax.random.key(0)
                )
            )
            state = eqx.filter_jit(
                lambda w: fresh_state(w, optimizer, mp, offload=True)
            )(weights)
            del weights
            clock = TokenClock()
            start_update = 0
            if config.resume_replay_checkpoint is not None:
                checkpoint = config.resume_replay_checkpoint
                with fsspec.open(f"{checkpoint}/metadata.json") as metadata_file:
                    metadata = json.load(metadata_file)
                digest = hashlib.sha256(
                    json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                if digest != config.resume_metadata_digest:
                    reject("Replay checkpoint metadata changed")
                state, clock = restore_dna_checkpoint(
                    state,
                    checkpoint,
                    mesh,
                    data_seed=0,
                    tokenizer_sha256=tokenizer_sha256,
                )
                gc.collect()
                verify_replay_clock(config, clock)
                endpoint_factor = config.peak_lr_multiplier * min(
                    1.0, (config.steps - 1) / config.warmup_updates
                )
                for key in ("learning_rate", "adam_lr"):
                    # Pinned-host scalars can expose zero-copy NumPy views.
                    # Retaining one would prevent the next donated train step.
                    actual = np.array(state.opt_state.hyperparams[key], copy=True)
                    expected = np.asarray(
                        getattr(opt_config, key) * endpoint_factor, dtype=actual.dtype
                    )
                    if not np.array_equal(actual, expected):
                        reject(
                            f"Restored endpoint {key} differs from the source schedule"
                        )
                # The first batch was read to recover the original LR reference.
                # Skip the remaining primary-training windows without encoding them.
                for _ in range(config.steps - 1):
                    next(rows)
                start_update = config.steps
                tracker.log_summary(
                    {
                        "validation/replay_from_checkpoint": True,
                        "validation/replay_original_live_state_available": False,
                        "validation/new_primary_updates": 0,
                    }
                )
            train_step = _make_train_step(
                optimizer,
                mp,
                z_loss_weight=1e-4,
                ema_beta=None,
                offload_opt_state=True,
                master_param_mode=MasterParamMode.FP32_PINNED_HOST,
            )
            train_step = damp_qb_updates(train_step, config.qb_update_rate)
            drops = []
            tracker.log_summary({"validation/phase": "training"})
            for update in range(start_update, config.steps):
                batch = first if update == 0 else next_batch(update)
                factor = config.peak_lr_multiplier * min(
                    1.0, update / config.warmup_updates
                )
                state = dataclasses.replace(
                    state,
                    opt_state=state.opt_state._replace(
                        hyperparams={
                            **state.opt_state.hyperparams,
                            "learning_rate": opt_config.learning_rate * factor,
                            "adam_lr": opt_config.adam_lr * factor,
                        }
                    ),
                )
                started = time.monotonic()
                state, metrics, _ = train_step(state, device_batch(batch, mesh))
                jax.block_until_ready(state)
                elapsed = time.monotonic() - started
                try:
                    measured = routing_diagnostics(
                        metrics, batch_size=config.batch_size, model_config=model_config
                    )
                except ValueError as error:
                    reject(str(error))
                if not np.isfinite(measured["train/loss"]):
                    reject("Nonfinite training loss")
                expected = (
                    batch.input_tokens
                    * model_config.num_experts_per_token
                    * model_config.num_layers
                )
                if measured[MOE_VALID_ASSIGNMENTS_METRIC] != expected:
                    reject("Router assignment count differs from real tokens")
                if (
                    config.backend == "dropless"
                    and measured[MOE_DROPPED_ASSIGNMENTS_METRIC] != 0
                ):
                    reject("Dropless backend rejected expert assignments")
                drops.append(measured["moe/drop_fraction"])
                clock = clock.advance(
                    input_tokens=batch.input_tokens,
                    loss_targets=batch.loss_targets,
                    examples=config.batch_size,
                )
                tracker.log(
                    {
                        **measured,
                        "validation/update": update + 1,
                        "validation/step_seconds": elapsed,
                        "throughput/input_tokens": clock.input_tokens,
                        "throughput/tokens_per_second": batch.input_tokens / elapsed,
                        "optimizer/lr_factor": factor,
                        "optimizer/learning_rate": opt_config.learning_rate * factor,
                        "optimizer/adam_lr": opt_config.adam_lr * factor,
                        "run_progress": 0.99 * (update + 1) / config.steps,
                    },
                    step=log_offset + update + 1,
                )
            if not bool(state_all_finite(state)):
                reject("Final model or optimizer state is nonfinite")
            tracker.log_summary(
                {"validation/phase": "saving", "validation/final_state_finite": True}
            )
            if config.resume_replay_checkpoint is None:
                checkpoint = f"{config.checkpoint_root}/step-{clock.updates}"
                save_dna_checkpoint(
                    state,
                    clock,
                    checkpoint,
                    data_seed=0,
                    tokenizer_sha256=tokenizer_sha256,
                )
            if config.replay_control:
                tracker.log_summary({"validation/phase": "replay_control"})
                replay_batches = [
                    device_batch(next_batch(config.steps + i), mesh) for i in range(2)
                ]
                try:
                    replay_metrics = three_way_replay(
                        state,
                        clock,
                        checkpoint,
                        mesh,
                        replay_batches,
                        train_step,
                        lambda values: tracker.log_summary(
                            {
                                f"validation/{key}": value
                                for key, value in values.items()
                            }
                        ),
                        tokenizer_sha256=tokenizer_sha256,
                    )
                except (ValueError, jax.errors.JaxRuntimeError) as error:
                    reject(str(error))
                tracker.log_summary(
                    {
                        f"validation/{key}": value
                        for key, value in replay_metrics.items()
                    }
                )
            tracker.log_summary(
                {
                    "validation/phase": "complete",
                    "validation/final_checkpoint": checkpoint,
                    **(
                        {
                            "validation/final20_mean_drop_fraction": float(
                                np.mean(drops[-20:])
                            ),
                            "validation/final20_max_drop_fraction": float(
                                np.max(drops[-20:])
                            ),
                        }
                        if drops
                        else {}
                    ),
                    "validation/input_tokens": clock.input_tokens,
                }
            )
            tracker.log({"run_progress": 1.0}, step=log_offset + config.steps + 1)
    tracker.current_tracker().finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=["random", "pretrained"], required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--cluster", choices=["cw-us-east-02a", "cw-rno2a"], required=True
    )
    parser.add_argument("--nodes", type=int, choices=[1, 2, 4, 8], default=1)
    parser.add_argument("--backend", choices=["pooled", "dropless"], default="pooled")
    parser.add_argument("--capacity-factor", type=float, default=1.15)
    parser.add_argument("--transport-capacity-factor", type=float)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup-updates", type=int, default=100)
    parser.add_argument("--router-fp32", action="store_true")
    parser.add_argument("--replay-control", action="store_true")
    parser.add_argument("--resume-replay-checkpoint")
    parser.add_argument("--resume-metadata-digest")
    parser.add_argument("--resume-reference-uri")
    parser.add_argument("--resume-reference-sha256")
    parser.add_argument("--qb-update-rate", type=float, default=1.0)
    parser.add_argument("--peak-lr-multiplier", type=float, default=1.0)
    config = RoutingProbeConfig(**vars(parser.parse_args()))
    dispatch(config, worker=run_probe, moe_implementation=config.implementation)


if __name__ == "__main__":
    main()
