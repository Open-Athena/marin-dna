"""Restore the exp586 continuation peak and prove a fresh checkpoint round trip."""

import argparse
import dataclasses
import hashlib
import json
import os

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

from exp586_moe.config import (
    CONTINUATION_TOTAL_UPDATES,
    GLOBAL_BATCH_SIZE,
)
from exp586_moe.data import pretrained_dna_tokenizer, tokenizer_digest
from exp586_moe.schedule import optimizer_config
from exp586_moe.state import (
    fresh_state,
    restore_dna_checkpoint,
    save_dna_checkpoint,
    state_error_by_group,
    state_max_abs_error,
    validate_restored_training_state,
)
from exp586_moe.train import TrainingConfig, scientific_config, training_mesh
from experiments.grug.dispatch import _forwarded_env_vars
from experiments.grug.moe_hero_ep.model import Transformer
from experiments.grug.moe_hero_ep.train import (
    _apply_hero_ep_runtime_defaults,
    grug_trainer_mesh_config,
)

PEAK_CHECKPOINT = (
    "s3://marin-us-east-02a/MarinDNA/exp586_plantcad2_moe/checkpoints/"
    "exp586-plantcad2-d768-pretrained-lrm1-seed0-s02-v1/2026.10.19/"
    "checkpoints/step-418887"
)
PEAK_METADATA_SHA256 = (
    "885305733a08ac9e44d9d4e0838074cb9db5f5434b375c885e0afd4ad90e160e"
)
PEAK_SCIENTIFIC_CONFIG_SHA256 = (
    "c4c2de2c609059ad604f2fff13f650a569e8ba99ea5ca007f0e7d4745673356a"
)


@dataclasses.dataclass(frozen=True)
class CheckpointRoundtripConfig:
    cluster: str
    nodes: int
    validation_id: str

    def __post_init__(self) -> None:
        if self.cluster not in ("cw-rno2a", "cw-us-east-02a"):
            raise ValueError("Checkpoint validation requires an approved H100 cluster")
        if self.nodes not in (1, 2, 4, 8):
            raise ValueError("Checkpoint validation requires 8, 16, 32, or 64 H100s")
        if not self.validation_id or not self.validation_id.replace("-", "").isalnum():
            raise ValueError(
                "Validation ID must contain only letters, digits, and hyphens"
            )

    @property
    def run_id(self) -> str:
        return f"exp586-plantcad2-d768-pretrained-s02-checkpoint-{self.validation_id}"

    @property
    def checkpoint_root(self) -> str:
        return (
            "s3://marin-us-east-02a/tmp/ttl=7d/MarinDNA/"
            f"exp586_plantcad2_moe/{self.run_id}/checkpoints"
        )


def run_roundtrip(config: CheckpointRoundtripConfig) -> None:
    training = TrainingConfig(
        "pretrained", 1.0, config.cluster, config.nodes, stage="s02"
    )
    mp = jmp.get_policy("params=bfloat16,compute=bfloat16,output=bfloat16")
    trainer = TrainerConfig(
        id=config.run_id,
        seed=0,
        train_batch_size=GLOBAL_BATCH_SIZE,
        num_train_steps=CONTINUATION_TOTAL_UPDATES,
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
            tags=["dna-exp586", "H100", config.cluster, "checkpoint-roundtrip"],
            save_code=False,
            background=False,
            replicate_path=config.checkpoint_root,
        ),
    )
    trainer.initialize()
    assert jax.device_count() == config.nodes * 8
    assert all("H100" in device.device_kind for device in jax.devices())
    tracker.log_summary({"validation/phase": "restoring", "nodes": config.nodes})

    tokenizer_sha256 = tokenizer_digest(pretrained_dna_tokenizer())
    scientific = scientific_config(training, tokenizer_sha256)
    output_digest = hashlib.sha256(
        json.dumps(scientific, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if output_digest != PEAK_SCIENTIFIC_CONFIG_SHA256:
        raise ValueError("Validation configuration does not match the peak checkpoint")
    optimizer = optimizer_config(
        reference_tokens_per_update=training.schedule.tokens_per_update,
        multiplier=1.0,
    ).build(CONTINUATION_TOTAL_UPDATES)
    mesh = training_mesh(training)
    with jax.set_mesh(mesh):
        weights = eqx.filter_jit(Transformer.init)(
            training.model, key=jax.random.key(0)
        )
        exemplar = eqx.filter_jit(
            lambda value: fresh_state(value, optimizer, mp, offload=True)
        )(weights)
        del weights
        restored, clock = restore_dna_checkpoint(
            exemplar,
            PEAK_CHECKPOINT,
            mesh,
            data_seed=0,
            tokenizer_sha256=tokenizer_sha256,
            scientific_config_sha256=PEAK_SCIENTIFIC_CONFIG_SHA256,
            metadata_sha256=PEAK_METADATA_SHA256,
        )
        restored_summary = validate_restored_training_state(restored, clock)
        target = f"{config.checkpoint_root}/step-{clock.updates}"
        if StoragePath(target).exists():
            raise ValueError(f"Validation destination already exists: {target}")
        tracker.log_summary(
            {
                "validation/phase": "saving",
                "validation/source_checkpoint": PEAK_CHECKPOINT,
                "validation/source_step": clock.updates,
                **{
                    f"validation/restored_{key}": value
                    for key, value in restored_summary.items()
                },
            }
        )
        save_dna_checkpoint(
            restored,
            clock,
            target,
            data_seed=0,
            tokenizer_sha256=tokenizer_sha256,
            scientific_config_sha256=output_digest,
            runtime_moe_implementation=training.model.moe_implementation,
            is_temporary=True,
        )
        tracker.log_summary({"validation/phase": "reloading"})
        reloaded, reloaded_clock = restore_dna_checkpoint(
            exemplar,
            target,
            mesh,
            data_seed=0,
            tokenizer_sha256=tokenizer_sha256,
            scientific_config_sha256=output_digest,
        )
        reloaded_summary = validate_restored_training_state(reloaded, reloaded_clock)
        if reloaded_clock != clock:
            raise ValueError("Checkpoint round trip changed the token/data clock")
        errors = {
            key: float(value)
            for key, value in state_error_by_group(restored, reloaded).items()
        }
        max_abs_error = float(state_max_abs_error(restored, reloaded))
        if max_abs_error != 0.0 or any(value != 0.0 for value in errors.values()):
            raise ValueError(f"Checkpoint round trip changed training state: {errors}")
        metadata = json.loads((StoragePath(target) / "metadata.json").read_text())
        if metadata.get("exp586_clock") != dataclasses.asdict(clock):
            raise ValueError(
                "Checkpoint metadata readback changed the token/data clock"
            )
        tracker.log_summary(
            {
                "validation/phase": "complete",
                "validation/checkpoint": target,
                "validation/max_abs_error": max_abs_error,
                **{f"validation/error_{key}": value for key, value in errors.items()},
                **{
                    f"validation/reloaded_{key}": value
                    for key, value in reloaded_summary.items()
                },
            }
        )
        tracker.log({"run_progress": 1.0}, step=1)
    tracker.current_tracker().finish()


def dispatch(config: CheckpointRoundtripConfig) -> None:
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
            entrypoint=Entrypoint.from_callable(run_roundtrip, args=[config]),
            resources=resources,
            processes_per_task=8,
            environment=create_environment(env_vars=env, extras=["gpu"]),
            priority=priority_band_value("batch"),
            timeout=Duration.from_hours(2),
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
    parser.add_argument("--nodes", type=int, choices=[1, 2, 4, 8], required=True)
    parser.add_argument("--validation-id", required=True)
    dispatch(CheckpointRoundtripConfig(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
