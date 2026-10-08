"""Qualify a larger H100 placement from an immutable DNA peak checkpoint."""

import argparse
import dataclasses
import gc
import hashlib
import json
import math
import time
from typing import Any

import equinox as eqx
import jax
import jmp
import numpy as np
from jax.experimental import multihost_utils
from levanter import tracker
from levanter.callbacks.watch import WatchConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from rigging.filesystem.storage_path import StoragePath

from exp582_moe.corpus import distributed_batch, open_dataset
from exp582_moe.data import (
    pretrained_dna_tokenizer,
    scratch_tokenizer,
    tokenizer_digest,
)
from exp582_moe.gpu_smoke import SmokeConfig, dispatch
from exp582_moe.routing_probe import routing_diagnostics
from exp582_moe.schedule import optimizer_config, set_token_learning_rates
from exp582_moe.state import (
    fresh_state,
    restore_dna_checkpoint,
    save_dna_checkpoint,
    state_all_finite,
)
from exp582_moe.tracking import wandb_history_offset
from exp582_moe.train import (
    BATCH_SIZE,
    TOKENS_PER_UPDATE,
    TOTAL_UPDATES,
    TrainingConfig,
    ValidationSample,
    check_clock,
    config_json,
    evaluate,
    primary_json,
    scientific_config,
    training_mesh,
)
from experiments.grug.moe_hero_ep.model import Transformer
from experiments.grug.moe_hero_ep.train import (
    MasterParamMode,
    _make_train_step,
    grug_trainer_mesh_config,
)


@dataclasses.dataclass(frozen=True)
class PlacementSmokeConfig(SmokeConfig):
    nodes: int = 16
    source_metadata_sha256: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.nodes not in (8, 16):
            raise ValueError("Placement comparison requires 64 or 128 H100s")
        if len(self.source_metadata_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.source_metadata_sha256
        ):
            raise ValueError("Placement validation requires a pinned metadata SHA-256")

    @property
    def source_config(self) -> TrainingConfig:
        return TrainingConfig(
            self.condition,
            1.0 if self.condition == "pretrained" else 2.0,
            self.cluster,
            self.nodes,
        )

    @property
    def source_checkpoint(self) -> str:
        config = self.source_config
        return f"{config.revision_root}/step-{config.schedule.peak_update}"


def verify_source(config: PlacementSmokeConfig, tokenizer_sha256: str) -> str:
    """Read only the permanent source and require its exact scientific binding."""
    blob = (StoragePath(config.source_checkpoint) / "metadata.json").read_bytes()
    if hashlib.sha256(blob).hexdigest() != config.source_metadata_sha256:
        raise ValueError("Placement source metadata digest mismatch")
    metadata = json.loads(blob)
    binding = config_json(scientific_config(config.source_config, tokenizer_sha256))
    digest = hashlib.sha256(binding.encode()).hexdigest()
    stored = (
        StoragePath(config.source_config.revision_root) / "training.json"
    ).read_text()
    if (
        stored != binding
        or metadata.get("scientific_config_sha256") != digest
        or metadata.get("is_temporary") is not False
        or metadata.get("step") != config.source_config.schedule.peak_update
    ):
        raise ValueError("Placement source scientific binding or permanence mismatch")
    return digest


def local_state_digest(state: Any) -> str:
    """Hash one rank's addressable shards, without gathering full model arrays."""
    digest = hashlib.sha256()
    for value in jax.tree.leaves(state):
        digest.update(str((value.shape, value.dtype)).encode())
        for shard in value.addressable_shards:
            digest.update(str(shard.index).encode())
            digest.update(np.asarray(shard.data).tobytes())
    return digest.hexdigest()


def run_placement_smoke(config: PlacementSmokeConfig) -> None:
    """Restore, train, save and exactly restore on a separate smoke identity."""
    source = config.source_config
    mp = jmp.get_policy("params=bfloat16,compute=bfloat16,output=bfloat16")
    trainer = TrainerConfig(
        id=config.run_id,
        seed=source.seed,
        train_batch_size=BATCH_SIZE,
        num_train_steps=TOTAL_UPDATES,
        require_accelerator=True,
        mp=mp,
        mesh=grug_trainer_mesh_config(source.context_axis_size),
        use_explicit_mesh_axes=True,
        watch=WatchConfig(interval=0, watch_targets=[]),
        tracker=WandbConfig(
            entity="eric-czech",
            project="marin",
            name=config.run_id,
            group="exp582-plantcad2-moe-validation",
            tags=["dna-exp582", "validation", "H100", "placement", config.condition],
            save_code=False,
            background=False,
            replicate_path=config.checkpoint_root,
        ),
    )
    trainer.initialize()
    assert jax.device_count() == config.nodes * 8
    assert all("H100" in d.device_kind for d in jax.devices())
    log_offset = wandb_history_offset()
    tracker.log_configuration(config)
    tracker.log_summary({"validation/phase": "loading"})
    tokenizer = (
        pretrained_dna_tokenizer()
        if config.condition == "pretrained"
        else scratch_tokenizer()
    )
    tokenizer_sha256 = tokenizer_digest(tokenizer)
    digest = primary_json(lambda: verify_source(config, tokenizer_sha256))
    opt_config = optimizer_config(
        reference_tokens_per_update=TOKENS_PER_UPDATE,
        multiplier=source.lr_multiplier,
    )
    optimizer = opt_config.build(TOTAL_UPDATES)
    dataset = open_dataset("train", source.seed)
    mesh = training_mesh(source)
    tracker.log_hyperparameters(
        {
            "source_checkpoint": config.source_checkpoint,
            "scientific_config_sha256": digest,
            "batch_size": BATCH_SIZE,
            "context_axis_size": source.context_axis_size,
            "expert_axis_size": 8,
        }
    )
    with jax.set_mesh(mesh):
        state = eqx.filter_jit(
            lambda: fresh_state(
                Transformer.init(source.model, key=jax.random.key(source.seed)),
                optimizer,
                mp,
                offload=True,
            )
        )()
        state, clock = restore_dna_checkpoint(
            state,
            config.source_checkpoint,
            mesh,
            data_seed=source.seed,
            tokenizer_sha256=tokenizer_sha256,
        )
        check_clock(clock)
        assert clock.updates == source.schedule.peak_update
        tracker.log_summary(
            {
                "validation/phase": "evaluating",
                "validation/source_update": clock.updates,
            }
        )
        baseline = evaluate(state, ValidationSample(), tokenizer, mesh)
        tracker.log_summary({"validation/source/" + k: v for k, v in baseline.items()})
        if baseline["eval/drop_fraction"] != 0:
            raise ValueError("Expert drops in restored-source evaluation")
        train_step = _make_train_step(
            optimizer,
            mp,
            z_loss_weight=1e-4,
            ema_beta=None,
            offload_opt_state=True,
            master_param_mode=MasterParamMode.FP32_PINNED_HOST,
            watch_config=WatchConfig(
                watch_targets=["grads", "updates"],
                include_per_parameter_norms=False,
                split_scan_layers=False,
                include_histograms=False,
            ),
        )
        for update in range(1, 27):
            tracker.log_summary({"validation/phase": "training"})
            batch = distributed_batch(
                dataset, tokenizer, mesh, clock.examples, BATCH_SIZE, augment=True
            )
            state = dataclasses.replace(
                state,
                opt_state=set_token_learning_rates(
                    state.opt_state, opt_config, source.schedule, clock
                ),
            )
            start = time.monotonic()
            state, metrics, watch = train_step(state, batch)
            jax.block_until_ready((state, metrics, watch))
            elapsed = time.monotonic() - start
            clock = clock.advance(
                input_tokens=TOKENS_PER_UPDATE,
                loss_targets=BATCH_SIZE * 8191,
                examples=BATCH_SIZE,
            )
            values = routing_diagnostics(
                metrics, batch_size=BATCH_SIZE, model_config=source.model
            )
            values.update({k: float(v) for k, v in (watch or {}).items()})
            tracker.log(
                {
                    **values,
                    "validation/update": update,
                    "validation/step_seconds": elapsed,
                    "train/step": clock.updates,
                    "train/input_tokens": clock.input_tokens,
                    "run_progress": update / 27,
                },
                step=log_offset + update,
            )
            if (
                not all(math.isfinite(float(v)) for v in values.values())
                or values["moe/drop_fraction"] != 0
            ):
                raise ValueError("Nonfinite training metrics or expert drops")
            check_clock(clock)
            if update == 25:
                tracker.log_summary({"validation/phase": "checkpointing"})
                if not bool(state_all_finite(state)):
                    raise ValueError("Nonfinite checkpoint state")
                checkpoint = f"{config.checkpoint_root}/step-{clock.updates}"
                save_dna_checkpoint(
                    state,
                    clock,
                    checkpoint,
                    data_seed=source.seed,
                    tokenizer_sha256=tokenizer_sha256,
                    scientific_config_sha256=digest,
                )
                before = local_state_digest(state)
                template = jax.tree.map(
                    lambda a: jax.ShapeDtypeStruct(
                        a.shape, a.dtype, sharding=a.sharding
                    ),
                    state,
                )
                del state
                gc.collect()
                state, restored_clock = restore_dna_checkpoint(
                    template,
                    checkpoint,
                    mesh,
                    data_seed=source.seed,
                    tokenizer_sha256=tokenizer_sha256,
                )
                exact = before == local_state_digest(state) and clock == restored_clock
                if not np.all(multihost_utils.process_allgather(np.asarray(exact))):
                    raise ValueError("Checkpoint round trip changed a rank's state")
                tracker.log_summary(
                    {
                        "validation/restore_exact": True,
                        "validation/checkpoint": checkpoint,
                    }
                )
        tracker.log_summary({"validation/phase": "complete"})
        tracker.log({"run_progress": 1.0}, step=log_offset + 27)
    tracker.current_tracker().finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=["pretrained", "random"], required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--cluster", choices=["cw-rno2a", "cw-us-east-02a"], required=True
    )
    parser.add_argument("--nodes", type=int, choices=[8, 16], default=16)
    parser.add_argument("--source-metadata-sha256", required=True)
    dispatch(
        PlacementSmokeConfig(**vars(parser.parse_args())),
        worker=run_placement_smoke,
        inline_watch_enabled=True,
    )


if __name__ == "__main__":
    main()
