"""Bounded full-model H100 validation of augmented batches and perf5 state."""

from __future__ import annotations

import argparse
import gc
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jmp
import numpy as np
from exp586_moe.config import SOURCE_CHECKPOINT, SOURCE_METADATA_DIGEST, d768_config
from exp586_moe.data_parallel import local_model, replicated_mesh
from exp586_moe.perf5_optimizer import (
    assert_partitioned_optimizer_state,
    match_gradient_placement,
    model_matches_fp32_master,
    partition_expert_optimizer_state,
    sharded_expert_ns,
)
from exp586_moe.schedule import optimizer_config, set_token_learning_rates
from exp586_moe.state import (
    fresh_state,
    state_all_finite,
    state_error_by_group,
    state_max_abs_error,
)
from experiments.grug.dispatch import _forwarded_env_vars
from experiments.grug.moe_hero_ep.train import (
    _apply_hero_ep_runtime_defaults,
    _drop_metrics,
)
from experiments.grug.moe_hero_ep.weights import restore_weights
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
from rigging.timing import Duration

from exp593_dna.annotations import load_annotations
from exp593_dna.batches import distributed_batch
from exp593_dna.corpus import open_corpus
from exp593_dna.inputs import INPUTS_MANIFEST_SHA256, PreparedInputs
from exp593_dna.pipeline import AugmentationConfig, Augmenter, load_encoder
from exp593_dna.schedule import (
    REFERENCE_TOKENS_PER_UPDATE,
    InputTokenSchedule,
    TokenClock,
)
from exp593_dna.state import restore_training_checkpoint, save_training_checkpoint
from exp593_dna.train_step import make_train_step

WANDB_BASE_URL = "https://forge.coreweave.com/api/wandb"


@dataclass(frozen=True)
class SmokeConfig:
    run_id: str
    cluster: str
    nodes: int = 1

    def __post_init__(self) -> None:
        if self.nodes not in (1, 2, 4, 8) or self.cluster not in (
            "cw-rno2a",
            "cw-us-east-02a",
        ):
            raise ValueError("Unapproved H100 smoke placement")
        if not self.run_id.startswith("dna-exp593-") or "smoke" not in self.run_id:
            raise ValueError("Validation requires a separate exp593 smoke identity")

    @property
    def checkpoint_root(self) -> str:
        return f"s3://marin-us-east-02a/MarinDNA/exp593_augmented_dna/validation/{self.run_id}/checkpoints"


def run_smoke(config: SmokeConfig) -> None:
    """Exercise every context bucket, live norms and exact checkpoint roundtrip."""
    os.environ.update(
        WANDB_BASE_URL=WANDB_BASE_URL, WANDB_ENTITY="eric-czech", WANDB_PROJECT="marin"
    )
    mp = jmp.get_policy("params=bfloat16,compute=bfloat16,output=bfloat16")
    trainer = TrainerConfig(
        id=config.run_id,
        seed=0,
        train_batch_size=64,
        num_train_steps=5,
        require_accelerator=True,
        mp=mp,
        use_explicit_mesh_axes=True,
        watch=WatchConfig(interval=0, watch_targets=[]),
        tracker=WandbConfig(
            entity="eric-czech",
            project="marin",
            name=config.run_id,
            id=config.run_id,
            group="dna-exp593-validation",
            background=False,
            save_code=False,
            tags=["dna-exp593", "validation", "H100", config.cluster],
            replicate_path=config.checkpoint_root,
        ),
    )
    trainer.initialize()
    if jax.device_count() != config.nodes * 8 or not all(
        "H100" in d.device_kind for d in jax.devices()
    ):
        raise ValueError("Smoke did not receive the requested H100 gang")
    tracker.log_configuration(config)
    tracker.log_summary({"validation/phase": "loading_inputs"})
    inputs = PreparedInputs.open(Path("/tmp/exp593-inputs"))
    dataset = open_corpus(inputs, "train", 0)
    encoder = load_encoder()
    annotations = load_annotations(inputs.path("annotations/train.parquet"))
    augmenters = {
        bucket: Augmenter(
            encoder,
            replace(
                AugmentationConfig(),
                length_buckets=(bucket,),
                bucket_probabilities=(1.0,),
            ),
            annotations=annotations,
        )
        for bucket in (2048, 4096, 6144, 8192)
    }
    mesh = replicated_mesh()
    model = replace(
        local_model(d768_config("pretrained")),
        compact_input_lookup_ids=(),
        compact_input_lookup_one_hot=False,
    )
    opt_config = optimizer_config(
        reference_tokens_per_update=REFERENCE_TOKENS_PER_UPDATE
    )
    optimizer = match_gradient_placement(
        opt_config.build(5, ns4d_impl=sharded_expert_ns(mesh))
    )
    # Only this bounded validation clock is shortened; peak LR still comes from
    # the complete scientific token horizon and original reference batch.
    schedule = InputTokenSchedule(training_tokens=10_000_000, warmup_tokens=131_072)
    bindings = {
        "inputs_manifest": INPUTS_MANIFEST_SHA256,
        "tokenizer": encoder.digest,
        "source_checkpoint": SOURCE_CHECKPOINT,
        "source_metadata": SOURCE_METADATA_DIGEST,
        "validation": True,
        "augmentation": asdict(AugmentationConfig()),
    }
    # JSON roundtripping makes tuples/lists identical to checkpoint metadata.
    import json

    bindings = json.loads(json.dumps(bindings))
    tracker.log_hyperparameters({"model": asdict(model), "bindings": bindings})

    with jax.set_mesh(mesh):
        tracker.log_summary({"validation/phase": "loading_text_checkpoint"})
        weights = restore_weights(
            SOURCE_CHECKPOINT, SOURCE_METADATA_DIGEST, model, mesh
        )
        state = eqx.filter_jit(
            lambda w: fresh_state(w, optimizer, mp, offload=False, retain_master=True)
        )(weights)
        del weights
        state, partitions = partition_expert_optimizer_state(state, mesh)
        jax.block_until_ready(state)
        assert_partitioned_optimizer_state(state, mesh)
        if not bool(model_matches_fp32_master(state.params, state.master_params)):
            raise ValueError("BF16 source parameters differ from FP32 masters")
        tracker.log_summary(
            {
                "validation/optimizer_partitions": partitions,
                "validation/phase": "updates",
            }
        )
        train_step = make_train_step(optimizer, mp)
        clock = TokenClock()

        def update(
            current: Any, current_clock: TokenClock, bucket: int, *, log: bool = True
        ) -> tuple[Any, TokenClock, str]:
            batch = distributed_batch(
                dataset, augmenters[bucket], mesh, current_clock.examples
            )
            current = replace(
                current,
                opt_state=set_token_learning_rates(
                    current.opt_state, opt_config, schedule, current_clock
                ),
            )
            started = time.monotonic()
            current, metrics, norms = train_step(current, batch.model_batch, True)
            jax.block_until_ready((current, metrics, norms))
            elapsed = time.monotonic() - started
            loss = float(metrics["train/loss"])
            host_norms = {k: float(v) for k, v in norms.items()}
            if (
                not np.isfinite(loss)
                or len(host_norms) != 20
                or not all(np.isfinite(v) for v in host_norms.values())
            ):
                raise ValueError("Nonfinite loss or missing/nonfinite norm scalars")
            routing = _drop_metrics(
                metrics[MOE_DROPPED_ASSIGNMENTS_METRIC],
                metrics[MOE_SENDER_DROPPED_ASSIGNMENTS_METRIC],
                metrics[MOE_RECEIVER_DROPPED_ASSIGNMENTS_METRIC],
                metrics[MOE_SKIPPED_PADDING_ASSIGNMENTS_METRIC],
                metrics[MOE_VALID_ASSIGNMENTS_METRIC],
                batch_size=64,
                sequence_length=bucket,
                top_k=8,
                num_layers=8,
            )
            actual_tokens = batch.counts["dna_tokens"] + batch.counts["text_tokens"]
            if (
                routing[MOE_VALID_ASSIGNMENTS_METRIC] != actual_tokens * 8 * 8
                or routing[MOE_DROPPED_ASSIGNMENTS_METRIC] != 0
            ):
                raise ValueError(
                    "Router dropped real inputs or valid-token accounting differs"
                )
            next_clock = current_clock.advance(**batch.counts)
            if log:
                tracker.log(
                    {
                        "train/loss": loss,
                        "validation/bucket": bucket,
                        "validation/update_seconds_including_compile": elapsed,
                        "validation/actual_input_tokens": next_clock.input_tokens,
                        "validation/scored_targets": next_clock.loss_targets,
                        "validation/fim_examples": next_clock.fim_examples,
                        **host_norms,
                        **routing,
                    },
                    step=next_clock.updates,
                )
                tracker.log_summary({f"validation/bucket_{bucket}/passed": True})
            return current, next_clock, batch.input_fingerprint

        for bucket in augmenters:
            state, clock, _ = update(state, clock, bucket)
        if not bool(state_all_finite(state)):
            raise ValueError("Full optimizer state is nonfinite")
        checkpoint = f"{config.checkpoint_root}/step-{clock.updates}"
        tracker.log_summary({"validation/phase": "checkpoint_roundtrip"})
        save_training_checkpoint(state, clock, checkpoint, bindings)
        replay, replay_clock = restore_training_checkpoint(
            state, checkpoint, mesh, bindings
        )
        error = float(state_max_abs_error(state, replay))
        if error != 0 or replay_clock != clock:
            raise ValueError(f"Full-state checkpoint roundtrip differs: {error}")
        tracker.log_summary({"validation/restore_max_abs_error": error})
        state, clock, fingerprint = update(state, clock, 2048)
        replay, replay_clock, replay_fingerprint = update(
            replay, replay_clock, 2048, log=False
        )
        differences = {
            "validation/replay/" + k: float(v)
            for k, v in state_error_by_group(state, replay).items()
        }
        if (
            clock != replay_clock
            or fingerprint != replay_fingerprint
            or not bool(state_all_finite(replay))
        ):
            raise ValueError("Resumed input/data clock or finite state differs")
        if (
            differences.get("validation/replay/master_params/relative_l2", float("inf"))
            > 1e-3
        ):
            raise ValueError(
                "Resume update changed FP32 parameters beyond the smoke tolerance"
            )
        del replay
        gc.collect()
        final_checkpoint = f"{config.checkpoint_root}/step-{clock.updates}"
        save_training_checkpoint(state, clock, final_checkpoint, bindings)
        tracker.log_summary(
            {
                "validation/phase": "complete",
                "validation/final_checkpoint": final_checkpoint,
                "validation/replay_input_fingerprint": fingerprint,
                **differences,
            }
        )
        tracker.log({"run_progress": 1.0}, step=clock.updates + 1)
    tracker.current_tracker().finish()


def dispatch(config: SmokeConfig) -> None:
    _apply_hero_ep_runtime_defaults(
        inline_watch_enabled=False,
        moe_implementation="scatter",
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
            "CW_KEY_ID": os.environ["CW_KEY_ID"],
            "CW_KEY_SECRET": os.environ["CW_KEY_SECRET"],
            "WANDB_API_KEY": os.environ["WANDB_API_KEY"],
            "WANDB_ENTITY": "eric-czech",
            "WANDB_PROJECT": "marin",
            "WANDB_BASE_URL": WANDB_BASE_URL,
        },
        resources,
    )
    env.update(
        WANDB_BASE_URL=WANDB_BASE_URL, WANDB_ENTITY="eric-czech", WANDB_PROJECT="marin"
    )
    job = current_client().submit(
        JobRequest(
            name=f"validate-{config.run_id}",
            entrypoint=Entrypoint.from_callable(run_smoke, args=[config]),
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
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--cluster", choices=["cw-rno2a", "cw-us-east-02a"], required=True
    )
    parser.add_argument("--nodes", type=int, choices=[1, 2, 4, 8], default=1)
    dispatch(SmokeConfig(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
