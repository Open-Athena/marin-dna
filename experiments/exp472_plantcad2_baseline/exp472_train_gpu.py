"""Continue selected exp472 runs from their pre-cooldown CoreWeave checkpoints.

``TRIAL`` selects one provisional first-sweep survivor. ``CLUSTER`` and
``NODES`` select placement without changing the production W&B or checkpoint
identity. Set ``SMOKE=1`` and a unique ``SMOKE_ID`` for a short, separately
named validation run under a one-day CoreWeave TTL path. Omit ``--run`` to
preview the lowered Marin plan.
"""

import os
from dataclasses import dataclass, replace

import click
from fray.types import ResourceConfig
from levanter.layers.attention import AttentionBackend
from levanter.optim.config import AdamConfig
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.experiment.train import train_lm
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.cluster_config import marin_temp_bucket
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.exp472_plantcad2_baseline.common import (
    CACHE_VERSION,
    DEFAULT_GLOBAL_BATCH_SIZE,
    DEFAULT_TRAIN_STEPS,
    DISABLED_WANDB_WATCH,
    EVALUATION_COUNT,
    EXPERIMENT_RELATIVE,
    MAX_EVAL_BATCHES,
    MODEL_CONFIG,
    MODEL_PARAMS,
    PERMANENT_CHECKPOINT_COUNT,
    SEQ_LEN,
    SHUFFLE,
    SWEEP_POINTS_BY_KEY,
    TEMPORARY_CHECKPOINT_INTERVAL,
    TOKENIZED_CACHE_RELATIVE,
    TRAIN_CACHE_NAME,
    SweepPoint,
    augment_reverse_complements,
    env_int,
    existing_plantcad_cache,
    global_batch_size,
    require_marin_prefix,
    required_env,
)
from experiments.exp472_plantcad2_baseline.exp472_sweep_gpu import (
    ALLOWED_NODES,
    ClusterSpec,
    GpuBatchConfig,
    gpu_batch_fit,
    parse_cluster,
    parse_nodes,
)

COREWEAVE_ROOT = "s3://marin-us-east-02a"
TOKENIZED_CACHE = prefix_join(COREWEAVE_ROOT, TOKENIZED_CACHE_RELATIVE)
EXPERIMENT_PREFIX = prefix_join(COREWEAVE_ROOT, EXPERIMENT_RELATIVE)

SOURCE_ARTIFACT_VERSION = "2026.08.20"
SOURCE_CHECKPOINT_STEP = 164_920
SOURCE_RESUME_STEP = SOURCE_CHECKPOINT_STEP + 1
ADDITIONAL_TRAIN_STEPS = DEFAULT_TRAIN_STEPS
TRAIN_STAGE = "s01"
TRAIN_RECIPE_VERSION = "v1"
WANDB_ENTITY = "eric-czech"
WANDB_PROJECT = "marin"
WANDB_GROUP = f"exp472-plantcad2-baseline-train-{TRAIN_STAGE}"

# Provisional selection from completed first-sweep runs on 2026-08-25. The
# operator will revisit this tuple when the remaining first-sweep runs finish.
SOURCE_TRIAL_KEYS = (
    "lr0p0005-wd0p2",
    "lr0p0002-wd0p1",
    "lr0p0001-wd0p8",
    "lr0p0001-wd0p1",
)


@dataclass(frozen=True)
class SourceRun:
    point: SweepPoint
    run_id: str
    artifact_version: str = SOURCE_ARTIFACT_VERSION
    checkpoint_step: int = SOURCE_CHECKPOINT_STEP

    @property
    def resume_step(self) -> int:
        """TrainerState.step restored by the zero-indexed checkpoint."""
        return self.checkpoint_step + 1

    @property
    def artifact_root(self) -> str:
        return prefix_join(
            EXPERIMENT_PREFIX,
            f"checkpoints/{self.run_id}/{self.artifact_version}",
        )


SOURCE_RUNS = {
    key: SourceRun(
        point=SWEEP_POINTS_BY_KEY[key],
        run_id=f"exp472-plantcad2-angiosperm-{key}-v2",
    )
    for key in SOURCE_TRIAL_KEYS
}


@dataclass(frozen=True)
class RunShape:
    additional_steps: int
    end_step: int
    steps_per_eval: int
    permanent_checkpoint_every: int | None
    run_id: str
    checkpoint_name: str
    tags: tuple[str, ...]


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


def _parse_source() -> SourceRun:
    key = required_env("TRIAL").strip().lower()
    try:
        return SOURCE_RUNS[key]
    except KeyError as exc:
        choices = ", ".join(SOURCE_RUNS)
        raise SystemExit(f"TRIAL must be one of: {choices}") from exc


def _training_env() -> dict[str, str]:
    entity = required_env("WANDB_ENTITY")
    project = required_env("WANDB_PROJECT")
    if (entity, project) != (WANDB_ENTITY, WANDB_PROJECT):
        raise ValueError(
            "continuation runs must log to "
            f"{WANDB_ENTITY}/{WANDB_PROJECT}, got {entity}/{project}"
        )
    required_env("HF_TOKEN")
    required_env("WANDB_API_KEY")
    env = {
        "MARIN_PREFIX": required_env("MARIN_PREFIX"),
        "WANDB_ENTITY": entity,
        "WANDB_PROJECT": project,
    }
    if mode := os.environ.get("WANDB_MODE"):
        env["WANDB_MODE"] = mode
    return env


def _source_checkpoint(source: SourceRun) -> ArtifactStep[LevanterCheckpoint]:
    return ArtifactStep[LevanterCheckpoint].adopt(
        (
            "inputs/exp472-training-source/"
            f"{source.point.key}/step-{source.checkpoint_step}"
        ),
        source.artifact_version,
        source=source.artifact_root,
        kind=LevanterCheckpoint,
        config={
            "source_run_id": source.run_id,
            "source_artifact_version": source.artifact_version,
            "source_checkpoint_step": source.checkpoint_step,
        },
    )


def _run_shape(
    source: SourceRun,
    *,
    cluster: str,
    spec: ClusterSpec,
    nodes: int,
    smoke: bool,
) -> RunShape:
    base_id = (
        f"exp472-plantcad2-angiosperm-{source.point.key}-"
        f"train-{TRAIN_STAGE}-{TRAIN_RECIPE_VERSION}"
    )
    if smoke:
        smoke_id = required_env("SMOKE_ID").strip().lower()
        additional_steps = env_int("SMOKE_STEPS", 20)
        run_id = (
            f"{base_id}-smoke-{smoke_id}-{cluster}-{spec.gpu_variant.lower()}-n{nodes}"
        )
        steps_per_eval = SOURCE_RESUME_STEP + additional_steps + 1
        permanent_checkpoint_every = None
    else:
        additional_steps = ADDITIONAL_TRAIN_STEPS
        run_id = base_id
        steps_per_eval = max(
            1,
            (additional_steps + EVALUATION_COUNT - 1) // EVALUATION_COUNT,
        )
        permanent_checkpoint_every = max(
            1,
            (additional_steps + PERMANENT_CHECKPOINT_COUNT - 1)
            // PERMANENT_CHECKPOINT_COUNT,
        )

    end_step = source.resume_step + additional_steps
    tags = (
        "MarinDNA",
        "exp472",
        "plantcad2-baseline",
        "angiosperm-65-genomes",
        "selected-continuation",
        f"train_stage={TRAIN_STAGE}",
        f"source_run={source.run_id}",
        f"source_checkpoint_step={source.checkpoint_step}",
        f"source_artifact_version={source.artifact_version}",
        "initialization=full-trainer-state",
        "augmentation=reverse-complement-p0.5",
        "schedule=constant80-linear20-zero",
        f"lr={source.point.learning_rate:g}",
        f"wd={source.point.weight_decay:g}",
        f"batch={DEFAULT_GLOBAL_BATCH_SIZE}",
        f"additional_steps={additional_steps}",
        f"start_step={source.resume_step}",
        f"end_step={end_step}",
        f"final_checkpoint_step={end_step - 1}",
        f"params={MODEL_PARAMS}",
    )
    if smoke:
        tags = (
            *tags,
            "smoke",
            f"cluster={cluster}",
            f"gpu={spec.gpu_variant}",
            f"nodes={nodes}",
        )
    return RunShape(
        additional_steps=additional_steps,
        end_step=end_step,
        steps_per_eval=steps_per_eval,
        permanent_checkpoint_every=permanent_checkpoint_every,
        run_id=run_id,
        checkpoint_name=f"checkpoints/{run_id}",
        tags=tags,
    )


def _apply_training_overrides(
    step: ArtifactStep[LevanterCheckpoint],
    *,
    source: SourceRun,
    shape: RunShape,
    batch: GpuBatchConfig,
    cluster: str,
    spec: ClusterSpec,
    nodes: int,
    smoke: bool,
) -> ArtifactStep[LevanterCheckpoint]:
    base_build_config = step.build_config

    def build_config(ctx):
        if not ctx.is_fingerprint and ctx.prefix.rstrip(
            "/"
        ) != EXPERIMENT_PREFIX.rstrip("/"):
            raise ValueError(
                f"execution prefix {ctx.prefix!r} must be exactly {EXPERIMENT_PREFIX!r}"
            )
        pod = base_build_config(ctx)
        source_checkpoint_dir = pod.train_config.initialize_from_checkpoint_path
        if not ctx.is_fingerprint and source_checkpoint_dir is None:
            raise ValueError("continuation requires its source checkpoint dependency")
        exact_checkpoint = (
            prefix_join(
                source_checkpoint_dir,
                f"step-{source.checkpoint_step}",
            )
            if source_checkpoint_dir is not None
            else None
        )
        if (
            not ctx.is_fingerprint
            and exact_checkpoint is not None
            and not StoragePath(exact_checkpoint).exists()
        ):
            raise FileNotFoundError(
                f"source checkpoint does not exist: {exact_checkpoint}"
            )

        tracker = replace(
            pod.train_config.trainer.tracker,
            entity=WANDB_ENTITY,
        )
        if not ctx.is_fingerprint and not smoke:
            tracker = replace(
                tracker,
                tags=[
                    *tracker.tags,
                    f"cluster={cluster}",
                    f"gpu={spec.gpu_variant}",
                    f"nodes={nodes}",
                    f"data_parallelism={batch.data_parallelism}",
                    f"tensor_parallelism={batch.tensor_parallelism}",
                    f"per_device_parallelism={batch.per_device_parallelism}",
                    f"gradient_accumulation={batch.gradient_accumulation}",
                ],
            )
        trainer = replace(
            pod.train_config.trainer,
            initialize_from=exact_checkpoint,
            max_eval_batches=1 if smoke else MAX_EVAL_BATCHES,
            tracker=tracker,
            checkpointer=replace(
                pod.train_config.trainer.checkpointer,
                save_interval=TEMPORARY_CHECKPOINT_INTERVAL,
                keep=(
                    [{"every": shape.permanent_checkpoint_every}]
                    if shape.permanent_checkpoint_every is not None
                    else []
                ),
            ),
            watch=DISABLED_WANDB_WATCH,
        )
        if not ctx.is_fingerprint:
            trainer = replace(
                trainer,
                per_device_parallelism=batch.per_device_parallelism,
                per_device_eval_parallelism=batch.per_device_parallelism,
            )

        data = replace(
            pod.train_config.data,
            auto_build_caches=False,
            shuffle=SHUFFLE,
            components={
                name: replace(component, pack=True)
                for name, component in pod.train_config.data.components.items()
            },
            block_cross_document_attention=True,
        )
        data = augment_reverse_complements(data)
        train_config = replace(
            pod.train_config,
            trainer=trainer,
            data=data,
            data_seed=0,
            # TrainerConfig.initialize_from preserves model, AdamW moments,
            # skip-step state, RNG, data position, and absolute trainer step.
            initialize_from_checkpoint_path=None,
            initialize_model_from_checkpoint_path=None,
            hf_save_steps=shape.end_step + 1,
        )
        return replace(pod, train_config=train_config)

    return replace(step, build_config=build_config)


def build_run(
    source: SourceRun,
    *,
    cluster: str,
    spec: ClusterSpec,
    nodes: int,
    smoke: bool,
) -> ArtifactStep[LevanterCheckpoint]:
    batch_size = global_batch_size()
    if batch_size != DEFAULT_GLOBAL_BATCH_SIZE:
        raise ValueError(
            "continuation must preserve the source global batch size "
            f"{DEFAULT_GLOBAL_BATCH_SIZE}, got {batch_size}"
        )
    batch = gpu_batch_fit(spec, nodes, batch_size)
    shape = _run_shape(
        source,
        cluster=cluster,
        spec=spec,
        nodes=nodes,
        smoke=smoke,
    )
    env = _training_env()
    step = train_lm(
        name=shape.checkpoint_name,
        run_id=shape.run_id,
        model=replace(MODEL_CONFIG, attn_backend=AttentionBackend.JAX_FLASH),
        optimizer=AdamConfig(
            learning_rate=source.point.learning_rate,
            weight_decay=source.point.weight_decay,
            warmup=0.1,
            rewarmup=0.0,
            decay=0.2,
            cycle_length=[source.resume_step, shape.additional_steps],
            min_lr_ratio=0.0,
            lr_schedule="linear",
            skip_bad_steps=True,
        ),
        datasets={
            existing_plantcad_cache(
                name=TRAIN_CACHE_NAME,
                version=CACHE_VERSION,
                source=TOKENIZED_CACHE,
            ): 1.0
        },
        validation=(),
        init_from=_source_checkpoint(source),
        batch_size=batch_size,
        seq_len=SEQ_LEN,
        num_train_steps=shape.end_step,
        z_loss_weight=None,
        evals=None,
        resources=ResourceConfig.with_gpu(
            spec.gpu_variant,
            count=spec.gpus_per_node,
            replicas=nodes,
            cpu=spec.cpu,
            ram=spec.ram,
            disk=spec.disk,
        ),
        tensor_parallel_size=batch.tensor_parallelism,
        steps_per_eval=shape.steps_per_eval,
        wandb_project=WANDB_PROJECT,
        wandb_group=WANDB_GROUP,
        tags=shape.tags,
        env_vars=env,
    )
    if smoke:
        step = replace(
            step,
            override_path=marin_temp_bucket(1, f"checkpoints/{shape.run_id}"),
        )
    return _apply_training_overrides(
        step,
        source=source,
        shape=shape,
        batch=batch,
        cluster=cluster,
        spec=spec,
        nodes=nodes,
        smoke=smoke,
    )


@click.command(help=__doc__)
@build_options
def main() -> ArtifactStep[LevanterCheckpoint]:
    require_marin_prefix(EXPERIMENT_PREFIX)
    smoke = _truthy_env("SMOKE")
    source = _parse_source()
    cluster, spec = parse_cluster()
    nodes = parse_nodes()
    if nodes not in ALLOWED_NODES:
        raise AssertionError(f"unsupported node count escaped parser: {nodes}")
    return build_run(
        source,
        cluster=cluster,
        spec=spec,
        nodes=nodes,
        smoke=smoke,
    )


if __name__ == "__main__":
    main()
