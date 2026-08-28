"""Launch one full strict phyloP-control specialist on one preemptible H100."""

from __future__ import annotations

import os
from dataclasses import replace

import click
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from fray.types import ANY_REGION, ResourceConfig
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.experiment.train import train_lm
from marin.processing.tokenize.tokenize import HfTokenizeConfig, tokenize
from marin.training.training import LevanterCheckpoint, TrainLmOnPodConfig

from exp517_functional_specialists.experiment import (
    BATCH_SIZE,
    HF_SAVE_STEPS,
    MARIN_COMMIT,
    MODEL,
    MODEL_TOKENIZER_PATH,
    NATIVE_CHECKPOINT_STEPS,
    OPTIMIZER,
    SEED,
    SEQUENCE_LENGTH,
    TOKENIZER_PATH,
    TOKENIZER_SHA256,
    TOKENIZER_SOURCE,
    TOKENIZER_SOURCE_REVISION,
    TRAIN_FORMAT,
    TRAIN_STEPS,
    DnaTokenizedCache,
    bounded_wandb_tag,
    required_env,
    validate_vendored_tokenizer,
)
from exp517_functional_specialists.phylop_uniform_experiment import (
    ARMS,
    DATA_VERSION,
    PUBLICATION_PRODUCER_COMMIT,
    SOURCE_CONFIG_SHA256,
    SOURCE_PRODUCER_COMMIT,
    PhylopUniformArm,
)
from exp517_functional_specialists.phylop_uniform_h100_smoke import (
    selected_h100_cluster,
    validated_coreweave_prefix,
)

FULL_H100_PER_DEVICE_PARALLELISM = 1_024
FULL_H100_NUM_SHARDS = 64
FULL_H100_LOCAL_TOKENIZER_WORKERS = 16


def selected_full_per_device_parallelism() -> int:
    """Return the one-H100 microbatch selected after the full-run OOM."""
    raw = os.environ.get(
        "EXP517_H100_PDP",
        str(FULL_H100_PER_DEVICE_PARALLELISM),
    ).strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"EXP517_H100_PDP must be an integer, got {raw!r}") from error
    if value != FULL_H100_PER_DEVICE_PARALLELISM:
        raise ValueError(
            "EXP517_H100_PDP must equal the selected one-H100 value "
            f"{FULL_H100_PER_DEVICE_PARALLELISM}, got {value}"
        )
    return value


def selected_h100_arm() -> PhylopUniformArm:
    """Return the strict-control arm selected for the H100 path."""
    key = required_env("EXP517_H100_ARM")
    try:
        return ARMS[key]
    except KeyError as error:
        raise ValueError(
            f"EXP517_H100_ARM must be one of {sorted(ARMS)}, got {key!r}"
        ) from error


def full_h100_training_tags(
    arm: PhylopUniformArm,
    cluster: str,
    per_device_parallelism: int,
) -> list[str]:
    """Return fixed scientific and H100 execution lineage tags."""
    tags = [
        "dna",
        "marin-dna",
        "exp517",
        "phylop",
        "uniform-grid",
        "strict-selector-control",
        "exhaustive-six-arm",
        f"region={arm.key}",
        "accelerator=h100",
        "gpu_count=1",
        f"coreweave_cluster={cluster}",
        f"per_device_parallelism={per_device_parallelism}",
        bounded_wandb_tag("hf_repo", arm.hf_repo),
        f"hf_revision={arm.resolved_revision()}",
        f"publication={PUBLICATION_PRODUCER_COMMIT}",
        f"source={SOURCE_PRODUCER_COMMIT}",
        f"marin_commit={MARIN_COMMIT}",
        "seed=0",
        "batch=8192",
        "steps=5000",
    ]
    if any(not 1 <= len(tag) <= 64 for tag in tags):
        raise ValueError(f"invalid H100 W&B tags for {arm.key}: {tags}")
    return tags


def full_h100_tokenized_dataset(
    arm: PhylopUniformArm,
) -> ArtifactStep[DnaTokenizedCache]:
    """Tokenize the complete immutable arm into CoreWeave-local S3."""
    validate_vendored_tokenizer()
    revision = arm.resolved_revision()

    def build_config(ctx: StepContext) -> HfTokenizeConfig:
        return HfTokenizeConfig(
            id=arm.hf_repo,
            revision=revision,
            cache_path=ctx.output_path,
            tokenizer=TOKENIZER_PATH,
            format=TRAIN_FORMAT,
            tags=[
                "dna",
                "marin-dna",
                "exp517",
                "phylop",
                "uniform-grid",
                "strict-selector-control",
                "h100-full",
                f"region={arm.key}",
                f"hf_revision={revision}",
                f"publication={PUBLICATION_PRODUCER_COMMIT}",
                f"source={SOURCE_PRODUCER_COMMIT}",
                f"source_config={SOURCE_CONFIG_SHA256}",
                f"tokenizer_source={TOKENIZER_SOURCE}",
                f"tokenizer_revision={TOKENIZER_SOURCE_REVISION}",
                *[
                    f"{name}_sha256={digest}"
                    for name, digest in sorted(TOKENIZER_SHA256.items())
                ],
            ],
            max_workers=FULL_H100_LOCAL_TOKENIZER_WORKERS,
            num_shards=FULL_H100_NUM_SHARDS,
            worker_resources=ResourceConfig.with_cpu(
                cpu=2,
                ram="12g",
                disk="20g",
                regions=[ANY_REGION],
                preemptible=True,
            ),
        )

    return ArtifactStep(
        name=f"h100/inputs/phylop-uniform-{arm.key}-char-bos",
        version=DATA_VERSION,
        artifact_type=DnaTokenizedCache,
        run=tokenize_with_local_workers,
        build_config=build_config,
    )


def tokenize_with_local_workers(config: HfTokenizeConfig) -> None:
    """Run Zephyr workers inside one explicitly sized CoreWeave CPU task.

    CoreWeave's bare CPU callable images do not contain ``cloudpickle`` and the
    pinned Fray actor-group path does not attach a uv environment to CPU actors.
    The top-level coordinator is already running the locked project environment,
    so a local Fray client keeps all tokenization actors inside that task while
    preserving the maintained Marin tokenizer implementation and cache format.
    """
    client = LocalClient(max_threads=FULL_H100_LOCAL_TOKENIZER_WORKERS + 2)
    iris_task_id = os.environ.pop("IRIS_TASK_ID", None)
    try:
        with set_current_client(client):
            tokenize(config)
    finally:
        client.shutdown()
        if iris_task_id is not None:
            os.environ["IRIS_TASK_ID"] = iris_task_id


def build_full_h100_training(
    arm: PhylopUniformArm,
) -> ArtifactStep[LevanterCheckpoint]:
    """Build one full independently resumable strict-control H100 artifact."""
    cache = full_h100_tokenized_dataset(arm)
    cluster = selected_h100_cluster()
    per_device_parallelism = selected_full_per_device_parallelism()
    run_id = (
        "dna-exp517-phylop-uniform-0p25b-"
        f"{arm.key}-h100-pdp{per_device_parallelism}-v1"
    )
    required_env("WANDB_API_KEY")
    forwarded_env = {
        "WANDB_ENTITY": required_env("WANDB_ENTITY"),
        "WANDB_PROJECT": required_env("WANDB_PROJECT"),
        "MARIN_PREFIX": validated_coreweave_prefix(),
        "EXP517_H100_CLUSTER": cluster,
        "EXP517_H100_PREEMPTIBLE": "true",
        "HF_HUB_DOWNLOAD_TIMEOUT": "120",
        "UV_LOCK_TIMEOUT": "7200",
    }
    step = train_lm(
        name=f"h100/checkpoints/{run_id}",
        version=DATA_VERSION,
        run_id=run_id,
        model=MODEL,
        optimizer=OPTIMIZER,
        datasets={cache: 1.0},
        validation=(),
        init_from=None,
        batch_size=BATCH_SIZE,
        seq_len=SEQUENCE_LENGTH,
        num_train_steps=TRAIN_STEPS,
        z_loss_weight=4.312883184368223e-06,
        evals=None,
        resources=ResourceConfig.with_gpu(
            "H100",
            count=1,
            cpu=8,
            ram="64g",
            disk="128g",
            regions=[ANY_REGION],
            preemptible=True,
        ),
        tensor_parallel_size=1,
        steps_per_eval=HF_SAVE_STEPS,
        wandb_project=forwarded_env["WANDB_PROJECT"],
        wandb_group="dna-exp517-phylop-uniform-h100-specialists",
        tags=full_h100_training_tags(arm, cluster, per_device_parallelism),
        env_vars=forwarded_env,
    )
    base_build_config = step.build_config

    def build_config(ctx: StepContext) -> TrainLmOnPodConfig:
        pod = base_build_config(ctx)
        return replace(
            pod,
            train_config=replace(
                pod.train_config,
                data=replace(
                    pod.train_config.data,
                    tokenizer=MODEL_TOKENIZER_PATH,
                ),
                trainer=replace(
                    pod.train_config.trainer,
                    seed=SEED,
                    per_device_parallelism=per_device_parallelism,
                    checkpointer=replace(
                        pod.train_config.trainer.checkpointer,
                        keep=[{"every": NATIVE_CHECKPOINT_STEPS}],
                    ),
                ),
                data_seed=SEED,
                hf_save_steps=HF_SAVE_STEPS,
            ),
        )

    return replace(step, build_config=build_config)


@click.command(help=__doc__)
@build_options
def main() -> ArtifactStep[DnaTokenizedCache] | ArtifactStep[LevanterCheckpoint]:
    """Return the selected full H100 training or tokenization-only artifact."""
    arm = selected_h100_arm()
    tokenize_only = os.environ.get("EXP517_H100_TOKENIZE_ONLY", "false").strip().lower()
    if tokenize_only == "true":
        return full_h100_tokenized_dataset(arm)
    if tokenize_only != "false":
        raise ValueError(
            "EXP517_H100_TOKENIZE_ONLY must be true or false, "
            f"got {tokenize_only!r}"
        )
    return build_full_h100_training(arm)


if __name__ == "__main__":
    main()
