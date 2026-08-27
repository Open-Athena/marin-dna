"""Validate the issue #517 strict phyloP recipe on one preemptible H100."""

from __future__ import annotations

import os
from dataclasses import replace

import click
from fray.types import ANY_REGION, ResourceConfig
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.experiment.cli import build_options
from marin.experiment.train import train_lm
from marin.processing.tokenize.tokenize import HfTokenizeConfig, tokenize
from marin.training.training import LevanterCheckpoint

from exp517_functional_specialists.experiment import (
    BATCH_SIZE,
    MARIN_COMMIT,
    MODEL,
    OPTIMIZER,
    SEED,
    SEQUENCE_LENGTH,
    TOKENIZER_PATH,
    TOKENIZER_SHA256,
    TOKENIZER_SOURCE,
    TOKENIZER_SOURCE_REVISION,
    TRAIN_FORMAT,
    DnaTokenizedCache,
    required_env,
    validate_vendored_tokenizer,
)
from exp517_functional_specialists.phylop_uniform_experiment import (
    ARMS,
    PUBLICATION_PRODUCER_COMMIT,
    SOURCE_CONFIG_SHA256,
    SOURCE_PRODUCER_COMMIT,
    PhylopUniformArm,
)

SMOKE_VERSION = "2026.08.27"
SMOKE_TRAIN_STEPS = 3
SMOKE_SAMPLE_COUNT = 16_384
SMOKE_NUM_SHARDS = 8
DEFAULT_H100_CLUSTER = "cw-us-east-02a"
ALLOWED_H100_CLUSTERS = frozenset({"cw-rno2a", "cw-us-east-02a"})
DEFAULT_H100_PER_DEVICE_PARALLELISM = BATCH_SIZE
ALLOWED_H100_PER_DEVICE_PARALLELISM = frozenset({1_024, 2_048, 4_096, 8_192})


def selected_h100_cluster() -> str:
    """Return the explicitly selected production H100 peer."""
    cluster = os.environ.get("EXP517_H100_CLUSTER", DEFAULT_H100_CLUSTER).strip()
    if cluster not in ALLOWED_H100_CLUSTERS:
        raise ValueError(
            "EXP517_H100_CLUSTER must be one of "
            f"{sorted(ALLOWED_H100_CLUSTERS)}, got {cluster!r}"
        )
    return cluster


def selected_per_device_parallelism() -> int:
    """Start with the whole 8,192-sequence batch and allow OOM-only fallback."""
    raw = os.environ.get(
        "EXP517_H100_SMOKE_PDP",
        str(DEFAULT_H100_PER_DEVICE_PARALLELISM),
    ).strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(
            f"EXP517_H100_SMOKE_PDP must be an integer, got {raw!r}"
        ) from error
    if value not in ALLOWED_H100_PER_DEVICE_PARALLELISM:
        raise ValueError(
            "EXP517_H100_SMOKE_PDP must be one of "
            f"{sorted(ALLOWED_H100_PER_DEVICE_PARALLELISM)}, got {value}"
        )
    return value


def validated_coreweave_prefix() -> str:
    """Require the cluster-managed CoreWeave artifact prefix."""
    prefix = required_env("MARIN_PREFIX").rstrip("/")
    if not prefix.startswith("s3://"):
        raise ValueError(
            "MARIN_PREFIX must use CoreWeave-local S3 for the H100 smoke, "
            f"got {prefix!r}"
        )
    return prefix


def smoke_tokenized_dataset(arm: PhylopUniformArm) -> ArtifactStep[DnaTokenizedCache]:
    """Tokenize a representative immutable sample without reading GCS."""
    validate_vendored_tokenizer()
    revision = arm.resolved_revision()
    cluster = selected_h100_cluster()

    def build_config(ctx: StepContext) -> HfTokenizeConfig:
        return HfTokenizeConfig(
            id=arm.hf_repo,
            revision=revision,
            cache_path=ctx.output_path,
            tokenizer=TOKENIZER_PATH,
            format=TRAIN_FORMAT,
            sample_count=SMOKE_SAMPLE_COUNT,
            tags=[
                "dna",
                "marin-dna",
                "exp517",
                "phylop",
                "uniform-grid",
                "h100-smoke",
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
            max_workers=16,
            num_shards=SMOKE_NUM_SHARDS,
            worker_resources=ResourceConfig.with_cpu(
                cpu=2,
                ram="12g",
                disk="20g",
                regions=[ANY_REGION],
                target_cluster=cluster,
                preemptible=True,
            ),
        )

    return ArtifactStep(
        name=f"smoke/inputs/phylop-uniform-{arm.key}-16k-char-bos",
        version=SMOKE_VERSION,
        artifact_type=DnaTokenizedCache,
        run=remote(
            tokenize,
            resources=ResourceConfig.with_cpu(
                cpu=1,
                ram="16g",
                disk="20g",
                regions=[ANY_REGION],
                target_cluster=cluster,
                preemptible=True,
            ),
            env_vars={
                "HF_HUB_DOWNLOAD_TIMEOUT": "120",
                "UV_LOCK_TIMEOUT": "7200",
            },
        ),
        build_config=build_config,
    )


def build_smoke(arm: PhylopUniformArm) -> ArtifactStep[LevanterCheckpoint]:
    """Build a three-step, single-H100 memory and throughput validation."""
    cache = smoke_tokenized_dataset(arm)
    cluster = selected_h100_cluster()
    per_device_parallelism = selected_per_device_parallelism()
    run_id = (
        "dna-exp517-phylop-uniform-0p25b-"
        f"{arm.key}-h100-pdp{per_device_parallelism}-smoke-v1"
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
        name=f"smoke/checkpoints/{run_id}",
        version=SMOKE_VERSION,
        run_id=run_id,
        model=MODEL,
        optimizer=OPTIMIZER,
        datasets={cache: 1.0},
        validation=(),
        init_from=None,
        batch_size=BATCH_SIZE,
        seq_len=SEQUENCE_LENGTH,
        num_train_steps=SMOKE_TRAIN_STEPS,
        z_loss_weight=4.312883184368223e-06,
        evals=None,
        resources=ResourceConfig.with_gpu(
            "H100",
            count=1,
            cpu=8,
            ram="64g",
            disk="128g",
            regions=[ANY_REGION],
            target_cluster=cluster,
            preemptible=True,
        ),
        tensor_parallel_size=1,
        steps_per_eval=SMOKE_TRAIN_STEPS,
        wandb_project=forwarded_env["WANDB_PROJECT"],
        wandb_group="dna-exp517-phylop-uniform-h100-validation",
        tags=[
            "dna",
            "marin-dna",
            "exp517",
            "phylop",
            "uniform-grid",
            "h100-smoke",
            f"region={arm.key}",
            "accelerator=h100",
            "gpu_count=1",
            f"coreweave_cluster={cluster}",
            f"per_device_parallelism={per_device_parallelism}",
            f"hf_revision={arm.resolved_revision()}",
            f"marin_commit={MARIN_COMMIT}",
        ],
        env_vars=forwarded_env,
    )
    base_build_config = step.build_config

    def build_config(ctx: StepContext):
        pod = base_build_config(ctx)
        return replace(
            pod,
            train_config=replace(
                pod.train_config,
                trainer=replace(
                    pod.train_config.trainer,
                    seed=SEED,
                    per_device_parallelism=per_device_parallelism,
                    checkpointer=replace(
                        pod.train_config.trainer.checkpointer,
                        keep=[{"every": SMOKE_TRAIN_STEPS}],
                    ),
                ),
                data_seed=SEED,
                hf_save_steps=SMOKE_TRAIN_STEPS,
            ),
        )

    return replace(step, build_config=build_config)


@click.command(help=__doc__)
@build_options
def main() -> ArtifactStep[LevanterCheckpoint]:
    """Return the isolated CDS H100 smoke artifact."""
    return build_smoke(ARMS["cds"])


if __name__ == "__main__":
    main()
