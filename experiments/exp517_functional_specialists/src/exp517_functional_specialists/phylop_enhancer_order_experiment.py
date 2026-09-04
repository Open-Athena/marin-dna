"""Launch the strict-phyloP enhancer order-exposure control for issue #517."""

from __future__ import annotations

import string
from dataclasses import replace

import click
from fray.types import ResourceConfig
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
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
    PER_DEVICE_PARALLELISM,
    SEED,
    SEQUENCE_LENGTH,
    TOKENIZER_PATH,
    TOKENIZER_SHA256,
    TOKENIZER_SOURCE,
    TOKENIZER_SOURCE_REVISION,
    TRAIN_FORMAT,
    TRAIN_STEPS,
    UNPUBLISHED_REVISION,
    DnaTokenizedCache,
    bounded_wandb_tag,
    required_env,
    selected_tpu_preemptible,
    selected_tpu_ram,
    selected_tpu_region,
    selected_tpu_variants,
    validate_vendored_tokenizer,
    validated_marin_prefix,
)

HF_REPO = "marin-dna/phylop-uniform-v1-enhancer-arm-a-vertebrate-order"
HF_REVISION = "6a592fffcdd155d19e6c8e0986eab606aab19606"
PUBLICATION_PRODUCER_COMMIT = "90b86f6426c919470f0eb26e1b1aa2cab6a261ed"
PUBLICATION_CONFIG_SHA256 = (
    "a5d7ff16ecc2b4574e4803e4858392ffa00aefba17da1e155c4859355ad7b437"
)
SOURCE_PRODUCER_COMMIT = "2162b6aa8299a9748eeb8031318b49072bb8c3fc"
SOURCE_CONFIG_SHA256 = (
    "94d512050de327f96fda1105ce9c6ae5562944e402802516c7cde54795d8cdd1"
)
SOURCE_ROWS = 7_876_044
TRAIN_ROWS = 15_719_320
VALIDATION_ROWS = 16_384
EFFECTIVE_ROW_EPOCHS = 40_960_000 / TRAIN_ROWS
DATA_VERSION = "2026.09.04"
RUN_ID = "dna-exp517-phylop-uniform-0p25b-enhancer-order-v1"


def _resolved_hex_revision(name: str, value: str) -> str:
    if (
        value == UNPUBLISHED_REVISION
        or len(value) != 40
        or not set(value) <= set(string.hexdigits)
    ):
        raise ValueError(f"{name} must be a published 40-character hexadecimal value")
    return value


def resolved_hf_revision() -> str:
    """Return the immutable public dataset revision or fail closed."""
    if HF_REPO != "marin-dna/phylop-uniform-v1-enhancer-arm-a-vertebrate-order":
        raise ValueError("the order-exposure control must use its public dataset")
    return _resolved_hex_revision("HF_REVISION", HF_REVISION)


def resolved_publication_commit() -> str:
    """Return the immutable publication producer commit or fail closed."""
    return _resolved_hex_revision(
        "PUBLICATION_PRODUCER_COMMIT", PUBLICATION_PRODUCER_COMMIT
    )


def training_tags(tpu_region: str) -> list[str]:
    """Return the fixed recipe and order-control provenance as W&B tags."""
    tags = [
        "dna",
        "marin-dna",
        "exp517",
        "phylop",
        "uniform-grid",
        "strict-selector-control",
        "enhancer-arm-a",
        "one-per-ncbi-order",
        "order-exposure-control",
        "region=enhancer_arm_a",
        f"tpu_region={tpu_region}",
        bounded_wandb_tag("hf_repo", HF_REPO),
        f"hf_revision={resolved_hf_revision()}",
        f"publication={resolved_publication_commit()}",
        f"source={SOURCE_PRODUCER_COMMIT}",
        f"marin_commit={MARIN_COMMIT}",
        "seed=0",
        "batch=8192",
        "steps=5000",
    ]
    if any(not 1 <= len(tag) <= 64 for tag in tags):
        raise ValueError(f"invalid W&B tags: {tags}")
    return tags


def tokenized_dataset() -> ArtifactStep[DnaTokenizedCache]:
    """Tokenize the public immutable one-per-order enhancer dataset."""
    validate_vendored_tokenizer()
    revision = resolved_hf_revision()
    publication_commit = resolved_publication_commit()

    def build_config(ctx: StepContext) -> HfTokenizeConfig:
        return HfTokenizeConfig(
            id=HF_REPO,
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
                "enhancer-arm-a",
                "one-per-ncbi-order",
                f"hf_revision={revision}",
                f"publication={publication_commit}",
                f"publication_config={PUBLICATION_CONFIG_SHA256}",
                f"source={SOURCE_PRODUCER_COMMIT}",
                f"source_config={SOURCE_CONFIG_SHA256}",
                f"tokenizer_source={TOKENIZER_SOURCE}",
                f"tokenizer_revision={TOKENIZER_SOURCE_REVISION}",
                *[
                    f"{name}_sha256={digest}"
                    for name, digest in sorted(TOKENIZER_SHA256.items())
                ],
            ],
            max_workers=128,
            num_shards=64,
            worker_resources=ResourceConfig.with_cpu(cpu=2, ram="12g", disk="20g"),
        )

    return ArtifactStep(
        name="inputs/phylop-uniform-enhancer-order-char-bos",
        version=DATA_VERSION,
        artifact_type=DnaTokenizedCache,
        run=remote(
            tokenize,
            resources=ResourceConfig.with_cpu(cpu=1, ram="16g", disk="20g"),
            env_vars={
                "HF_HUB_DOWNLOAD_TIMEOUT": "120",
                "UV_LOCK_TIMEOUT": "7200",
            },
        ),
        build_config=build_config,
    )


def build_training() -> ArtifactStep[LevanterCheckpoint]:
    """Build the independently resumable one-per-order enhancer control."""
    cache = tokenized_dataset()
    required_env("WANDB_API_KEY")
    tpu_region = selected_tpu_region()
    tpu_variants = selected_tpu_variants(tpu_region)
    tpu_ram = selected_tpu_ram()
    tpu_preemptible = selected_tpu_preemptible()
    marin_prefix = validated_marin_prefix(tpu_region)
    forwarded_env = {
        "WANDB_ENTITY": required_env("WANDB_ENTITY"),
        "WANDB_PROJECT": required_env("WANDB_PROJECT"),
        "MARIN_PREFIX": marin_prefix,
        "EXP517_TPU_REGION": tpu_region,
        "EXP517_TPU_PREEMPTIBLE": str(tpu_preemptible).lower(),
        "HF_HUB_DOWNLOAD_TIMEOUT": "120",
        "UV_LOCK_TIMEOUT": "7200",
    }
    step = train_lm(
        name=f"checkpoints/{RUN_ID}",
        version=DATA_VERSION,
        run_id=RUN_ID,
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
        resources=ResourceConfig.with_tpu(
            tpu_variants,
            cpu=16,
            ram=tpu_ram,
            disk="100g",
            regions=[tpu_region],
            preemptible=tpu_preemptible,
        ),
        tensor_parallel_size=1,
        steps_per_eval=HF_SAVE_STEPS,
        wandb_project=forwarded_env["WANDB_PROJECT"],
        wandb_group="dna-exp517-phylop-enhancer-order-control",
        tags=training_tags(tpu_region),
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
                    per_device_parallelism=PER_DEVICE_PARALLELISM,
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
def main() -> ArtifactStep[LevanterCheckpoint]:
    """Return the fixed one-per-order enhancer training graph."""
    return build_training()


if __name__ == "__main__":
    main()
