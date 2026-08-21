"""Train the preregistered CDS full-window random-validation control."""

from __future__ import annotations

from dataclasses import replace

import click
from fray.types import ResourceConfig
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.experiment.cli import build_options
from marin.experiment.train import train_lm
from marin.processing.tokenize.tokenize import HfTokenizeConfig, tokenize
from marin.training.training import LevanterCheckpoint

from exp473_center_seeded_projection.experiment import (
    BATCH_SIZE,
    HF_SAVE_STEPS,
    MARIN_COMMIT,
    MODEL,
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

CONTROL_VERSION = "2026.08.21"
DATASET_REPO = "marin-dna/vertebrate-v1-issue473-fullwindow-cds-random-val"
DATASET_REVISION = "7ef0bc9fcff17efc5792af92d8da34176617dd13"
RUN_ID = "dna-exp473-0p25b-cds-fullwindow-random-val-v1"
WANDB_GROUP = "dna-exp473-validation-damage-control"
VALIDATION_ROWS = 16_384
VALIDATION_SEED = 42
VALIDATION_STEPS = 500


def resolved_dataset_revision() -> str:
    """Return the immutable public dataset revision or fail before graph creation."""
    revision = DATASET_REVISION
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError(
            "DATASET_REVISION must pin the public random-validation dataset "
            "to a 40-character lowercase hexadecimal revision"
        )
    return revision


def control_tags() -> list[str]:
    """Return W&B tags that identify the single preregistered control."""
    revision = resolved_dataset_revision()
    return [
        "dna",
        "marin-dna",
        "exp473",
        "validation-control",
        "region=cds",
        "policy=full_window",
        "validation_split=random_uniform_original_pre_rc",
        f"validation_rows={VALIDATION_ROWS}",
        f"validation_seed={VALIDATION_SEED}",
        bounded_wandb_tag("hf_repo", DATASET_REPO),
        f"hf_revision={revision}",
        f"marin_commit={MARIN_COMMIT}",
        f"seed={SEED}",
        f"batch={BATCH_SIZE}",
        f"steps={TRAIN_STEPS}",
    ]


def tokenized_control_dataset() -> ArtifactStep[DnaTokenizedCache]:
    """Tokenize the public train and validation splits as one Marin component."""
    validate_vendored_tokenizer()
    revision = resolved_dataset_revision()

    def build_config(ctx: StepContext) -> HfTokenizeConfig:
        return HfTokenizeConfig(
            id=DATASET_REPO,
            revision=revision,
            cache_path=ctx.output_path,
            tokenizer=TOKENIZER_PATH,
            format=TRAIN_FORMAT,
            tags=[
                "dna",
                "marin-dna",
                "exp473",
                "validation-control",
                "region=cds",
                "policy=full_window",
                "validation_split=random_uniform_original_pre_rc",
                f"validation_rows={VALIDATION_ROWS}",
                f"validation_seed={VALIDATION_SEED}",
                f"hf_revision={revision}",
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
        name="inputs/cds-fullwindow-random-validation-char-bos",
        version=CONTROL_VERSION,
        artifact_type=DnaTokenizedCache,
        run=remote(
            tokenize,
            resources=ResourceConfig.with_cpu(cpu=1, ram="16g", disk="20g"),
            pip_dependency_groups=["cpu"],
            env_vars={
                "HF_HUB_DOWNLOAD_TIMEOUT": "120",
                "UV_LOCK_TIMEOUT": "7200",
            },
        ),
        build_config=build_config,
    )


def build_random_validation_training() -> ArtifactStep[LevanterCheckpoint]:
    """Build the single matched training graph on preemptible TPU capacity."""
    cache = tokenized_control_dataset()
    tpu_region = selected_tpu_region()
    tpu_variants = selected_tpu_variants(tpu_region)
    tpu_ram = selected_tpu_ram()
    if not selected_tpu_preemptible():
        raise ValueError("the random-validation control requires a preemptible TPU")
    marin_prefix = validated_marin_prefix(tpu_region)
    forwarded_env = {
        "WANDB_API_KEY": required_env("WANDB_API_KEY"),
        "WANDB_ENTITY": required_env("WANDB_ENTITY"),
        "WANDB_PROJECT": required_env("WANDB_PROJECT"),
        "MARIN_PREFIX": marin_prefix,
        "EXP473_TPU_REGION": tpu_region,
        "EXP473_TPU_PREEMPTIBLE": "true",
        "HF_HUB_DOWNLOAD_TIMEOUT": "120",
        "UV_LOCK_TIMEOUT": "7200",
    }
    step = train_lm(
        name=f"checkpoints/{RUN_ID}",
        version=CONTROL_VERSION,
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
            preemptible=True,
        ),
        tensor_parallel_size=1,
        steps_per_eval=VALIDATION_STEPS,
        wandb_project=forwarded_env["WANDB_PROJECT"],
        wandb_group=WANDB_GROUP,
        tags=control_tags(),
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
    """Return the single random-validation training control."""
    return build_random_validation_training()


if __name__ == "__main__":
    main()
