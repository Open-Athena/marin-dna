"""Launch one strict phyloP-control uniform-grid 0.25B specialist for issue #517."""

from __future__ import annotations

import string
from dataclasses import dataclass, replace

import click
from fray.types import ResourceConfig
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.experiment.cli import build_options
from marin.experiment.train import train_lm
from marin.processing.tokenize.tokenize import HfTokenizeConfig, tokenize
from marin.training.training import LevanterCheckpoint

from exp517_functional_specialists.experiment import (
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

PUBLICATION_PRODUCER_COMMIT = "fbc8968b14415b2722e7bcc4afaf95051acd6638"
SOURCE_PRODUCER_COMMIT = "2162b6aa8299a9748eeb8031318b49072bb8c3fc"
SOURCE_CONFIG_SHA256 = (
    "94d512050de327f96fda1105ce9c6ae5562944e402802516c7cde54795d8cdd1"
)
DATA_VERSION = "2026.08.27"


@dataclass(frozen=True)
class PhylopUniformArm:
    """One exhaustive strict phyloP-control uniform-grid specialist arm."""

    key: str
    hf_repo: str
    revision: str

    def resolved_revision(self) -> str:
        if not self.hf_repo.startswith("marin-dna/phylop-uniform-v1-"):
            raise ValueError(
                f"{self.key} must use its public strict phyloP-control dataset"
            )
        if (
            self.revision == UNPUBLISHED_REVISION
            or len(self.revision) != 40
            or not set(self.revision) <= set(string.hexdigits)
        ):
            raise ValueError(
                f"{self.key} must pin its published 40-character hexadecimal "
                "Hub revision"
            )
        return self.revision


ARMS = {
    arm.key: arm
    for arm in (
        PhylopUniformArm(
            "cds",
            "marin-dna/phylop-uniform-v1-cds",
            "452a5a3538f22630c3dea94d441ac30216bb28ea",
        ),
        PhylopUniformArm(
            "utr3",
            "marin-dna/phylop-uniform-v1-utr3",
            "2b73d5d9ebda34a361536db5e3d2697b6a1b1d6c",
        ),
        PhylopUniformArm(
            "tss_utr5",
            "marin-dna/phylop-uniform-v1-tss-utr5",
            "5134205d86cd03e7833843d99e947e43e7aa11ac",
        ),
        PhylopUniformArm(
            "ncrna_exon",
            "marin-dna/phylop-uniform-v1-ncrna-exon",
            "54667e7bb49505f463afc147676e880a30c11d89",
        ),
        PhylopUniformArm(
            "enhancer_arm_a",
            "marin-dna/phylop-uniform-v1-enhancer-arm-a",
            "6f879b3747330e2c92e1402ead55cda6621f50ff",
        ),
        PhylopUniformArm(
            "background",
            "marin-dna/phylop-uniform-v1-background",
            "7d84519dccb4286622a14642a82a4f045d93a42c",
        ),
    )
}


def resolved_publication_commit() -> str:
    """Reject graph construction until the publication producer is immutable."""
    if (
        PUBLICATION_PRODUCER_COMMIT == UNPUBLISHED_REVISION
        or len(PUBLICATION_PRODUCER_COMMIT) != 40
        or not set(PUBLICATION_PRODUCER_COMMIT) <= set(string.hexdigits)
    ):
        raise ValueError(
            "PUBLICATION_PRODUCER_COMMIT must be a published 40-character "
            "hexadecimal commit"
        )
    return PUBLICATION_PRODUCER_COMMIT


def training_tags(arm: PhylopUniformArm, tpu_region: str) -> list[str]:
    """Return the fixed recipe and full publication provenance as W&B tags."""
    tags = [
        "dna",
        "marin-dna",
        "exp517",
        "phylop",
        "uniform-grid",
        "strict-selector-control",
        "exhaustive-six-arm",
        f"region={arm.key}",
        f"tpu_region={tpu_region}",
        bounded_wandb_tag("hf_repo", arm.hf_repo),
        f"hf_revision={arm.resolved_revision()}",
        f"publication={resolved_publication_commit()}",
        f"source={SOURCE_PRODUCER_COMMIT}",
        f"marin_commit={MARIN_COMMIT}",
        "seed=0",
        "batch=8192",
        "steps=5000",
    ]
    if any(not 1 <= len(tag) <= 64 for tag in tags):
        raise ValueError(f"invalid W&B tags for {arm.key}: {tags}")
    return tags


def selected_arm() -> PhylopUniformArm:
    key = required_env("EXP517_PHYLOP_ARM")
    try:
        return ARMS[key]
    except KeyError as exc:
        raise ValueError(
            f"EXP517_PHYLOP_ARM must be one of {sorted(ARMS)}, got {key!r}"
        ) from exc


def tokenized_dataset(arm: PhylopUniformArm) -> ArtifactStep[DnaTokenizedCache]:
    """Tokenize one public immutable strict-control Hub dataset."""
    validate_vendored_tokenizer()
    revision = arm.resolved_revision()
    publication_commit = resolved_publication_commit()

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
                f"region={arm.key}",
                f"hf_revision={revision}",
                f"publication={publication_commit}",
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
        name=f"inputs/phylop-uniform-{arm.key}-char-bos",
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


def build_training(arm: PhylopUniformArm) -> ArtifactStep[LevanterCheckpoint]:
    """Build one independently resumable strict-control training artifact."""
    cache = tokenized_dataset(arm)
    run_id = f"dna-exp517-phylop-uniform-0p25b-{arm.key}-v1"
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
        name=f"checkpoints/{run_id}",
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
        wandb_group="dna-exp517-phylop-uniform-specialists",
        tags=training_tags(arm, tpu_region),
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
    """Return the strict phyloP-control arm selected by the environment."""
    return build_training(selected_arm())


if __name__ == "__main__":
    main()
