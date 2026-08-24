"""Launch one annotation-first 0.25B functional specialist for issue #517."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Self

import click
from fray.types import ResourceConfig
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.config import AdamConfig
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.experiment.cli import build_options
from marin.experiment.train import train_lm
from marin.processing.tokenize.tokenize import (
    HfTokenizeConfig,
    TokenizedCache,
    tokenize,
)
from marin.training.training import LevanterCheckpoint

from exp517_functional_specialists.formats import DNALmDatasetFormat

MARIN_COMMIT = "53b5b33041f742c7f4991223b0085e41ece4c458"
PRODUCER_COMMIT = "d06519abc5dc2c6c14d4c9765057a6a363305ee5"
TOKENIZER_PATH = "tokenizer"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_TOKENIZER_PATH = str(PROJECT_ROOT / TOKENIZER_PATH)
TOKENIZER_SOURCE = "marin-dna/tokenizer-char-bos"
TOKENIZER_SOURCE_REVISION = "a73e9d9ee636f722b4c378703c9e2997857809b2"
TOKENIZER_SHA256 = {
    "special_tokens_map.json": (
        "02b7b977703736f58dd672a20b0e6159fa10bfc41c9ca721eba0402552e35f2d"
    ),
    "tokenizer.json": (
        "d066e668d7ba6ed48640b7a0ad45b5ae05d5dbd612b2d7f91fae1e2473fc93e9"
    ),
    "tokenizer_config.json": (
        "4e814edcdb1cb8f408a2cdf951e9be4093bbcd3d12e58883f56aa06eb39fc8c7"
    ),
}
SEQUENCE_LENGTH = 256
BATCH_SIZE = 8_192
TRAIN_STEPS = 5_000
SEED = 0
HF_SAVE_STEPS = 500
NATIVE_CHECKPOINT_STEPS = 500
PER_DEVICE_PARALLELISM = 1_024
WANDB_MAX_TAG_LENGTH = 64
DATA_VERSION = "2026.08.24"
UNPUBLISHED_REVISION = "UNPUBLISHED"
DEFAULT_TPU_REGION = "us-east5"
DEFAULT_TPU_VARIANT = "v5p-8"
DEFAULT_TPU_RAM = "56g"
DEFAULT_TPU_PREEMPTIBLE = True
ALLOWED_TPU_RAM = frozenset({DEFAULT_TPU_RAM, "96g"})
ALLOWED_TPU_REGIONS = frozenset(
    {DEFAULT_TPU_REGION, "us-central1", "us-east1", "europe-west4"}
)
ALLOWED_TPU_VARIANTS_BY_REGION = {
    "us-east5": frozenset({"v5p-8", "v6e-4"}),
    "us-east1": frozenset({"v6e-4"}),
    "us-central1": frozenset({"v5p-8"}),
    "europe-west4": frozenset({"v5litepod-8", "v5litepod-16", "v6e-4", "v6e-8"}),
}
ARTIFACT_BUCKET_BY_TPU_REGION = {
    "us-east5": "marin-us-east5",
    "us-east1": "marin-us-east1",
    "us-central1": "marin-us-central1",
    "europe-west4": "marin-eu-west4",
}

MODEL = Qwen3Config(
    max_seq_len=SEQUENCE_LENGTH,
    hidden_dim=1_152,
    intermediate_dim=4_608,
    num_layers=12,
    num_heads=9,
    num_kv_heads=9,
    head_dim=128,
    rope=Llama3RotaryEmbeddingsConfig(),
    use_sliding_window=False,
    tie_word_embeddings=False,
    tokenizer=MODEL_TOKENIZER_PATH,
    initializer_range=0.02,
)
OPTIMIZER = AdamConfig(
    learning_rate=0.00430097,
    weight_decay=0.1,
    beta1=0.66756,
    beta2=0.952222,
    epsilon=6.77142e-15,
    max_grad_norm=0.995188,
    warmup=0.1,
    decay=0.2,
    lr_schedule="linear",
    min_lr_ratio=0.0,
)
TRAIN_FORMAT = DNALmDatasetFormat(text_key="sequence", lowercase_weight=0.01)


@dataclass(frozen=True)
class Arm:
    """One preregistered annotation-first specialist arm."""

    key: str
    hf_repo: str
    revision: str

    def resolved_revision(self) -> str:
        if not self.hf_repo.startswith("marin-dna/functional-"):
            raise ValueError(f"{self.key} must use its public marin-dna Hub dataset")
        if self.revision == UNPUBLISHED_REVISION or len(self.revision) != 40:
            raise ValueError(
                f"{self.key} must pin its published 40-character Hub revision"
            )
        return self.revision


ARMS = {
    arm.key: arm
    for arm in (
        Arm("cds", "marin-dna/functional-cds", UNPUBLISHED_REVISION),
        Arm("utr3", "marin-dna/functional-utr3", UNPUBLISHED_REVISION),
        Arm("tss_region", "marin-dna/functional-tss", UNPUBLISHED_REVISION),
        Arm("ncrna", "marin-dna/functional-ncrna", UNPUBLISHED_REVISION),
        Arm("enhancer", "marin-dna/functional-enhancer", UNPUBLISHED_REVISION),
    )
}


class DnaTokenizedCache(TokenizedCache):
    """Tokenized Marin artifact retaining the DNA-specific format."""

    @classmethod
    def raw_load(cls, source: str) -> Self:
        return cls(path=source)

    @property
    def format(self) -> DNALmDatasetFormat:
        raw = self._config.get("format")
        if not isinstance(raw, dict):
            raise TypeError(f"{self.path}: tokenized cache record has no DNA format")
        result = DNALmDatasetFormat(
            text_key=str(raw.get("text_key", "sequence")),
            uppercase_weight=float(raw.get("uppercase_weight", 1.0)),
            lowercase_weight=float(raw.get("lowercase_weight", 1.0)),
        )
        if result != TRAIN_FORMAT:
            raise ValueError(
                f"{self.path}: tokenized cache format {result} != {TRAIN_FORMAT}"
            )
        return result


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"missing required environment variable {name}")
    return value


def selected_tpu_region() -> str:
    """Return the explicit TPU child region, retaining east5 by default."""
    region = os.environ.get("EXP517_TPU_REGION", DEFAULT_TPU_REGION).strip()
    if region not in ALLOWED_TPU_REGIONS:
        raise ValueError(
            "EXP517_TPU_REGION must be one of "
            f"{sorted(ALLOWED_TPU_REGIONS)}, got {region!r}"
        )
    return region


def selected_tpu_variants(tpu_region: str) -> tuple[str, ...]:
    """Return one or more compatible TPU variants for flexible scheduling."""
    raw = os.environ.get("EXP517_TPU_VARIANT", DEFAULT_TPU_VARIANT)
    variants = tuple(part.strip() for part in raw.split(",") if part.strip())
    allowed = ALLOWED_TPU_VARIANTS_BY_REGION[tpu_region]
    if not variants or len(set(variants)) != len(variants):
        raise ValueError(
            "EXP517_TPU_VARIANT must contain unique comma-separated variants, "
            f"got {raw!r}"
        )
    unsupported = set(variants) - allowed
    if unsupported:
        raise ValueError(
            f"EXP517_TPU_VARIANT must be one of {sorted(allowed)} in "
            f"{tpu_region!r}, got {sorted(unsupported)}"
        )
    return variants


def selected_tpu_ram() -> str:
    """Return bounded host RAM; larger RAM is execution-only OOM recovery."""
    ram = os.environ.get("EXP517_TPU_RAM", DEFAULT_TPU_RAM).strip().lower()
    if ram not in ALLOWED_TPU_RAM:
        raise ValueError(
            f"EXP517_TPU_RAM must be one of {sorted(ALLOWED_TPU_RAM)}, got {ram!r}"
        )
    return ram


def selected_tpu_preemptible() -> bool:
    """Return the explicit capacity class, defaulting to preemptible."""
    raw = (
        os.environ.get(
            "EXP517_TPU_PREEMPTIBLE",
            str(DEFAULT_TPU_PREEMPTIBLE),
        )
        .strip()
        .lower()
    )
    if raw not in {"true", "false"}:
        raise ValueError(
            f"EXP517_TPU_PREEMPTIBLE must be 'true' or 'false', got {raw!r}"
        )
    return raw == "true"


def validated_marin_prefix(tpu_region: str) -> str:
    """Require the artifact bucket to be local to the TPU region."""
    prefix = required_env("MARIN_PREFIX").rstrip("/")
    expected_bucket = ARTIFACT_BUCKET_BY_TPU_REGION[tpu_region]
    expected_prefix = f"gs://{expected_bucket}"
    if prefix != expected_prefix and not prefix.startswith(f"{expected_prefix}/"):
        raise ValueError(
            f"MARIN_PREFIX must use {expected_bucket!r} for TPU region "
            f"{tpu_region!r}, got {prefix!r}"
        )
    return prefix


def bounded_wandb_tag(name: str, value: str) -> str:
    """Retain a recognizable collision-resistant tag within W&B's limit."""
    tag = f"{name}={value}"
    if len(tag) <= WANDB_MAX_TAG_LENGTH:
        return tag
    digest = hashlib.sha256(value.encode()).hexdigest()[:8]
    value_length = WANDB_MAX_TAG_LENGTH - len(name) - len(digest) - 2
    if value_length < 1:
        raise ValueError(f"W&B tag name is too long: {name!r}")
    return f"{name}={value[:value_length]}~{digest}"


def training_tags(arm: Arm) -> list[str]:
    """Return bounded tags while full source provenance stays in the cache."""
    tags = [
        "dna",
        "marin-dna",
        "exp517",
        "annotation-first",
        f"region={arm.key}",
        bounded_wandb_tag("hf_repo", arm.hf_repo),
        f"hf_revision={arm.resolved_revision()}",
        f"producer={PRODUCER_COMMIT}",
        f"marin_commit={MARIN_COMMIT}",
        "seed=0",
        "batch=8192",
        "steps=5000",
    ]
    if any(not 1 <= len(tag) <= WANDB_MAX_TAG_LENGTH for tag in tags):
        raise ValueError(f"invalid W&B tags for {arm.key}: {tags}")
    return tags


def selected_arm() -> Arm:
    key = required_env("EXP517_ARM")
    try:
        return ARMS[key]
    except KeyError as exc:
        raise ValueError(
            f"EXP517_ARM must be one of {sorted(ARMS)}, got {key!r}"
        ) from exc


def validate_vendored_tokenizer() -> None:
    """Fail if any byte of the exact proven tokenizer has changed."""
    for filename, expected in TOKENIZER_SHA256.items():
        path = Path(TOKENIZER_PATH, filename)
        if not path.is_file():
            raise FileNotFoundError(f"missing vendored tokenizer file {path}")
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        assert observed == expected, f"{path} sha256 changed: {observed} != {expected}"


def tokenized_dataset(arm: Arm) -> ArtifactStep[DnaTokenizedCache]:
    """Tokenize one public immutable Hub dataset into Marin's cache format."""
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
                f"region={arm.key}",
                f"hf_revision={revision}",
                f"producer={PRODUCER_COMMIT}",
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
        name=f"inputs/{arm.key}-char-bos",
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


def build_training(arm: Arm) -> ArtifactStep[LevanterCheckpoint]:
    """Build one independently resumable specialist training artifact."""
    cache = tokenized_dataset(arm)
    run_id = f"dna-exp517-0p25b-{arm.key}-v1"
    tpu_region = selected_tpu_region()
    tpu_variants = selected_tpu_variants(tpu_region)
    tpu_ram = selected_tpu_ram()
    tpu_preemptible = selected_tpu_preemptible()
    marin_prefix = validated_marin_prefix(tpu_region)
    forwarded_env = {
        "WANDB_API_KEY": required_env("WANDB_API_KEY"),
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
        wandb_group="dna-exp517-functional-specialists",
        tags=training_tags(arm),
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
    """Return the one specialist selected by ``EXP517_ARM``."""
    return build_training(selected_arm())


if __name__ == "__main__":
    main()
