"""Launch one new matched 0.25B projection-policy arm for issue #473."""

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

from exp473_center_seeded_projection.formats import DNALmDatasetFormat

MARIN_COMMIT = "6bb4d74694fa185cabf20d037f414235e6a12eed"
TOKENIZER_PATH = "tokenizer"
TOKENIZER_SOURCE = "marin-dna/tokenizer-char-bos"
TOKENIZER_SOURCE_REVISION = "a73e9d9ee636f722b4c378703c9e2997857809b2"
TOKENIZER_SHA256 = {
    "special_tokens_map.json": "02b7b977703736f58dd672a20b0e6159fa10bfc41c9ca721eba0402552e35f2d",
    "tokenizer.json": "d066e668d7ba6ed48640b7a0ad45b5ae05d5dbd612b2d7f91fae1e2473fc93e9",
    "tokenizer_config.json": "4e814edcdb1cb8f408a2cdf951e9be4093bbcd3d12e58883f56aa06eb39fc8c7",
}
SEQUENCE_LENGTH = 256
BATCH_SIZE = 8_192
TRAIN_STEPS = 5_000
SEED = 0
HF_SAVE_STEPS = 500
NATIVE_CHECKPOINT_STEPS = 500
PER_DEVICE_PARALLELISM = 1_024
DATA_VERSION = "2026.08.20"

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
    tokenizer=TOKENIZER_PATH,
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
    """One preregistered full-corpus training arm."""

    key: str
    region: str
    policy: str
    hf_repo: str
    revision: str | None = None
    revision_env: str | None = None

    def resolved_revision(self) -> str:
        if self.revision is not None:
            assert len(self.revision) == 40
            return self.revision
        assert self.revision_env is not None
        revision = required_env(self.revision_env)
        if len(revision) != 40:
            raise ValueError(
                f"{self.revision_env} must be an immutable 40-character HF revision"
            )
        return revision


ARMS = {
    arm.key: arm
    for arm in (
        Arm(
            key="cds_center_1",
            region="cds",
            policy="center_1",
            hf_repo="marin-dna/vertebrate-v1-issue473-center1-cds",
            revision_env="EXP473_CENTER1_CDS_REVISION",
        ),
        Arm(
            key="enhancer_full_window",
            region="ccre_enhancer_centered",
            policy="full_window",
            hf_repo=(
                "marin-dna/vertebrate-v1-issue473-fullwindow-ccre-enhancer-centered"
            ),
            revision_env="EXP473_FULLWINDOW_ENHANCER_REVISION",
        ),
        Arm(
            key="enhancer_center_1",
            region="ccre_enhancer_centered",
            policy="center_1",
            hf_repo=("marin-dna/vertebrate-v1-issue473-center1-ccre-enhancer-centered"),
            revision_env="EXP473_CENTER1_ENHANCER_REVISION",
        ),
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


def selected_arm() -> Arm:
    key = required_env("EXP473_ARM")
    try:
        return ARMS[key]
    except KeyError as exc:
        raise ValueError(
            f"EXP473_ARM must be one of {sorted(ARMS)}, got {key!r}"
        ) from exc


def validate_vendored_tokenizer() -> None:
    """Fail if any byte of the exact #417 tokenizer has changed."""
    for filename, expected in TOKENIZER_SHA256.items():
        path = Path(TOKENIZER_PATH, filename)
        if not path.is_file():
            raise FileNotFoundError(f"missing vendored tokenizer file {path}")
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        assert observed == expected, f"{path} sha256 changed: {observed} != {expected}"


def tokenized_dataset(arm: Arm) -> ArtifactStep[DnaTokenizedCache]:
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
                "exp473",
                f"region={arm.region}",
                f"policy={arm.policy}",
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
    cache = tokenized_dataset(arm)
    run_id = f"dna-exp473-0p25b-{arm.key}-v1"
    forwarded_env = {
        "WANDB_API_KEY": required_env("WANDB_API_KEY"),
        "WANDB_ENTITY": required_env("WANDB_ENTITY"),
        "WANDB_PROJECT": required_env("WANDB_PROJECT"),
        "MARIN_PREFIX": required_env("MARIN_PREFIX"),
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
            "v5p-8",
            cpu=16,
            ram="56g",
            disk="100g",
            regions=["us-east5"],
        ),
        tensor_parallel_size=1,
        steps_per_eval=HF_SAVE_STEPS,
        wandb_project=forwarded_env["WANDB_PROJECT"],
        wandb_group="dna-exp473-center-seeded-projection",
        tags=[
            "dna",
            "marin-dna",
            "exp473",
            "projection-policy",
            f"region={arm.region}",
            f"policy={arm.policy}",
            f"hf_repo={arm.hf_repo}",
            f"hf_revision={arm.resolved_revision()}",
            f"marin_commit={MARIN_COMMIT}",
            "seed=0",
            "batch=8192",
            "steps=5000",
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
    """Return the one new arm selected by EXP473_ARM."""
    return build_training(selected_arm())


if __name__ == "__main__":
    main()
