"""Issue #417: matched mammals-only versus combined-vertebrate CDS training.

Both Qwen3-0.25B arms inherit the exp353 model, optimizer, batch, step horizon,
seed, validation cadence, and checkpoint cadence.  The only experimental axis
is the projected CDS corpus.  Source letter case is semantic repeat masking:
lowercase targets receive 1% loss weight in both training and validation.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import replace
from pathlib import Path

from fray.types import ResourceConfig
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.config import AdamConfig
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.experiment.cli import experiment_main
from marin.experiment.train import train_lm
from marin.processing.tokenize.tokenize import (
    HfTokenizeConfig,
    TokenizedCache,
    tokenize,
)
from marin_dna.levanter.formats import DNALmDatasetFormat

MARIN_DNA_REVISION = "eaac2efffb73d33b87ba75bcf5521809af74fec7"
PROJECTION_PIPELINE_REVISION = "06549d8f7f3ba76151b9c54a5e52d3e3f4402a2d"

ARMS = ("mammals_only", "combined_vertebrates")
ARM_ENV = "EXP417_ARMS"
DATASET_REPOS = {
    "mammals_only": "bolinas-dna/vertebrate-v1-cds_mammals_only",
    "combined_vertebrates": "bolinas-dna/vertebrate-v1-cds",
}
DATASET_REVISION_ENVS = {
    "mammals_only": "EXP417_CDS_MAMMALS_REVISION",
    "combined_vertebrates": "EXP417_CDS_COMBINED_REVISION",
}
RUN_IDS = {
    "mammals_only": "dna-exp417-cds-mammals-only-p255m-b2m-5k",
    "combined_vertebrates": "dna-exp417-cds-combined-vertebrates-p255m-b2m-5k",
}

TOKENIZER_PATH = "tokenizer"
TOKENIZER_SOURCE = "marin-dna/tokenizer-char-bos"
TOKENIZER_SOURCE_REVISION = "a73e9d9ee636f722b4c378703c9e2997857809b2"
TOKENIZER_SHA256 = {
    "special_tokens_map.json": "02b7b977703736f58dd672a20b0e6159fa10bfc41c9ca721eba0402552e35f2d",
    "tokenizer.json": "d066e668d7ba6ed48640b7a0ad45b5ae05d5dbd612b2d7f91fae1e2473fc93e9",
    "tokenizer_config.json": "4e814edcdb1cb8f408a2cdf951e9be4093bbcd3d12e58883f56aa06eb39fc8c7",
}

DNA_BASE_SEQ_LEN = 255
SEQ_LEN = 256
VOCAB_SIZE = 7
REPEAT_MASK_LOSS_WEIGHT = 0.01
DATA_FORMAT = DNALmDatasetFormat(
    text_key="sequence",
    uppercase_weight=1.0,
    lowercase_weight=REPEAT_MASK_LOSS_WEIGHT,
)

HIDDEN_DIM = 1_152
INTERMEDIATE_DIM = 4_608
NUM_LAYERS = 12
NUM_HEADS = 9
HEAD_DIM = 128
INITIALIZER_RANGE = 0.02
MODEL = Qwen3Config(
    max_seq_len=SEQ_LEN,
    hidden_dim=HIDDEN_DIM,
    intermediate_dim=INTERMEDIATE_DIM,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    num_kv_heads=NUM_HEADS,
    head_dim=HEAD_DIM,
    use_sliding_window=False,
    rope=Llama3RotaryEmbeddingsConfig(),
    tie_word_embeddings=False,
    tokenizer=TOKENIZER_PATH,
    initializer_range=INITIALIZER_RANGE,
)

TRAIN_BATCH_SIZE = 8_192
TRAIN_STEPS = 5_000
ACTUAL_TOKENS = TRAIN_BATCH_SIZE * TRAIN_STEPS * SEQ_LEN
SEED = 0

# Exact standard-Adam recipe from exp353.  Current Marin retains the same
# implementation at levanter.optim.config.AdamConfig.
LEARNING_RATE = 0.00430097
BETA1 = 0.66756
BETA2 = 0.952222
EPSILON = 6.77142e-15
MAX_GRAD_NORM = 0.995188
Z_LOSS_WEIGHT = 4.312883184368223e-06
OPTIMIZER = AdamConfig(
    learning_rate=LEARNING_RATE,
    weight_decay=0.1,
    beta1=BETA1,
    beta2=BETA2,
    epsilon=EPSILON,
    max_grad_norm=MAX_GRAD_NORM,
    warmup=0.1,
    decay=0.2,
    lr_schedule="linear",
    min_lr_ratio=0.0,
)

VALIDATION_EVERY = 500
NATIVE_CHECKPOINT_EVERY = 500
HF_SAVE_EVERY = 500
PER_DEVICE_PARALLELISM = 1_024
WANDB_GROUP = "dna-exp417-v1"

TRAIN_TPU = ("v6e-4", "v5p-8")
TRAIN_REGIONS = ("us-east5",)
TRAIN_HOST_CPU = 16
TRAIN_HOST_RAM = "56g"
TRAIN_DISK = "100g"

DATASET_ARTIFACT_VERSION = "2026.08.01"
_SHA40 = re.compile(r"[0-9a-f]{40}")


def selected_arms() -> tuple[str, ...]:
    """Return the requested arm subset, preserving canonical order."""
    raw = os.environ.get(ARM_ENV)
    if raw is None:
        return ARMS
    requested = tuple(item.strip() for item in raw.split(","))
    assert requested and all(requested), f"{ARM_ENV} must name one or both arms"
    unknown = sorted(set(requested) - set(ARMS))
    assert not unknown, f"unknown {ARM_ENV} values {unknown}; expected {ARMS}"
    assert len(set(requested)) == len(requested), f"{ARM_ENV} contains duplicate arms"
    return tuple(arm for arm in ARMS if arm in requested)


def dataset_revision(arm: str) -> str:
    """Read and validate the immutable Hugging Face revision for one arm."""
    assert arm in ARMS
    env_name = DATASET_REVISION_ENVS[arm]
    revision = os.environ.get(env_name, "")
    assert _SHA40.fullmatch(revision), (
        f"{env_name} must be the 40-character Hugging Face commit for "
        f"{DATASET_REPOS[arm]!r}; got {revision!r}"
    )
    return revision


def validate_vendored_tokenizer() -> None:
    """Fail if a committed tokenizer byte differs from the frozen recipe."""
    for filename, expected in TOKENIZER_SHA256.items():
        path = Path(TOKENIZER_PATH, filename)
        assert path.is_file(), f"missing vendored tokenizer file {path}"
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        assert observed == expected, f"{path} sha256 changed: {observed} != {expected}"


class VertebrateCDSTokenizedCache(TokenizedCache):
    """Reload a token cache with its case-aware DNA loss format intact."""

    @property
    def format(self) -> DNALmDatasetFormat:
        record = self.record
        assert record is not None, f"missing artifact record at {self.path}"
        config = record.config or {}
        assert config.get("tokenizer") == TOKENIZER_PATH
        serialized_format = config.get("format")
        assert isinstance(serialized_format, dict)
        assert serialized_format.get("text_key") == "sequence"
        assert serialized_format.get("uppercase_weight") == 1.0
        assert serialized_format.get("lowercase_weight") == REPEAT_MASK_LOSS_WEIGHT
        return DATA_FORMAT


def tokenized_dataset(arm: str) -> ArtifactStep[VertebrateCDSTokenizedCache]:
    """Tokenize both train and validation splits of one immutable CDS corpus."""
    assert arm in ARMS
    validate_vendored_tokenizer()
    repo = DATASET_REPOS[arm]
    revision = dataset_revision(arm)

    def build_config(ctx: StepContext) -> HfTokenizeConfig:
        return HfTokenizeConfig(
            id=repo,
            revision=revision,
            cache_path=ctx.output_path,
            tokenizer=TOKENIZER_PATH,
            format=DATA_FORMAT,
            tags=[
                "dna",
                "cds",
                "exp417",
                f"arm={arm}",
                f"source-revision={revision}",
                f"projection-pipeline={PROJECTION_PIPELINE_REVISION}",
                f"repeat-mask-lowercase-weight={REPEAT_MASK_LOSS_WEIGHT}",
                *[
                    f"{filename}-sha256={digest}"
                    for filename, digest in sorted(TOKENIZER_SHA256.items())
                ],
            ],
            max_workers=32,
            worker_resources=ResourceConfig.with_cpu(cpu=2, ram="8g", disk="16g"),
            levanter_batch_size=2_048,
        )

    run_tokenize = remote(
        tokenize,
        name=f"dna-exp417-tokenize-cds-{arm}",
        resources=ResourceConfig.with_cpu(cpu=4, ram="16g", disk="32g"),
        env_vars={"HF_HUB_DOWNLOAD_TIMEOUT": "120", "UV_LOCK_TIMEOUT": "7200"},
        pip_dependency_groups=[],
    )
    return ArtifactStep(
        name=f"datasets/dna-exp417-cds-{arm}-tokenized",
        version=DATASET_ARTIFACT_VERSION,
        artifact_type=VertebrateCDSTokenizedCache,
        run=run_tokenize,
        build_config=build_config,
    )


def build_arm(arm: str) -> ArtifactStep:
    """Build one scratch run while holding every non-dataset choice fixed."""
    assert arm in ARMS
    dataset = tokenized_dataset(arm)
    run_id = RUN_IDS[arm]
    training = train_lm(
        name=f"checkpoints/{run_id}",
        model=MODEL,
        optimizer=OPTIMIZER,
        datasets={dataset: 1.0},
        batch_size=TRAIN_BATCH_SIZE,
        seq_len=SEQ_LEN,
        num_train_steps=TRAIN_STEPS,
        z_loss_weight=Z_LOSS_WEIGHT,
        evals=None,
        resources=ResourceConfig.with_tpu(
            TRAIN_TPU,
            cpu=TRAIN_HOST_CPU,
            ram=TRAIN_HOST_RAM,
            disk=TRAIN_DISK,
            regions=TRAIN_REGIONS,
        ),
        steps_per_eval=VALIDATION_EVERY,
        wandb_project="marin",
        wandb_group=WANDB_GROUP,
        run_id=run_id,
        tags=(
            "dna",
            "dna-exp417",
            "cds",
            "qwen3",
            "255m",
            "scratch",
            f"arm={arm}",
            f"repeat-mask-lowercase-weight={REPEAT_MASK_LOSS_WEIGHT}",
        ),
    )
    original_build_config = training.build_config

    def build_config_with_persistent_checkpoints(ctx: StepContext):
        pod_config = original_build_config(ctx)
        trainer = pod_config.train_config.trainer
        assert trainer.seed == SEED
        return replace(
            pod_config,
            train_config=replace(
                pod_config.train_config,
                trainer=replace(
                    trainer,
                    per_device_parallelism=PER_DEVICE_PARALLELISM,
                    checkpointer=replace(
                        trainer.checkpointer,
                        keep=[{"every": NATIVE_CHECKPOINT_EVERY}],
                    ),
                ),
                data_seed=SEED,
                hf_save_steps=HF_SAVE_EVERY,
            ),
        )

    return replace(training, build_config=build_config_with_persistent_checkpoints)


def build() -> dict[str, ArtifactStep]:
    """Return the selected independent arm handles."""
    return {arm: build_arm(arm) for arm in selected_arms()}


if __name__ == "__main__":
    experiment_main(build)()
