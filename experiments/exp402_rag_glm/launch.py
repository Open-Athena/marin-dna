"""Issue #402: train the first 46M fixed-layout ortholog-RAG gLM.

This experiment uses Marin's current lazy-artifact API. The raw Hugging Face
dataset is tokenized once with the issue-specific fixed-layout preprocessor,
then a scratch Qwen3 model trains for approximately one billion tokens.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from fray.types import ResourceConfig
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.adamh import AdamHConfig
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.experiment.cli import experiment_main
from marin.experiment.train import EvalSuite, train_lm
from marin.processing.tokenize.tokenize import (
    HfTokenizeConfig,
    TokenizedCache,
    tokenize,
)
from marin_dna.levanter.formats import RAGDNALmDatasetFormat
from marin_dna.pipelines.evals.lm_eval.task_configs import MENDELIAN_TRAITS_RAG_255

MARIN_DNA_REVISION = "46ba9f51fca53f50a055ac92041a4ded224b3ea1"
DATASET_REPO = "bolinas-dna/zoonomia-rag-v1-v1"
DATASET_REVISION = "5e6b30cf878b61c99e6432ad8ab7865b18cbe0e7"
TOKENIZER_PATH = "tokenizer"
TOKENIZER_SHA256 = {
    "tokenizer.json": "153a64aabc254c997a634a1a70f2f4d15e96976ccc5df3adeb0e09da2f2fad87",
    "tokenizer_config.json": "f8ce17f8700fd20305a0b6d857c3add59cc44ebe8165456a0db5786f48eededa",
    "special_tokens_map.json": "665d3a4f8b01ce09165d983256c457b1c6dddbb4dc639eab910fb5e17a210238",
}

SEQ_LEN = 2_048
VOCAB_SIZE = 8
TRAIN_BATCH_SIZE = 64
TARGET_TOKENS = 1_000_000_000
TRAIN_STEPS = round(TARGET_TOKENS / (TRAIN_BATCH_SIZE * SEQ_LEN))
ACTUAL_TOKENS = TRAIN_STEPS * TRAIN_BATCH_SIZE * SEQ_LEN
RAG_EVAL_EVERY = 1_000
TRAIN_TPU = ("v6e-4", "v5p-8")
TRAIN_REGIONS = ("us-east5",)
TRAIN_HOST_CPU = 16
TRAIN_HOST_RAM = "64g"

DATASET_ARTIFACT_VERSION = "2026.07.24"
CHECKPOINT_NAME = "checkpoints/dna-exp402-rag-h640-p46m-1b"
RUN_ID = "dna-exp402-rag-h640-p46M-1B"

# This is the existing 46M MarinDNA scaling rung, extended from 256 to 2,048
# positions and initialized from scratch because the old checkpoint has neither
# the longer context nor the new [SEQ] vocabulary entry.
MODEL = Qwen3Config(
    max_seq_len=SEQ_LEN,
    hidden_dim=640,
    intermediate_dim=2_560,
    num_layers=7,
    num_heads=5,
    num_kv_heads=5,
    head_dim=128,
    use_sliding_window=False,
    rope=Llama3RotaryEmbeddingsConfig(),
    tie_word_embeddings=False,
    tokenizer=TOKENIZER_PATH,
)


def resolve_completed_adamh(
    *, batch_size: int = TRAIN_BATCH_SIZE, tokens: int = TARGET_TOKENS
) -> AdamHConfig:
    """Resolve and pin the Complete(d)-inspired AdamH transfer for this run."""
    assert batch_size > 0
    assert tokens > 0
    reference_batch_size = 64
    reference_tokens = 2.5e9
    scaling_ratio = (batch_size * reference_tokens) / (reference_batch_size * tokens)
    learning_rate = min(
        0.01,
        0.00630 * (batch_size / reference_batch_size) ** 0.5 * (reference_tokens / tokens) ** 0.3,
    )
    adam_lr = min(0.01, 0.000656 * scaling_ratio**0.5)
    epsilon = 1.85e-8 * (1.0 / scaling_ratio) ** 0.5
    beta2 = max(0.9, min(0.9999, 0.9999 ** (batch_size / reference_batch_size)))
    return AdamHConfig(
        learning_rate=learning_rate,
        adam_lr=adam_lr,
        min_lr_ratio=0.0,
        warmup=0.1,
        beta1=0.9,
        beta2=beta2,
        epsilon=epsilon,
        max_grad_norm=0.1,
        lr_schedule="linear",
        decay=0.2,
        nesterov=False,
    )


OPTIMIZER = resolve_completed_adamh()


def validate_vendored_tokenizer() -> None:
    """Fail if any committed tokenizer byte differs from the frozen recipe."""
    for filename, expected in TOKENIZER_SHA256.items():
        path = Path(TOKENIZER_PATH, filename)
        assert path.is_file(), f"missing vendored tokenizer file {path}"
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        assert observed == expected, f"{path} sha256 changed: {observed} != {expected}"


class RAGTokenizedCache(TokenizedCache):
    """Tokenized cache that preserves the registered RAG format on reload.

    Marin's generic ``TokenizedCache.format`` currently reconstructs custom
    formats as plain text. This experiment-local type deliberately restores the
    fixed RAG format so training reads the cached ``loss_weight`` arrays.
    """

    @property
    def format(self) -> RAGDNALmDatasetFormat:
        record = self.record
        assert record is not None, f"missing artifact record at {self.path}"
        config = record.config or {}
        assert config.get("tokenizer") == TOKENIZER_PATH
        serialized_format = config.get("format")
        assert isinstance(serialized_format, dict)
        assert serialized_format.get("text_key") == "seq"
        assert serialized_format.get("uppercase_weight") == 1.0
        assert serialized_format.get("lowercase_weight") == 1.0
        return RAGDNALmDatasetFormat()


def rag_tokenized_dataset() -> ArtifactStep[RAGTokenizedCache]:
    """Return the immutable training/validation token-cache build."""
    validate_vendored_tokenizer()

    def build_config(ctx: StepContext) -> HfTokenizeConfig:
        return HfTokenizeConfig(
            id=DATASET_REPO,
            revision=DATASET_REVISION,
            cache_path=ctx.output_path,
            tokenizer=TOKENIZER_PATH,
            format=RAGDNALmDatasetFormat(),
            tags=[
                "dna",
                "rag",
                "exp402",
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
        name="dna-exp402-tokenize-rag",
        resources=ResourceConfig.with_cpu(cpu=4, ram="16g", disk="32g"),
        env_vars={"HF_HUB_DOWNLOAD_TIMEOUT": "120", "UV_LOCK_TIMEOUT": "7200"},
        pip_dependency_groups=[],
    )
    return ArtifactStep(
        name="datasets/dna-exp402-rag-tokenized",
        version=DATASET_ARTIFACT_VERSION,
        artifact_type=RAGTokenizedCache,
        run=run_tokenize,
        build_config=build_config,
    )


def build() -> ArtifactStep:
    """Assemble the tokenization dependency and scratch 46M training run."""
    dataset = rag_tokenized_dataset()
    return train_lm(
        name=CHECKPOINT_NAME,
        model=MODEL,
        optimizer=OPTIMIZER,
        datasets={dataset: 1.0},
        batch_size=TRAIN_BATCH_SIZE,
        seq_len=SEQ_LEN,
        num_train_steps=TRAIN_STEPS,
        z_loss_weight=1.0e-7,
        evals=EvalSuite(tasks=(MENDELIAN_TRAITS_RAG_255,), every=RAG_EVAL_EVERY),
        resources=ResourceConfig.with_tpu(
            TRAIN_TPU,
            cpu=TRAIN_HOST_CPU,
            ram=TRAIN_HOST_RAM,
            disk="100g",
            regions=TRAIN_REGIONS,
        ),
        steps_per_eval=500,
        wandb_project="marin",
        wandb_group="dna-exp402-v1",
        run_id=RUN_ID,
        tags=("dna", "dna-exp402", "rag", "qwen3", "46M", "scratch"),
    )


if __name__ == "__main__":
    experiment_main(build)()
