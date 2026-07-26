"""Issue #402: train the 45.9M ortholog-RAG model with a 2M-token batch.

This scratch run matches the historical 256-token experiments' effective
2,097,152-token optimizer batch while retaining the frozen 2,048-token RAG
documents. It trains for 30,000 updates (62.9B tokens) and permanently saves
native optimizer state and Hugging Face exports every 1,000 updates.
"""

from dataclasses import replace

from fray.types import ResourceConfig
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import experiment_main
from marin.experiment.train import EvalSuite, train_lm

from launch import (
    HF_SAVE_EVERY,
    MENDELIAN_TRAITS_RAG_255,
    MODEL,
    RAG_EVAL_EVERY,
    SEQ_LEN,
    TRAIN_HOST_CPU,
    TRAIN_REGIONS,
    TRAIN_TPU,
    online_eval_enabled,
    resolve_completed_adamh,
)
from launch import RAGTokenizedCache as _BaseRAGTokenizedCache
from launch import rag_tokenized_dataset as _base_rag_tokenized_dataset

LARGE_BATCH_SIZE = 1_024
TRAIN_STEPS_LARGE_BATCH = 30_000
ACTUAL_TOKENS_LARGE_BATCH = TRAIN_STEPS_LARGE_BATCH * LARGE_BATCH_SIZE * SEQ_LEN
OPTIMIZER_LARGE_BATCH = resolve_completed_adamh(
    batch_size=LARGE_BATCH_SIZE,
    tokens=ACTUAL_TOKENS_LARGE_BATCH,
)
PER_DEVICE_PARALLELISM = 256
TRAIN_DEVICE_COUNT = 4
GRADIENT_ACCUMULATION_STEPS = LARGE_BATCH_SIZE // (
    PER_DEVICE_PARALLELISM * TRAIN_DEVICE_COUNT
)
assert GRADIENT_ACCUMULATION_STEPS == 1
NATIVE_CHECKPOINT_EVERY = 1_000
TRAIN_HOST_RAM_LARGE_BATCH = "56g"
CHECKPOINT_NAME = "checkpoints/dna-exp402-rag-h640-p46m-b2m-30k"
RUN_ID = "dna-exp402-rag-h640-p46M-B2M-30K-scratch"


class RAGTokenizedCache(_BaseRAGTokenizedCache):
    """Preserve the executable-local type recorded by the frozen cache."""


def rag_tokenized_dataset() -> ArtifactStep[RAGTokenizedCache]:
    """Reuse the frozen token cache with its recorded executable-local type."""
    return replace(
        _base_rag_tokenized_dataset(),
        artifact_type=RAGTokenizedCache,
    )


def build() -> ArtifactStep:
    """Assemble the scratch 45.9M, 2M-token-batch training run."""
    dataset = rag_tokenized_dataset()
    training = train_lm(
        name=CHECKPOINT_NAME,
        model=MODEL,
        optimizer=OPTIMIZER_LARGE_BATCH,
        datasets={dataset: 1.0},
        batch_size=LARGE_BATCH_SIZE,
        seq_len=SEQ_LEN,
        num_train_steps=TRAIN_STEPS_LARGE_BATCH,
        z_loss_weight=1.0e-7,
        evals=(
            EvalSuite(tasks=(MENDELIAN_TRAITS_RAG_255,), every=RAG_EVAL_EVERY)
            if online_eval_enabled()
            else None
        ),
        resources=ResourceConfig.with_tpu(
            TRAIN_TPU,
            cpu=TRAIN_HOST_CPU,
            ram=TRAIN_HOST_RAM_LARGE_BATCH,
            disk="100g",
            regions=TRAIN_REGIONS,
        ),
        steps_per_eval=RAG_EVAL_EVERY,
        wandb_project="marin",
        wandb_group="dna-exp402-v1",
        run_id=RUN_ID,
        tags=(
            "dna",
            "dna-exp402",
            "rag",
            "qwen3",
            "46M",
            "scratch",
            "large-batch",
            "batch-2m-tokens",
            "30k",
        ),
    )
    original_build_config = training.build_config

    def build_config_with_persistent_checkpoints(ctx: StepContext):
        pod_config = original_build_config(ctx)
        trainer = pod_config.train_config.trainer
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
                hf_save_steps=HF_SAVE_EVERY,
            ),
        )

    return replace(training, build_config=build_config_with_persistent_checkpoints)


if __name__ == "__main__":
    experiment_main(build)()
