"""Issue #402: train the 103.8M ortholog-RAG model with a 2M-token batch."""

from dataclasses import replace

from fray.types import ResourceConfig
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import experiment_main
from marin.experiment.train import EvalSuite, train_lm

from launch import (
    HF_SAVE_EVERY,
    MENDELIAN_TRAITS_RAG_255,
    RAG_EVAL_EVERY,
    SEQ_LEN,
    TRAIN_HOST_CPU,
    TRAIN_REGIONS,
    TRAIN_TPU,
    online_eval_enabled,
)
from launch_100m import MODEL, TRAIN_HOST_RAM_100M
from launch_large_batch_30k import (
    ACTUAL_TOKENS_LARGE_BATCH,
    LARGE_BATCH_SIZE,
    NATIVE_CHECKPOINT_EVERY,
    OPTIMIZER_LARGE_BATCH,
    PER_DEVICE_PARALLELISM,
    TRAIN_STEPS_LARGE_BATCH,
)
from launch_large_batch_30k import RAGTokenizedCache as _BaseRAGTokenizedCache
from launch_large_batch_30k import (
    rag_tokenized_dataset as _base_rag_tokenized_dataset,
)

CHECKPOINT_NAME = "checkpoints/dna-exp402-rag-h768-p104m-b2m-30k"
RUN_ID = "dna-exp402-rag-h768-p104M-B2M-30K-scratch"


class RAGTokenizedCache(_BaseRAGTokenizedCache):
    """Preserve the executable-local type recorded by the frozen cache."""


def rag_tokenized_dataset() -> ArtifactStep[RAGTokenizedCache]:
    """Reuse the frozen token cache with its recorded executable-local type."""
    return replace(
        _base_rag_tokenized_dataset(),
        artifact_type=RAGTokenizedCache,
    )


def build() -> ArtifactStep:
    """Assemble the scratch 103.8M, 2M-token-batch training run."""
    assert ACTUAL_TOKENS_LARGE_BATCH == (
        TRAIN_STEPS_LARGE_BATCH * LARGE_BATCH_SIZE * SEQ_LEN
    )
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
            ram=TRAIN_HOST_RAM_100M,
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
            "104M",
            "scratch",
            "scale-rung",
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
