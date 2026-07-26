"""Issue #402: train the fresh 45.9M ortholog-RAG model for 30,000 steps.

This follow-up keeps the frozen corpus, model geometry, context, and token batch
from ``launch.py``. It resolves AdamH for the longer 3.93B-token horizon and
permanently retains both native optimizer-state checkpoints and Hugging Face
exports every 1,000 steps.
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
    TRAIN_BATCH_SIZE,
    TRAIN_HOST_CPU,
    TRAIN_REGIONS,
    TRAIN_TPU,
    online_eval_enabled,
    resolve_completed_adamh,
)
from launch import (
    RAGTokenizedCache as _BaseRAGTokenizedCache,
)
from launch import (
    rag_tokenized_dataset as _base_rag_tokenized_dataset,
)

TRAIN_STEPS_30K = 30_000
ACTUAL_TOKENS_30K = TRAIN_STEPS_30K * TRAIN_BATCH_SIZE * SEQ_LEN
OPTIMIZER_30K = resolve_completed_adamh(tokens=ACTUAL_TOKENS_30K)
NATIVE_CHECKPOINT_EVERY = 1_000
TRAIN_HOST_RAM_30K = "56g"
CHECKPOINT_NAME = "checkpoints/dna-exp402-rag-h640-p46m-30k"
RUN_ID = "dna-exp402-rag-h640-p46M-30K-scratch"


class RAGTokenizedCache(_BaseRAGTokenizedCache):
    """Preserve the executable-local type recorded by the frozen cache."""


def rag_tokenized_dataset() -> ArtifactStep[RAGTokenizedCache]:
    """Reuse the frozen token cache with its recorded executable-local type."""
    return replace(
        _base_rag_tokenized_dataset(),
        artifact_type=RAGTokenizedCache,
    )


def build() -> ArtifactStep:
    """Assemble the fresh 45.9M, 30,000-step training run."""
    dataset = rag_tokenized_dataset()
    training = train_lm(
        name=CHECKPOINT_NAME,
        model=MODEL,
        optimizer=OPTIMIZER_30K,
        datasets={dataset: 1.0},
        batch_size=TRAIN_BATCH_SIZE,
        seq_len=SEQ_LEN,
        num_train_steps=TRAIN_STEPS_30K,
        z_loss_weight=1.0e-7,
        evals=(
            EvalSuite(tasks=(MENDELIAN_TRAITS_RAG_255,), every=RAG_EVAL_EVERY)
            if online_eval_enabled()
            else None
        ),
        resources=ResourceConfig.with_tpu(
            TRAIN_TPU,
            cpu=TRAIN_HOST_CPU,
            ram=TRAIN_HOST_RAM_30K,
            disk="100g",
            regions=TRAIN_REGIONS,
        ),
        steps_per_eval=RAG_EVAL_EVERY,
        wandb_project="marin",
        wandb_group="dna-exp402-v1",
        run_id=RUN_ID,
        tags=("dna", "dna-exp402", "rag", "qwen3", "46M", "scratch", "30k"),
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
