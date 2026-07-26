"""Disposable two-step full-batch memory smoke for the 45.9M large-batch run."""

from dataclasses import replace

from fray.types import ResourceConfig
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import experiment_main
from marin.experiment.train import train_lm

from launch import (
    MODEL,
    SEQ_LEN,
    TRAIN_HOST_CPU,
    TRAIN_REGIONS,
    TRAIN_TPU,
)
from launch import RAGTokenizedCache as _BaseRAGTokenizedCache
from launch import rag_tokenized_dataset as _base_rag_tokenized_dataset
from launch_large_batch_30k import (
    LARGE_BATCH_SIZE,
    OPTIMIZER_LARGE_BATCH,
    PER_DEVICE_PARALLELISM,
    TRAIN_HOST_RAM_LARGE_BATCH,
)

SMOKE_CHECKPOINT_NAME = "checkpoints/dna-exp402-rag-h640-p46m-b2m-fullbatch-smoke"
SMOKE_RUN_ID = "dna-exp402-rag-h640-p46M-B2M-fullbatch-smoke"
SMOKE_STEPS = 2


class RAGTokenizedCache(_BaseRAGTokenizedCache):
    """Match the executable-local type recorded by the frozen cache."""


def rag_tokenized_dataset() -> ArtifactStep[RAGTokenizedCache]:
    """Reuse the frozen cache while retaining its executable-local type."""
    return replace(
        _base_rag_tokenized_dataset(),
        artifact_type=RAGTokenizedCache,
    )


def build() -> ArtifactStep:
    """Run two full-batch updates in an isolated artifact and W&B run."""
    dataset = rag_tokenized_dataset()
    training = train_lm(
        name=SMOKE_CHECKPOINT_NAME,
        model=MODEL,
        optimizer=OPTIMIZER_LARGE_BATCH,
        datasets={dataset: 1.0},
        batch_size=LARGE_BATCH_SIZE,
        seq_len=SEQ_LEN,
        num_train_steps=SMOKE_STEPS,
        z_loss_weight=1.0e-7,
        evals=None,
        resources=ResourceConfig.with_tpu(
            TRAIN_TPU,
            cpu=TRAIN_HOST_CPU,
            ram=TRAIN_HOST_RAM_LARGE_BATCH,
            disk="100g",
            regions=TRAIN_REGIONS,
        ),
        steps_per_eval=1,
        wandb_project="marin",
        wandb_group="dna-exp402-v1",
        run_id=SMOKE_RUN_ID,
        tags=("dna", "dna-exp402", "rag", "46M", "large-batch", "memory-smoke"),
    )
    original_build_config = training.build_config

    def build_smoke_config(ctx: StepContext):
        pod_config = original_build_config(ctx)
        trainer = pod_config.train_config.trainer
        return replace(
            pod_config,
            train_config=replace(
                pod_config.train_config,
                trainer=replace(
                    trainer,
                    per_device_parallelism=PER_DEVICE_PARALLELISM,
                    checkpointer=replace(trainer.checkpointer, keep=[]),
                ),
                hf_save_steps=SMOKE_STEPS,
            ),
        )

    return replace(training, build_config=build_smoke_config)


if __name__ == "__main__":
    experiment_main(build)()
