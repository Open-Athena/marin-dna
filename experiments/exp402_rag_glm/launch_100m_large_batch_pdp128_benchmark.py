"""Six-step, compile-warmed PDP=128 benchmark for the 103.8M run."""

from dataclasses import replace

from fray.types import ResourceConfig
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import experiment_main
from marin.experiment.train import train_lm

from launch import SEQ_LEN, TRAIN_HOST_CPU, TRAIN_REGIONS, TRAIN_TPU
from launch import RAGTokenizedCache as _BaseRAGTokenizedCache
from launch import rag_tokenized_dataset as _base_rag_tokenized_dataset
from launch_100m import MODEL, TRAIN_HOST_RAM_100M
from launch_large_batch_30k import (
    LARGE_BATCH_SIZE,
    OPTIMIZER_LARGE_BATCH,
)

BENCHMARK_CHECKPOINT_NAME = (
    "checkpoints/dna-exp402-rag-h768-p104m-b2m-pdp128-warm6"
)
BENCHMARK_RUN_ID = "dna-exp402-rag-h768-p104M-B2M-pdp128-warm6"
BENCHMARK_STEPS = 6
BENCHMARK_PER_DEVICE_PARALLELISM = 128


class RAGTokenizedCache(_BaseRAGTokenizedCache):
    """Match the executable-local type recorded by the frozen cache."""


def rag_tokenized_dataset() -> ArtifactStep[RAGTokenizedCache]:
    """Reuse the frozen cache while retaining its executable-local type."""
    return replace(
        _base_rag_tokenized_dataset(),
        artifact_type=RAGTokenizedCache,
    )


def build() -> ArtifactStep:
    """Run six 2-microstep updates so updates 3--6 measure warmed throughput."""
    dataset = rag_tokenized_dataset()
    training = train_lm(
        name=BENCHMARK_CHECKPOINT_NAME,
        model=MODEL,
        optimizer=OPTIMIZER_LARGE_BATCH,
        datasets={dataset: 1.0},
        batch_size=LARGE_BATCH_SIZE,
        seq_len=SEQ_LEN,
        num_train_steps=BENCHMARK_STEPS,
        z_loss_weight=1.0e-7,
        evals=None,
        resources=ResourceConfig.with_tpu(
            TRAIN_TPU,
            cpu=TRAIN_HOST_CPU,
            ram=TRAIN_HOST_RAM_100M,
            disk="100g",
            regions=TRAIN_REGIONS,
        ),
        steps_per_eval=1_000,
        wandb_project="marin",
        wandb_group="dna-exp402-v1",
        run_id=BENCHMARK_RUN_ID,
        tags=(
            "dna",
            "dna-exp402",
            "rag",
            "104M",
            "large-batch",
            "pdp128",
            "warm-throughput-benchmark",
        ),
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
                    per_device_parallelism=BENCHMARK_PER_DEVICE_PARALLELISM,
                    max_eval_batches=0,
                    checkpointer=replace(trainer.checkpointer, keep=[]),
                ),
                hf_save_path=None,
            ),
        )

    return replace(training, build_config=build_smoke_config)


if __name__ == "__main__":
    experiment_main(build)()
