"""Disposable two-step full-batch memory smoke for the 103.8M large-batch run."""

from dataclasses import replace

from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import experiment_main

from launch_100m_large_batch_30k import build as _build_training

SMOKE_CHECKPOINT_NAME = "checkpoints/dna-exp402-rag-h768-p104m-b2m-fullbatch-smoke"
SMOKE_RUN_ID = "dna-exp402-rag-h768-p104M-B2M-fullbatch-smoke"
SMOKE_STEPS = 2


def build() -> ArtifactStep:
    """Use the production recipe but isolate and shorten its output."""
    training = _build_training()
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
                    id=SMOKE_RUN_ID,
                    num_train_steps=SMOKE_STEPS,
                    steps_per_eval=1,
                    tracker=replace(trainer.tracker, name=SMOKE_RUN_ID),
                    checkpointer=replace(trainer.checkpointer, keep=[]),
                ),
                hf_save_steps=SMOKE_STEPS,
            ),
        )

    return replace(
        training,
        name=SMOKE_CHECKPOINT_NAME,
        build_config=build_smoke_config,
    )


if __name__ == "__main__":
    experiment_main(build)()
