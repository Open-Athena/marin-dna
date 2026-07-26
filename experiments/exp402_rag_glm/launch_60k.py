"""Issue #402: continue the 45.9M ortholog-RAG model to 60,000 steps.

The continuation restores the full native step-24,000 trainer state from the
30k run. Step 24,000 is the stable/decay boundary: the recorded learning rate
is still the plateau value there, while step 24,001 is the first decayed point.
The original optimizer hyperparameters are therefore retained exactly, while
the longer trainer horizon moves the 20% decay window to steps 48k--60k.
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
)
from launch_30k import CHECKPOINT_NAME as SOURCE_CHECKPOINT_NAME
from launch_30k import (
    NATIVE_CHECKPOINT_EVERY,
    OPTIMIZER_30K,
    TRAIN_HOST_RAM_30K,
)
from launch_30k import (
    RAGTokenizedCache as _BaseRAGTokenizedCache,
)
from launch_30k import (
    build as build_30k,
)
from launch_30k import (
    rag_tokenized_dataset as _base_rag_tokenized_dataset,
)

TRAIN_STEPS_60K = 60_000
RESUME_STEP = 24_000
ACTUAL_TOKENS_60K = TRAIN_STEPS_60K * TRAIN_BATCH_SIZE * SEQ_LEN
RESUME_CHECKPOINT = (
    f"gs://marin-us-east5/{SOURCE_CHECKPOINT_NAME}/2026.07.26/checkpoints/step-{RESUME_STEP}"
)
CHECKPOINT_NAME = "checkpoints/dna-exp402-rag-h640-p46m-60k-from24k"
RUN_ID = "dna-exp402-rag-h640-p46M-60K-from24K"


class RAGTokenizedCache(_BaseRAGTokenizedCache):
    """Preserve the executable-local type recorded by the frozen cache."""


def rag_tokenized_dataset() -> ArtifactStep[RAGTokenizedCache]:
    """Reuse the frozen token cache with this executable's artifact type."""
    return replace(
        _base_rag_tokenized_dataset(),
        artifact_type=RAGTokenizedCache,
    )


def build() -> ArtifactStep:
    """Assemble the full-state 45.9M continuation from step 24k to step 60k."""
    dataset = rag_tokenized_dataset()
    source_training = build_30k()
    training = train_lm(
        name=CHECKPOINT_NAME,
        model=MODEL,
        # Keep the exact plateau LR and optimizer hyperparameters whose moments
        # are restored from step 24k. Only num_train_steps changes the schedule:
        # the resumed run stays flat through step 48k and then decays to 60k.
        optimizer=OPTIMIZER_30K,
        datasets={dataset: 1.0},
        batch_size=TRAIN_BATCH_SIZE,
        seq_len=SEQ_LEN,
        num_train_steps=TRAIN_STEPS_60K,
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
        # Retain cheap validation loss every 1k. The expensive VEP suites run
        # offline at 5k cadence and do not block training.
        steps_per_eval=RAG_EVAL_EVERY,
        wandb_project="marin",
        wandb_group="dna-exp402-v1",
        run_id=RUN_ID,
        tags=("dna", "dna-exp402", "rag", "qwen3", "46M", "resume", "60k"),
        # This dependency proves the source artifact exists before dispatch.
        # Its default weights-init field is replaced below by TrainerConfig's
        # full-state initialize_from path.
        init_from=source_training,
    )
    original_build_config = training.build_config

    def build_config_with_full_state_resume(ctx: StepContext):
        pod_config = original_build_config(ctx)
        trainer = pod_config.train_config.trainer
        assert pod_config.train_config.initialize_from_checkpoint_path is not None
        return replace(
            pod_config,
            train_config=replace(
                pod_config.train_config,
                # Marin's init_from helper normally resets step/data position.
                # TrainerConfig.initialize_from instead restores model, optimizer
                # moments, step, RNG, and resumes the data loader at that step.
                initialize_from_checkpoint_path=None,
                trainer=replace(
                    trainer,
                    initialize_from=RESUME_CHECKPOINT,
                    checkpointer=replace(
                        trainer.checkpointer,
                        keep=[{"every": NATIVE_CHECKPOINT_EVERY}],
                    ),
                ),
                hf_save_steps=HF_SAVE_EVERY,
            ),
        )

    return replace(training, build_config=build_config_with_full_state_resume)


if __name__ == "__main__":
    experiment_main(build)()
