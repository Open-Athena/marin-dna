"""Issue #402: continue the 103.8M ortholog-RAG model to 60,000 steps.

Only model scale, source checkpoint, output identity, and host RAM differ from
``launch_60k.py``. The full-state step-24k resume and extended WSD schedule are
identical.
"""

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
    TRAIN_BATCH_SIZE,
    TRAIN_HOST_CPU,
    TRAIN_REGIONS,
    TRAIN_TPU,
    online_eval_enabled,
)
from launch_30k import NATIVE_CHECKPOINT_EVERY, OPTIMIZER_30K
from launch_60k import RESUME_STEP, TRAIN_STEPS_60K
from launch_100m import MODEL, TRAIN_HOST_RAM_100M
from launch_100m_30k import (
    CHECKPOINT_NAME as SOURCE_CHECKPOINT_NAME,
)
from launch_100m_30k import (
    RAGTokenizedCache as _BaseRAGTokenizedCache,
)
from launch_100m_30k import (
    build as build_100m_30k,
)
from launch_100m_30k import (
    rag_tokenized_dataset as _base_rag_tokenized_dataset,
)

RESUME_CHECKPOINT = (
    f"gs://marin-us-east5/{SOURCE_CHECKPOINT_NAME}/2026.07.26/checkpoints/step-{RESUME_STEP}"
)
CHECKPOINT_NAME = "checkpoints/dna-exp402-rag-h768-p104m-60k-from24k"
RUN_ID = "dna-exp402-rag-h768-p104M-60K-from24K"


class RAGTokenizedCache(_BaseRAGTokenizedCache):
    """Preserve the executable-local type recorded by the frozen cache."""


def rag_tokenized_dataset() -> ArtifactStep[RAGTokenizedCache]:
    """Reuse the frozen token cache with this executable's artifact type."""
    return replace(
        _base_rag_tokenized_dataset(),
        artifact_type=RAGTokenizedCache,
    )


def build() -> ArtifactStep:
    """Assemble the full-state 103.8M continuation from step 24k to step 60k."""
    dataset = rag_tokenized_dataset()
    source_training = build_100m_30k()
    training = train_lm(
        name=CHECKPOINT_NAME,
        model=MODEL,
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
            "resume",
            "scale-rung",
            "60k",
        ),
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
