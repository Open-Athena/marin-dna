"""Invariants for issue #402's full-state 45.9M continuation to 60k."""

import math

from marin.execution.build_context import BuildContext, VersionCodex, build_context
from marin.execution.lazy import StepContext

from launch import HF_SAVE_EVERY, MODEL, ONLINE_EVAL_ENV, RAG_EVAL_EVERY, SEQ_LEN, TRAIN_BATCH_SIZE
from launch_30k import NATIVE_CHECKPOINT_EVERY, OPTIMIZER_30K
from launch_60k import (
    ACTUAL_TOKENS_60K,
    CHECKPOINT_NAME,
    RESUME_CHECKPOINT,
    RESUME_STEP,
    RUN_ID,
    TRAIN_STEPS_60K,
    RAGTokenizedCache,
    build,
    rag_tokenized_dataset,
)


def test_46m_continuation_restores_full_state_at_the_plateau_boundary(monkeypatch) -> None:
    monkeypatch.setenv(ONLINE_EVAL_ENV, "0")
    versions = VersionCodex(
        default="2026.07.26",
        overrides={CHECKPOINT_NAME: "test-dev"},
    )
    with build_context(BuildContext(versions=versions)):
        training = build()
    ctx = StepContext.for_fingerprint(training.runtime_args.keys(), training.deps)
    pod_config = training.build_config(ctx)
    train_config = pod_config.train_config
    trainer = train_config.trainer

    assert CHECKPOINT_NAME == "checkpoints/dna-exp402-rag-h640-p46m-60k-from24k"
    assert RUN_ID == "dna-exp402-rag-h640-p46M-60K-from24K"
    assert RESUME_STEP == 24_000
    assert RESUME_CHECKPOINT.endswith(
        "/checkpoints/dna-exp402-rag-h640-p46m-30k/2026.07.26/checkpoints/step-24000"
    )
    assert train_config.initialize_from_checkpoint_path is None
    assert trainer.initialize_from == RESUME_CHECKPOINT
    assert trainer.load_checkpoint is None
    assert trainer.load_checkpoint_path is None
    assert train_config.model == MODEL
    assert train_config.optimizer == OPTIMIZER_30K
    assert trainer.num_train_steps == TRAIN_STEPS_60K == 60_000
    assert ACTUAL_TOKENS_60K == TRAIN_STEPS_60K * TRAIN_BATCH_SIZE * SEQ_LEN
    assert ACTUAL_TOKENS_60K == 7_864_320_000
    assert trainer.steps_per_eval == RAG_EVAL_EVERY == 1_000
    assert trainer.checkpointer.keep == [{"every": NATIVE_CHECKPOINT_EVERY}]
    assert NATIVE_CHECKPOINT_EVERY == HF_SAVE_EVERY == 1_000
    assert train_config.hf_save_steps == HF_SAVE_EVERY
    assert train_config.eval_harness is None


def test_46m_continuation_preserves_cache_artifact_identity() -> None:
    dataset = rag_tokenized_dataset()
    assert dataset.artifact_type is RAGTokenizedCache
    assert RAGTokenizedCache.__name__ == "RAGTokenizedCache"


def test_60k_schedule_is_continuous_and_moves_decay_to_48k() -> None:
    schedule = OPTIMIZER_30K.lr_scheduler(TRAIN_STEPS_60K)
    plateau = OPTIMIZER_30K.learning_rate

    assert math.isclose(float(schedule(RESUME_STEP)), plateau, rel_tol=1e-7)
    assert math.isclose(float(schedule(48_000)), plateau, rel_tol=1e-7)
    assert 0.0 < float(schedule(48_001)) < plateau
    assert math.isclose(float(schedule(TRAIN_STEPS_60K)), 0.0, abs_tol=1e-12)
