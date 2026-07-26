"""Invariants for issue #402's fresh 30,000-step 45.9M run."""

import math

from marin.execution.build_context import BuildContext, VersionCodex, build_context
from marin.execution.lazy import StepContext

from launch import (
    HF_SAVE_EVERY,
    MODEL,
    ONLINE_EVAL_ENV,
    RAG_EVAL_EVERY,
    SEQ_LEN,
    TRAIN_BATCH_SIZE,
)
from launch_30k import (
    ACTUAL_TOKENS_30K,
    CHECKPOINT_NAME,
    NATIVE_CHECKPOINT_EVERY,
    OPTIMIZER_30K,
    RUN_ID,
    TRAIN_HOST_RAM_30K,
    TRAIN_STEPS_30K,
    RAGTokenizedCache,
    build,
    rag_tokenized_dataset,
)


def test_fresh_46m_run_uses_the_longer_completed_transfer() -> None:
    assert TRAIN_STEPS_30K == 30_000
    assert ACTUAL_TOKENS_30K == 3_932_160_000
    assert ACTUAL_TOKENS_30K == TRAIN_STEPS_30K * TRAIN_BATCH_SIZE * SEQ_LEN
    assert math.isclose(
        OPTIMIZER_30K.learning_rate,
        0.00630 * (2.5e9 / ACTUAL_TOKENS_30K) ** 0.3,
    )
    assert math.isclose(
        OPTIMIZER_30K.adam_lr,
        0.000656 * (2.5e9 / ACTUAL_TOKENS_30K) ** 0.5,
    )
    assert OPTIMIZER_30K.warmup == 0.1
    assert OPTIMIZER_30K.decay == 0.2
    assert OPTIMIZER_30K.lr_schedule == "linear"
    assert OPTIMIZER_30K.min_lr_ratio == 0.0


def test_fresh_46m_run_retains_native_and_hf_checkpoints(monkeypatch) -> None:
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

    assert CHECKPOINT_NAME == "checkpoints/dna-exp402-rag-h640-p46m-30k"
    assert RUN_ID == "dna-exp402-rag-h640-p46M-30K-scratch"
    assert TRAIN_HOST_RAM_30K == "56g"
    assert train_config.model == MODEL
    assert train_config.optimizer == OPTIMIZER_30K
    assert train_config.trainer.num_train_steps == TRAIN_STEPS_30K
    assert train_config.trainer.steps_per_eval == RAG_EVAL_EVERY == 1_000
    assert train_config.trainer.checkpointer.keep == [{"every": NATIVE_CHECKPOINT_EVERY}]
    assert NATIVE_CHECKPOINT_EVERY == HF_SAVE_EVERY == 1_000
    assert train_config.hf_save_steps == HF_SAVE_EVERY
    assert train_config.eval_harness is None


def test_fresh_46m_run_preserves_cache_artifact_identity() -> None:
    dataset = rag_tokenized_dataset()
    assert dataset.artifact_type is RAGTokenizedCache
    assert RAGTokenizedCache.__name__ == "RAGTokenizedCache"
