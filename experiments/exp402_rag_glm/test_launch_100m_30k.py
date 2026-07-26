"""Invariants for issue #402's fresh 30,000-step 103.8M run."""

from marin.execution.build_context import BuildContext, VersionCodex, build_context
from marin.execution.lazy import StepContext

from launch import HF_SAVE_EVERY, ONLINE_EVAL_ENV, RAG_EVAL_EVERY, VOCAB_SIZE
from launch_30k import (
    ACTUAL_TOKENS_30K,
    NATIVE_CHECKPOINT_EVERY,
    OPTIMIZER_30K,
    TRAIN_STEPS_30K,
)
from launch_100m import MODEL
from launch_100m_30k import (
    CHECKPOINT_NAME,
    RUN_ID,
    RAGTokenizedCache,
    build,
    rag_tokenized_dataset,
)


def test_fresh_104m_run_changes_only_the_model_scale_and_identity(monkeypatch) -> None:
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

    assert CHECKPOINT_NAME == "checkpoints/dna-exp402-rag-h768-p104m-30k"
    assert RUN_ID == "dna-exp402-rag-h768-p104M-30K-scratch"
    assert MODEL.total_trainable_params(VOCAB_SIZE) == 103_838_976
    assert train_config.model == MODEL
    assert train_config.optimizer == OPTIMIZER_30K
    assert train_config.trainer.num_train_steps == TRAIN_STEPS_30K == 30_000
    assert ACTUAL_TOKENS_30K == 3_932_160_000
    assert train_config.trainer.steps_per_eval == RAG_EVAL_EVERY == 1_000
    assert train_config.trainer.checkpointer.keep == [{"every": NATIVE_CHECKPOINT_EVERY}]
    assert NATIVE_CHECKPOINT_EVERY == HF_SAVE_EVERY == 1_000
    assert train_config.hf_save_steps == HF_SAVE_EVERY
    assert train_config.eval_harness is None


def test_fresh_104m_run_preserves_cache_artifact_identity() -> None:
    dataset = rag_tokenized_dataset()
    assert dataset.artifact_type is RAGTokenizedCache
    assert RAGTokenizedCache.__name__ == "RAGTokenizedCache"
