"""Invariants for issue #402's full-state 103.8M continuation to 60k."""

from marin.execution.build_context import BuildContext, VersionCodex, build_context
from marin.execution.lazy import StepContext

from launch import HF_SAVE_EVERY, ONLINE_EVAL_ENV, RAG_EVAL_EVERY, VOCAB_SIZE
from launch_30k import NATIVE_CHECKPOINT_EVERY, OPTIMIZER_30K
from launch_60k import RESUME_STEP, TRAIN_STEPS_60K
from launch_100m import MODEL
from launch_100m_60k import (
    CHECKPOINT_NAME,
    RESUME_CHECKPOINT,
    RUN_ID,
    RAGTokenizedCache,
    build,
    rag_tokenized_dataset,
)


def test_104m_continuation_changes_only_scale_source_and_identity(monkeypatch) -> None:
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

    assert CHECKPOINT_NAME == "checkpoints/dna-exp402-rag-h768-p104m-60k-from24k"
    assert RUN_ID == "dna-exp402-rag-h768-p104M-60K-from24K"
    assert RESUME_STEP == 24_000
    assert RESUME_CHECKPOINT.endswith(
        "/checkpoints/dna-exp402-rag-h768-p104m-30k/2026.07.26/checkpoints/step-24000"
    )
    assert train_config.initialize_from_checkpoint_path is None
    assert trainer.initialize_from == RESUME_CHECKPOINT
    assert trainer.load_checkpoint is None
    assert trainer.load_checkpoint_path is None
    assert MODEL.total_trainable_params(VOCAB_SIZE) == 103_838_976
    assert train_config.model == MODEL
    assert train_config.optimizer == OPTIMIZER_30K
    assert trainer.num_train_steps == TRAIN_STEPS_60K == 60_000
    assert trainer.steps_per_eval == RAG_EVAL_EVERY == 1_000
    assert trainer.checkpointer.keep == [{"every": NATIVE_CHECKPOINT_EVERY}]
    assert NATIVE_CHECKPOINT_EVERY == HF_SAVE_EVERY == 1_000
    assert train_config.hf_save_steps == HF_SAVE_EVERY
    assert train_config.eval_harness is None


def test_104m_continuation_preserves_cache_artifact_identity() -> None:
    dataset = rag_tokenized_dataset()
    assert dataset.artifact_type is RAGTokenizedCache
    assert RAGTokenizedCache.__name__ == "RAGTokenizedCache"
