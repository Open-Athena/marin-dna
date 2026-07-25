"""Focused invariants for issue #402's gated 103.8M scale rung."""

from marin.execution.build_context import BuildContext, VersionCodex, build_context
from marin.execution.lazy import StepContext

from launch import (
    ACTUAL_TOKENS,
    HF_SAVE_EVERY,
    ONLINE_EVAL_ENV,
    OPTIMIZER,
    RAG_EVAL_EVERY,
    SEQ_LEN,
    TRAIN_BATCH_SIZE,
    TRAIN_STEPS,
    VOCAB_SIZE,
)
from launch_100m import (
    CHECKPOINT_NAME,
    MODEL,
    RUN_ID,
    TRAIN_HOST_RAM_100M,
    RAGTokenizedCache,
    build,
    rag_tokenized_dataset,
)


def test_model_is_the_104m_rung_at_the_same_document_length() -> None:
    assert MODEL.max_seq_len == SEQ_LEN == 2_048
    assert MODEL.hidden_dim == 768
    assert MODEL.intermediate_dim == 3_072
    assert MODEL.num_layers == 11
    assert MODEL.num_heads == MODEL.num_kv_heads == 6
    assert MODEL.total_trainable_params(VOCAB_SIZE) == 103_838_976


def test_scale_rung_preserves_the_cache_artifact_type_name() -> None:
    dataset = rag_tokenized_dataset()
    assert dataset.artifact_type is RAGTokenizedCache
    assert RAGTokenizedCache.__name__ == "RAGTokenizedCache"


def test_scale_rung_preserves_horizon_optimizer_and_export_cadence(monkeypatch) -> None:
    monkeypatch.setenv(ONLINE_EVAL_ENV, "0")
    versions = VersionCodex(
        default="2026.07.24",
        overrides={CHECKPOINT_NAME: "test-dev"},
    )
    with build_context(BuildContext(versions=versions)):
        training = build()
    ctx = StepContext.for_fingerprint(training.runtime_args.keys(), training.deps)
    pod_config = training.build_config(ctx)

    assert CHECKPOINT_NAME == "checkpoints/dna-exp402-rag-h768-p104m-1b"
    assert RUN_ID == "dna-exp402-rag-h768-p104M-1B"
    assert TRAIN_BATCH_SIZE == 64
    assert TRAIN_STEPS == 7_629
    assert ACTUAL_TOKENS == 999_948_288
    assert TRAIN_HOST_RAM_100M == "56g"
    assert pod_config.train_config.optimizer == OPTIMIZER
    assert pod_config.train_config.hf_save_steps == HF_SAVE_EVERY == 1_000
    assert RAG_EVAL_EVERY == 1_000
    assert pod_config.train_config.eval_harness is None
