"""Invariants for issue #402's 103.8M, 2M-token-batch scratch run."""

from marin.execution.build_context import BuildContext, VersionCodex, build_context
from marin.execution.lazy import StepContext

from launch import HF_SAVE_EVERY, ONLINE_EVAL_ENV, RAG_EVAL_EVERY, VOCAB_SIZE
from launch_100m import MODEL
from launch_100m_large_batch_30k import CHECKPOINT_NAME, RUN_ID, build
from launch_100m_large_batch_pdp128_benchmark import (
    BENCHMARK_CHECKPOINT_NAME,
    BENCHMARK_PER_DEVICE_PARALLELISM,
    BENCHMARK_RUN_ID,
    BENCHMARK_STEPS,
)
from launch_100m_large_batch_pdp128_benchmark import (
    build as build_benchmark,
)
from launch_large_batch_30k import (
    ACTUAL_TOKENS_LARGE_BATCH,
    GRADIENT_ACCUMULATION_STEPS,
    LARGE_BATCH_SIZE,
    NATIVE_CHECKPOINT_EVERY,
    OPTIMIZER_LARGE_BATCH,
    PER_DEVICE_PARALLELISM,
    TRAIN_DEVICE_COUNT,
    TRAIN_STEPS_LARGE_BATCH,
)


def _pod_config(step_builder, name: str, monkeypatch):
    monkeypatch.setenv(ONLINE_EVAL_ENV, "0")
    versions = VersionCodex(default="2026.07.26", overrides={name: "test-dev"})
    with build_context(BuildContext(versions=versions)):
        training = step_builder()
    ctx = StepContext.for_fingerprint(training.runtime_args.keys(), training.deps)
    return training, training.build_config(ctx)


def test_104m_large_batch_changes_only_model_scale_and_identity(monkeypatch) -> None:
    training, pod_config = _pod_config(build, CHECKPOINT_NAME, monkeypatch)
    train_config = pod_config.train_config
    trainer = train_config.trainer

    assert training.name.endswith(CHECKPOINT_NAME)
    assert RUN_ID == "dna-exp402-rag-h768-p104M-B2M-30K-scratch"
    assert MODEL.total_trainable_params(VOCAB_SIZE) == 103_838_976
    assert trainer.train_batch_size == LARGE_BATCH_SIZE == 1_024
    assert trainer.per_device_parallelism == PER_DEVICE_PARALLELISM == 16
    assert TRAIN_DEVICE_COUNT == 4
    assert GRADIENT_ACCUMULATION_STEPS == 16
    assert (
        trainer.train_batch_size
        == trainer.per_device_parallelism
        * TRAIN_DEVICE_COUNT
        * GRADIENT_ACCUMULATION_STEPS
    )
    assert trainer.num_train_steps == TRAIN_STEPS_LARGE_BATCH == 30_000
    assert ACTUAL_TOKENS_LARGE_BATCH == 62_914_560_000
    assert trainer.steps_per_eval == RAG_EVAL_EVERY == 1_000
    assert trainer.checkpointer.keep == [{"every": NATIVE_CHECKPOINT_EVERY}]
    assert NATIVE_CHECKPOINT_EVERY == HF_SAVE_EVERY == 1_000
    assert train_config.hf_save_steps == HF_SAVE_EVERY
    assert train_config.model == MODEL
    assert train_config.optimizer == OPTIMIZER_LARGE_BATCH
    assert train_config.eval_harness is None


def test_104m_pdp128_benchmark_is_compile_warmed_and_isolated(monkeypatch) -> None:
    training, pod_config = _pod_config(
        build_benchmark, BENCHMARK_CHECKPOINT_NAME, monkeypatch
    )
    trainer = pod_config.train_config.trainer
    assert training.name.endswith(BENCHMARK_CHECKPOINT_NAME)
    assert trainer.id == BENCHMARK_RUN_ID
    assert trainer.tracker.name == BENCHMARK_RUN_ID
    assert trainer.num_train_steps == BENCHMARK_STEPS == 6
    assert trainer.train_batch_size == LARGE_BATCH_SIZE
    assert trainer.per_device_parallelism == BENCHMARK_PER_DEVICE_PARALLELISM == 128
    assert trainer.max_eval_batches == 0
    assert trainer.checkpointer.keep == []
    assert pod_config.train_config.hf_save_path is None
    assert pod_config.train_config.model == MODEL
    assert pod_config.train_config.optimizer == OPTIMIZER_LARGE_BATCH
