from __future__ import annotations

import math

import pytest
from marin.execution.lazy import StepContext

from exp535_4b_uniform_five_region.experiment import (
    ADAM_LEARNING_RATE,
    COOLDOWN_START_STEP,
    DATA_SEED,
    GLOBAL_BATCH_SIZE,
    LEARNING_RATE,
    PARENT_CHECKPOINT,
    PARENT_STEP,
    PER_DEVICE_PARALLELISM,
    PRODUCTION_ADDED_STEPS,
    PRODUCTION_HF_EXPORT_STEPS,
    PRODUCTION_TARGET_STEP,
    REGION_CACHES,
    SMOKE_HF_EXPORT_STEPS,
    SMOKE_TARGET_STEP,
    TPU_REGION,
    TPU_VARIANT,
    TPU_ZONE,
    build_training,
    optimizer_config,
    validate_vendored_tokenizer,
)


def _required_env(monkeypatch) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "test-key-that-must-not-be-serialized")
    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-us-east5")


def _pod(monkeypatch, mode: str):
    _required_env(monkeypatch)
    step = build_training(mode)  # type: ignore[arg-type]
    ctx = StepContext.for_fingerprint(runtime_arg_keys=step.runtime_args, deps=step.deps)
    return step, step.build_config(ctx)


def test_vendored_tokenizer_is_exact() -> None:
    validate_vendored_tokenizer()


def test_production_is_full_state_resume_with_uniform_cached_data(monkeypatch) -> None:
    step, pod = _pod(monkeypatch, "production")
    train = pod.train_config
    assert train.trainer.initialize_from == PARENT_CHECKPOINT
    assert train.initialize_from_checkpoint_path is None
    assert train.trainer.num_train_steps == PRODUCTION_TARGET_STEP == PARENT_STEP + 160_000
    assert train.trainer.train_batch_size == GLOBAL_BATCH_SIZE == 1_536
    assert train.trainer.per_device_parallelism == PER_DEVICE_PARALLELISM == 192
    assert train.data_seed == DATA_SEED == 535
    assert train.eval_harness is None
    assert train.labeled_eval is None
    assert train.data.auto_build_caches is False
    assert train.data.train_weights == {name: 0.2 for name in REGION_CACHES}
    assert set(train.data.components) == set(REGION_CACHES)
    for name, cache in REGION_CACHES.items():
        component = train.data.components[name]
        assert component.cache_dir == cache.cache_dir
        assert component.format.text_key == cache.text_key
        assert component.format.lowercase_weight == 0.01
    assert train.adapter.steps == PRODUCTION_HF_EXPORT_STEPS
    assert PRODUCTION_HF_EXPORT_STEPS == (
        235_573,
        255_573,
        275_573,
        295_573,
        315_573,
        335_573,
        355_573,
        375_573,
    )
    assert train.trainer.checkpointer.keep == [
        {"every": COOLDOWN_START_STEP, "until": COOLDOWN_START_STEP}
    ]
    assert "WANDB_API_KEY" not in pod.env_vars
    assert "test-key-that-must-not-be-serialized" not in repr(pod)
    resources = step.runtime_args["train_resources"]
    assert resources.device.variant == TPU_VARIANT == "v5p-16"
    assert resources.regions == [TPU_REGION]
    assert resources.zone == TPU_ZONE == "us-east5-a"
    assert resources.preemptible is True


def test_smoke_has_separate_identity_and_twenty_updates(monkeypatch) -> None:
    production, _ = _pod(monkeypatch, "production")
    smoke, pod = _pod(monkeypatch, "smoke")
    assert smoke.name != production.name
    assert smoke.version != production.version
    assert pod.train_config.trainer.num_train_steps == SMOKE_TARGET_STEP == PARENT_STEP + 20
    assert pod.train_config.adapter.steps == SMOKE_HF_EXPORT_STEPS == (SMOKE_TARGET_STEP,)
    assert pod.train_config.trainer.checkpointer.keep == []


def test_second_wsd_cycle_has_requested_boundaries() -> None:
    optimizer = optimizer_config(PRODUCTION_ADDED_STEPS)
    assert optimizer.learning_rate == LEARNING_RATE
    assert optimizer.adam_lr == ADAM_LEARNING_RATE
    assert optimizer.cycle_length == [PARENT_STEP, PRODUCTION_ADDED_STEPS]
    assert optimizer.rewarmup == 16_000
    assert optimizer.decay == 32_000
    schedule = optimizer.lr_scheduler(PRODUCTION_TARGET_STEP)
    assert math.isclose(float(schedule(PARENT_STEP)), 0.0, abs_tol=1e-12)
    assert math.isclose(float(schedule(PARENT_STEP + 16_000)), LEARNING_RATE, rel_tol=1e-6)
    assert math.isclose(float(schedule(COOLDOWN_START_STEP)), LEARNING_RATE, rel_tol=1e-6)
    assert math.isclose(float(schedule(PRODUCTION_TARGET_STEP)), 0.0, abs_tol=1e-12)


def test_requires_approved_gcs_prefix(monkeypatch) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setenv("MARIN_PREFIX", "gs://some-other-bucket")
    with pytest.raises(ValueError, match="MARIN_PREFIX"):
        build_training("smoke")
