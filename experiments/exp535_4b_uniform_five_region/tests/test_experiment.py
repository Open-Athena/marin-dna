from __future__ import annotations

import math
from types import SimpleNamespace

import cloudpickle
import draccus
import pytest
from levanter.adaptor import AdaptorExportConfig
from marin.execution.lazy import StepContext

from exp535_4b_uniform_five_region.experiment import (
    ADAM_LEARNING_RATE,
    COOLDOWN_START_STEP,
    DATA_SEED,
    ExactHfExportConfig,
    GLOBAL_BATCH_SIZE,
    LEARNING_RATE,
    PARENT_CHECKPOINT,
    PARENT_STEP,
    PER_DEVICE_PARALLELISM,
    PRODUCTION_ADDED_STEPS,
    PRODUCTION_HF_EXPORT_STEPS,
    PRODUCTION_STOP_STEP,
    PRODUCTION_TARGET_STEP,
    REGION_CACHES,
    SMOKE_HF_EXPORT_STEPS,
    SMOKE_STOP_STEP,
    SMOKE_TARGET_STEP,
    TPU_REGION,
    TPU_VARIANT,
    TPU_ZONE,
    TrainOnlyLmDataConfig,
    _require_preemptible_iris_capacity,
    _resources,
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
    assert PRODUCTION_TARGET_STEP == PARENT_STEP + 160_000
    assert train.trainer.num_train_steps == PRODUCTION_STOP_STEP
    assert PRODUCTION_STOP_STEP == PRODUCTION_TARGET_STEP + 1
    assert train.trainer.train_batch_size == GLOBAL_BATCH_SIZE == 1_536
    assert train.trainer.per_device_parallelism == PER_DEVICE_PARALLELISM == 192
    assert train.data_seed == DATA_SEED == 535
    assert train.eval_harness is None
    assert train.labeled_eval is None
    assert isinstance(train.data, TrainOnlyLmDataConfig)
    assert train.data.tagged_eval_sets(object()) == []
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
    assert resources.device.variant == TPU_VARIANT == "v5p-8"
    assert resources.regions == [TPU_REGION]
    assert resources.zone == TPU_ZONE == "us-east5-a"
    assert resources.preemptible is True


def test_smoke_has_separate_identity_and_twenty_updates(monkeypatch) -> None:
    production, _ = _pod(monkeypatch, "production")
    smoke, pod = _pod(monkeypatch, "smoke")
    assert smoke.name != production.name
    assert smoke.name == "checkpoints/dna-exp535-4b-uniform-five-region-smoke-v5p8-v2"
    assert pod.train_config.trainer.id == "dna-exp535-4b-uniform-five-region-smoke-v5p8-v2"
    assert SMOKE_TARGET_STEP == PARENT_STEP + 20
    assert pod.train_config.trainer.num_train_steps == SMOKE_STOP_STEP
    assert SMOKE_STOP_STEP == SMOKE_TARGET_STEP + 1
    assert SMOKE_STOP_STEP - (PARENT_STEP + 1) == 20
    assert pod.train_config.adapter.steps == SMOKE_HF_EXPORT_STEPS == (SMOKE_TARGET_STEP,)
    assert pod.train_config.trainer.checkpointer.keep == []


def test_train_only_data_config_survives_remote_serialization(monkeypatch) -> None:
    _, pod = _pod(monkeypatch, "smoke")
    restored = cloudpickle.loads(cloudpickle.dumps(pod))
    assert isinstance(restored.train_config.data, TrainOnlyLmDataConfig)
    assert restored.train_config.data.tagged_eval_sets(object()) == []
    assert type(restored.train_config.adapter).__module__ == "exp535_4b_uniform_five_region.exports"
    assert draccus.encode(restored.train_config)["adapter"] == {
        "type": "exact-hf",
        "steps": [SMOKE_TARGET_STEP],
    }


def test_exact_hf_export_fires_once_at_completed_target(monkeypatch) -> None:
    from exp535_4b_uniform_five_region import exports

    exported: list[int] = []
    installed: dict[str, object] = {}

    def fake_callback(*args, **kwargs):
        del args, kwargs

        def record(info) -> None:
            exported.append(info.step)

        return record

    class FakeTrainer:
        config = SimpleNamespace(
            checkpointer=SimpleNamespace(append_run_id_to_base_path=False)
        )
        run_id = "test-run"

        def add_hook(self, hook, *, every: int) -> None:
            installed["hook"] = hook
            installed["every"] = every

    monkeypatch.setattr(exports, "save_hf_checkpoint_callback", fake_callback)
    config = ExactHfExportConfig(steps=(SMOKE_TARGET_STEP,))
    config.install_export_hooks(
        trainer=FakeTrainer(),
        converter=object(),  # type: ignore[arg-type]
        tokenizer=object(),
        export=AdaptorExportConfig(hf_save_path="gs://example/hf"),
    )
    hook = installed["hook"]
    assert callable(hook)
    assert installed["every"] == 1
    candidate_steps = (
        SMOKE_TARGET_STEP - 1,
        SMOKE_TARGET_STEP,
        SMOKE_TARGET_STEP,
        SMOKE_TARGET_STEP + 1,
    )
    for step in candidate_steps:
        hook(SimpleNamespace(step=step))
    assert exported == [SMOKE_TARGET_STEP]


def test_tpu_submission_requires_preemptible_capacity() -> None:
    import fray.iris_backend as iris_backend
    from iris.cluster.types import WellKnownAttribute

    with _require_preemptible_iris_capacity():
        constraints = iris_backend.convert_constraints(_resources())
    preemptible = [
        constraint
        for constraint in constraints
        if constraint.key == WellKnownAttribute.PREEMPTIBLE
    ]
    assert len(preemptible) == 1
    assert preemptible[0].values[0].value == "true"
    assert preemptible[0].is_soft is False


def test_second_wsd_cycle_has_requested_boundaries() -> None:
    import jax

    jax.config.update("jax_platforms", "cpu")
    optimizer = optimizer_config(PRODUCTION_ADDED_STEPS)
    assert optimizer.learning_rate == LEARNING_RATE
    assert optimizer.adam_lr == ADAM_LEARNING_RATE
    assert optimizer.cycle_length == [PARENT_STEP, PRODUCTION_ADDED_STEPS]
    assert optimizer.rewarmup == 16_000
    assert optimizer.decay == 32_000
    schedule = optimizer.lr_scheduler(PRODUCTION_STOP_STEP)
    assert math.isclose(float(schedule(PARENT_STEP)), 0.0, abs_tol=1e-12)
    assert math.isclose(float(schedule(PARENT_STEP + 16_000)), LEARNING_RATE, rel_tol=1e-6)
    assert math.isclose(float(schedule(COOLDOWN_START_STEP)), LEARNING_RATE, rel_tol=1e-6)
    assert math.isclose(float(schedule(PRODUCTION_TARGET_STEP)), 0.0, abs_tol=1e-12)


def test_requires_approved_gcs_prefix(monkeypatch) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setenv("MARIN_PREFIX", "gs://some-other-bucket")
    with pytest.raises(ValueError, match="MARIN_PREFIX"):
        build_training("smoke")
