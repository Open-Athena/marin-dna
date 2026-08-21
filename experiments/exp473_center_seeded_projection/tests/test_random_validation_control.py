from __future__ import annotations

import exp473_center_seeded_projection.random_validation_control as control
import pytest
from exp473_center_seeded_projection.experiment import (
    BATCH_SIZE,
    HF_SAVE_STEPS,
    NATIVE_CHECKPOINT_STEPS,
    PER_DEVICE_PARALLELISM,
    SEED,
    SEQUENCE_LENGTH,
    TRAIN_STEPS,
)
from marin.execution.lazy import StepContext


def _set_required_env(monkeypatch) -> None:
    monkeypatch.delenv("EXP473_TPU_REGION", raising=False)
    monkeypatch.delenv("EXP473_TPU_VARIANT", raising=False)
    monkeypatch.delenv("EXP473_TPU_RAM", raising=False)
    monkeypatch.delenv("EXP473_TPU_PREEMPTIBLE", raising=False)
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setenv("WANDB_ENTITY", "test-entity")
    monkeypatch.setenv("WANDB_PROJECT", "marin")
    monkeypatch.setenv(
        "MARIN_PREFIX",
        "gs://marin-us-east5/MarinDNA/exp473_center_seeded_projection",
    )


def _pin_test_revision(monkeypatch) -> str:
    revision = "a" * 40
    monkeypatch.setattr(control, "DATASET_REVISION", revision)
    return revision


def test_random_validation_control_is_single_matched_preemptible_arm(
    monkeypatch,
) -> None:
    _set_required_env(monkeypatch)
    revision = _pin_test_revision(monkeypatch)
    step = control.build_random_validation_training()
    pod = step.build_config(
        StepContext.for_fingerprint(
            runtime_arg_keys=step.runtime_args,
            deps=step.deps,
        )
    )
    train = pod.train_config

    assert step.name.endswith(control.RUN_ID)
    assert len(step.deps) == 1
    assert step.runtime_args["train_resources"].preemptible is True
    assert train.trainer.train_batch_size == BATCH_SIZE
    assert train.trainer.num_train_steps == TRAIN_STEPS
    assert train.trainer.steps_per_eval == control.VALIDATION_STEPS == 500
    assert train.trainer.seed == SEED
    assert train.trainer.per_device_parallelism == PER_DEVICE_PARALLELISM
    assert train.trainer.checkpointer.keep == [{"every": NATIVE_CHECKPOINT_STEPS}]
    assert train.train_seq_len == SEQUENCE_LENGTH
    assert train.data_seed == SEED
    assert train.hf_save_steps == HF_SAVE_STEPS
    assert train.z_loss_weight == 4.312883184368223e-06
    assert pod.env_vars["EXP473_TPU_PREEMPTIBLE"] == "true"
    assert f"hf_revision={revision}" in control.control_tags()
    assert "validation_rows=16384" in control.control_tags()


def test_tokenizer_child_owns_a_locked_cpu_environment(monkeypatch) -> None:
    _pin_test_revision(monkeypatch)
    step = control.tokenized_control_dataset()

    assert step.run.pip_dependency_groups == ["cpu"]


def test_random_validation_control_rejects_on_demand_tpu(monkeypatch) -> None:
    _set_required_env(monkeypatch)
    _pin_test_revision(monkeypatch)
    monkeypatch.setenv("EXP473_TPU_PREEMPTIBLE", "false")
    with pytest.raises(ValueError, match="requires a preemptible TPU"):
        control.build_random_validation_training()


def test_dataset_revision_must_be_immutable_lowercase_hex(monkeypatch) -> None:
    assert control.resolved_dataset_revision() == (
        "7ef0bc9fcff17efc5792af92d8da34176617dd13"
    )

    for revision in ("main", "A" * 40, "g" * 40):
        monkeypatch.setattr(control, "DATASET_REVISION", revision)
        with pytest.raises(ValueError, match="40-character lowercase hexadecimal"):
            control.resolved_dataset_revision()
