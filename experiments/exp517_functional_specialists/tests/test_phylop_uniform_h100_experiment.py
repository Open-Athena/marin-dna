from __future__ import annotations

import os

import pytest
import exp517_functional_specialists.phylop_uniform_h100_experiment as h100_experiment
from exp517_functional_specialists.experiment import (
    BATCH_SIZE,
    HF_SAVE_STEPS,
    MODEL_TOKENIZER_PATH,
    NATIVE_CHECKPOINT_STEPS,
    SEQUENCE_LENGTH,
    TRAIN_STEPS,
)
from exp517_functional_specialists.phylop_uniform_experiment import ARMS
from exp517_functional_specialists.phylop_uniform_h100_experiment import (
    FULL_H100_LOCAL_TOKENIZER_WORKERS,
    FULL_H100_NUM_SHARDS,
    FULL_H100_PER_DEVICE_PARALLELISM,
    build_full_h100_training,
    full_h100_tokenized_dataset,
    tokenize_with_local_workers,
)
from fray.types import ANY_REGION
from marin.execution.lazy import StepContext


def _set_required_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXP517_H100_CLUSTER", "cw-rno2a")
    monkeypatch.delenv("EXP517_H100_PDP", raising=False)
    monkeypatch.setenv("EXP517_H100_ARM", "cds")
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setenv("WANDB_ENTITY", "test-entity")
    monkeypatch.setenv("WANDB_PROJECT", "marin")
    monkeypatch.setenv(
        "MARIN_PREFIX",
        "s3://marin-coreweave/MarinDNA/exp517-phylop-h100",
    )


def test_full_h100_tokenizes_complete_immutable_dataset() -> None:
    step = full_h100_tokenized_dataset(ARMS["cds"])
    config = step.build_config(StepContext.for_fingerprint(deps=step.deps))
    assert config.id == ARMS["cds"].hf_repo
    assert config.revision == ARMS["cds"].revision
    assert config.sample_count is None
    assert config.num_shards == FULL_H100_NUM_SHARDS == 64
    assert config.max_workers == FULL_H100_LOCAL_TOKENIZER_WORKERS == 16
    assert config.format.text_key == "sequence"
    assert config.worker_resources.regions == [ANY_REGION]
    assert config.worker_resources.target_cluster is None
    assert config.worker_resources.preemptible is True
    assert "h100-full" in config.tags


def test_full_h100_uses_validated_one_gpu_recipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_required_env(monkeypatch)
    step = build_full_h100_training(ARMS["cds"])
    pod = step.build_config(
        StepContext.for_fingerprint(
            runtime_arg_keys=step.runtime_args,
            deps=step.deps,
        )
    )
    train = pod.train_config
    resources = step.runtime_args["train_resources"]
    assert train.trainer.train_batch_size == BATCH_SIZE == 8_192
    assert train.trainer.per_device_parallelism == (
        FULL_H100_PER_DEVICE_PARALLELISM
    ) == 2_048
    assert train.trainer.num_train_steps == TRAIN_STEPS == 5_000
    assert train.train_seq_len == SEQUENCE_LENGTH == 256
    assert train.data.tokenizer == MODEL_TOKENIZER_PATH
    assert train.data.tokenizer.startswith("/")
    assert train.hf_save_steps == HF_SAVE_STEPS == 500
    assert train.trainer.checkpointer.keep == [
        {"every": NATIVE_CHECKPOINT_STEPS}
    ]
    assert resources.device.kind == "gpu"
    assert resources.device.variant == "H100"
    assert resources.device.count == 1
    assert resources.regions == [ANY_REGION]
    assert resources.target_cluster is None
    assert resources.preemptible is True
    assert pod.env_vars["MARIN_PREFIX"].startswith("s3://")
    assert "WANDB_API_KEY" not in pod.env_vars
    assert "test-key" not in repr(pod)


def test_full_h100_rejects_unvalidated_microbatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_required_env(monkeypatch)
    monkeypatch.setenv("EXP517_H100_PDP", "4096")
    with pytest.raises(ValueError, match="validated one-H100 value"):
        build_full_h100_training(ARMS["cds"])


def test_local_tokenizer_workers_do_not_rediscover_iris(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = full_h100_tokenized_dataset(ARMS["cds"]).build_config(
        StepContext.for_fingerprint(deps=())
    )
    observed_task_ids: list[str | None] = []
    monkeypatch.setenv("IRIS_TASK_ID", "parent-task")
    monkeypatch.setattr(
        h100_experiment,
        "tokenize",
        lambda _: observed_task_ids.append(os.environ.get("IRIS_TASK_ID")),
    )

    tokenize_with_local_workers(config)

    assert observed_task_ids == [None]
    assert os.environ["IRIS_TASK_ID"] == "parent-task"
