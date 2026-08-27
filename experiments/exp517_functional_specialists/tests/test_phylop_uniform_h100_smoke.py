from __future__ import annotations

import pytest
from exp517_functional_specialists.experiment import BATCH_SIZE, SEQUENCE_LENGTH
from exp517_functional_specialists.phylop_uniform_experiment import ARMS
from exp517_functional_specialists.phylop_uniform_h100_smoke import (
    DEFAULT_H100_CLUSTER,
    DEFAULT_H100_PER_DEVICE_PARALLELISM,
    SMOKE_NUM_SHARDS,
    SMOKE_SAMPLE_COUNT,
    SMOKE_TRAIN_STEPS,
    build_smoke,
    smoke_tokenized_dataset,
)
from fray.types import ANY_REGION
from marin.execution.lazy import StepContext


def _set_required_env(monkeypatch) -> None:
    monkeypatch.delenv("EXP517_H100_CLUSTER", raising=False)
    monkeypatch.delenv("EXP517_H100_SMOKE_PDP", raising=False)
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setenv("WANDB_ENTITY", "test-entity")
    monkeypatch.setenv("WANDB_PROJECT", "marin")
    monkeypatch.setenv(
        "MARIN_PREFIX",
        "s3://marin-coreweave/MarinDNA/exp517-phylop-h100-smoke",
    )


def test_smoke_uses_small_immutable_hf_sample() -> None:
    step = smoke_tokenized_dataset(ARMS["cds"])
    config = step.build_config(StepContext.for_fingerprint(deps=step.deps))
    assert config.id == ARMS["cds"].hf_repo
    assert config.revision == ARMS["cds"].revision
    assert config.sample_count == SMOKE_SAMPLE_COUNT == 16_384
    assert config.num_shards == SMOKE_NUM_SHARDS == 8
    assert config.format.text_key == "sequence"
    assert config.worker_resources.regions == [ANY_REGION]
    assert config.worker_resources.target_cluster == DEFAULT_H100_CLUSTER
    assert config.worker_resources.preemptible is True
    assert step.run.resources.regions == [ANY_REGION]
    assert step.run.resources.target_cluster == DEFAULT_H100_CLUSTER
    assert step.run.resources.preemptible is True
    assert "h100-smoke" in config.tags
    assert "gs://" not in repr(config)


def test_smoke_requests_one_preemptible_h100_at_full_batch(monkeypatch) -> None:
    _set_required_env(monkeypatch)
    step = build_smoke(ARMS["cds"])
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
        DEFAULT_H100_PER_DEVICE_PARALLELISM
    ) == 8_192
    assert train.trainer.num_train_steps == SMOKE_TRAIN_STEPS == 3
    assert train.train_seq_len == SEQUENCE_LENGTH == 256
    assert resources.device.kind == "gpu"
    assert resources.device.variant == "H100"
    assert resources.device.count == 1
    assert resources.regions == [ANY_REGION]
    assert resources.target_cluster == DEFAULT_H100_CLUSTER
    assert resources.preemptible is True
    assert pod.env_vars["MARIN_PREFIX"].startswith("s3://")
    assert "WANDB_API_KEY" not in pod.env_vars
    assert "test-key" not in repr(pod)


def test_smoke_rejects_gcs_and_unapproved_microbatch(monkeypatch) -> None:
    _set_required_env(monkeypatch)
    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-us-east5/exp517")
    with pytest.raises(ValueError, match="CoreWeave-local S3"):
        build_smoke(ARMS["cds"])

    monkeypatch.setenv("MARIN_PREFIX", "s3://marin-coreweave/exp517")
    monkeypatch.setenv("EXP517_H100_SMOKE_PDP", "512")
    with pytest.raises(ValueError, match="must be one of"):
        build_smoke(ARMS["cds"])
