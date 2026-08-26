from __future__ import annotations

from dataclasses import replace

import pytest
from exp517_functional_specialists.experiment import (
    BATCH_SIZE,
    DEFAULT_TPU_PREEMPTIBLE,
    DEFAULT_TPU_RAM,
    DEFAULT_TPU_REGION,
    DEFAULT_TPU_VARIANT,
    HF_SAVE_STEPS,
    NATIVE_CHECKPOINT_STEPS,
    PER_DEVICE_PARALLELISM,
    SEED,
    SEQUENCE_LENGTH,
    TRAIN_STEPS,
)
from exp517_functional_specialists.gpn_uniform_experiment import (
    ARMS,
    DATA_VERSION,
    PUBLICATION_PRODUCER_COMMIT,
    SOURCE_CONFIG_SHA256,
    SOURCE_PRODUCER_COMMIT,
    build_training,
    tokenized_dataset,
    training_tags,
)
from marin.execution.lazy import StepContext


def _set_required_env(monkeypatch) -> None:
    monkeypatch.delenv("EXP517_TPU_REGION", raising=False)
    monkeypatch.delenv("EXP517_TPU_VARIANT", raising=False)
    monkeypatch.delenv("EXP517_TPU_RAM", raising=False)
    monkeypatch.delenv("EXP517_TPU_PREEMPTIBLE", raising=False)
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setenv("WANDB_ENTITY", "test-entity")
    monkeypatch.setenv("WANDB_PROJECT", "marin")
    monkeypatch.setenv(
        "MARIN_PREFIX",
        "gs://marin-us-east5/MarinDNA/exp517_gpn_uniform_specialists",
    )


def _published_arm(key: str = "cds"):
    return replace(ARMS[key], revision="a" * 40)


def test_six_gpn_uniform_arms_require_published_revisions() -> None:
    expected = {
        "cds": (
            "marin-dna/gpn-star-p-uniform-v1-cds",
            "4c722c74e4616d8cbf8bce55844ec26da7fc516f",
        ),
        "utr3": (
            "marin-dna/gpn-star-p-uniform-v1-utr3",
            "42ac7aed4565d0ec2800c9d8e2b1829daec274bd",
        ),
        "tss_utr5": (
            "marin-dna/gpn-star-p-uniform-v1-tss-utr5",
            "c2fdcf05d24856f004be303470183e5fc39188b9",
        ),
        "ncrna_exon": (
            "marin-dna/gpn-star-p-uniform-v1-ncrna-exon",
            "c5cea96abe3ae84dafdb52967b1168a269e01f43",
        ),
        "enhancer_arm_a": (
            "marin-dna/gpn-star-p-uniform-v1-enhancer-arm-a",
            "243210a0d93d93423b42e817d82d0abc3de37ef8",
        ),
        "background": (
            "marin-dna/gpn-star-p-uniform-v1-background",
            "24f9ccb7cdc7c242d2ce88783e25db5597466543",
        ),
    }
    assert {
        key: (arm.hf_repo, arm.revision) for key, arm in ARMS.items()
    } == expected
    for arm in ARMS.values():
        assert arm.resolved_revision() == arm.revision
        with pytest.raises(ValueError, match="40-character hexadecimal"):
            replace(arm, revision="UNPUBLISHED").resolved_revision()


def test_gpn_tokenization_reads_only_immutable_hugging_face_input() -> None:
    arm = _published_arm()
    step = tokenized_dataset(arm)
    config = step.build_config(StepContext.for_fingerprint(deps=step.deps))
    assert config.id == "marin-dna/gpn-star-p-uniform-v1-cds"
    assert config.revision == "a" * 40
    assert config.format.text_key == "sequence"
    assert config.format.uppercase_weight == 1.0
    assert config.format.lowercase_weight == 0.01
    assert f"publication={PUBLICATION_PRODUCER_COMMIT}" in config.tags
    assert f"source={SOURCE_PRODUCER_COMMIT}" in config.tags
    assert f"source_config={SOURCE_CONFIG_SHA256}" in config.tags
    assert "s3://" not in repr(config)
    assert step.run.env_vars == {
        "HF_HUB_DOWNLOAD_TIMEOUT": "120",
        "UV_LOCK_TIMEOUT": "7200",
    }


def test_gpn_fixed_specialist_recipe(monkeypatch) -> None:
    _set_required_env(monkeypatch)
    arm = _published_arm("background")
    step = build_training(arm)
    pod = step.build_config(
        StepContext.for_fingerprint(
            runtime_arg_keys=step.runtime_args,
            deps=step.deps,
        )
    )
    train = pod.train_config
    assert DATA_VERSION == "2026.08.26"
    assert train.trainer.train_batch_size == BATCH_SIZE == 8_192
    assert train.trainer.num_train_steps == TRAIN_STEPS == 5_000
    assert train.trainer.seed == SEED == 0
    assert train.trainer.steps_per_eval == HF_SAVE_STEPS == 500
    assert train.trainer.per_device_parallelism == PER_DEVICE_PARALLELISM == 1_024
    assert train.trainer.checkpointer.keep == [{"every": NATIVE_CHECKPOINT_STEPS}]
    assert train.train_seq_len == SEQUENCE_LENGTH == 256
    assert train.data_seed == SEED
    assert train.hf_save_steps == HF_SAVE_STEPS
    assert step.runtime_args["train_resources"].regions == [DEFAULT_TPU_REGION]
    assert (
        step.runtime_args["train_resources"].device.variant == DEFAULT_TPU_VARIANT
    )
    assert step.runtime_args["train_resources"].ram == DEFAULT_TPU_RAM
    assert (
        step.runtime_args["train_resources"].preemptible
        is DEFAULT_TPU_PREEMPTIBLE
    )
    assert len(step.deps) == 1
    assert all("s3://" not in repr(dep) for dep in step.deps)
    assert "gpn-star-p" in training_tags(arm)
    assert "uniform-grid" in training_tags(arm)
    assert f"hf_revision={'a' * 40}" in training_tags(arm)
