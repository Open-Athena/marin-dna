from __future__ import annotations

from dataclasses import replace

from exp517_functional_specialists.experiment import (
    BATCH_SIZE,
    DEFAULT_TPU_PREEMPTIBLE,
    DEFAULT_TPU_RAM,
    DEFAULT_TPU_REGION,
    DEFAULT_TPU_VARIANT,
    HF_SAVE_STEPS,
    MODEL_TOKENIZER_PATH,
    NATIVE_CHECKPOINT_STEPS,
    PER_DEVICE_PARALLELISM,
    SEED,
    SEQUENCE_LENGTH,
    TRAIN_STEPS,
    UNPUBLISHED_REVISION,
)
from exp517_functional_specialists.phylop_uniform_experiment import (
    ARMS,
    DATA_VERSION,
    PUBLICATION_PRODUCER_COMMIT,
    SOURCE_CONFIG_SHA256,
    SOURCE_PRODUCER_COMMIT,
    build_training,
    resolved_publication_commit,
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
        "gs://marin-us-east5/MarinDNA/exp517_phylop_uniform_specialists",
    )


def _published_arm(key: str = "cds"):
    return replace(ARMS[key], revision="a" * 40)


def test_six_phylop_uniform_arms_pin_publication() -> None:
    assert set(ARMS) == {
        "background",
        "cds",
        "enhancer_arm_a",
        "ncrna_exon",
        "tss_utr5",
        "utr3",
    }
    assert SOURCE_PRODUCER_COMMIT == "2162b6aa8299a9748eeb8031318b49072bb8c3fc"
    assert SOURCE_CONFIG_SHA256 == (
        "94d512050de327f96fda1105ce9c6ae5562944e402802516c7cde54795d8cdd1"
    )
    assert PUBLICATION_PRODUCER_COMMIT == (
        "fbc8968b14415b2722e7bcc4afaf95051acd6638"
    )
    assert resolved_publication_commit() == PUBLICATION_PRODUCER_COMMIT
    assert {key: arm.revision for key, arm in ARMS.items()} == {
        "cds": "452a5a3538f22630c3dea94d441ac30216bb28ea",
        "utr3": "2b73d5d9ebda34a361536db5e3d2697b6a1b1d6c",
        "tss_utr5": "5134205d86cd03e7833843d99e947e43e7aa11ac",
        "ncrna_exon": "54667e7bb49505f463afc147676e880a30c11d89",
        "enhancer_arm_a": "6f879b3747330e2c92e1402ead55cda6621f50ff",
        "background": "7d84519dccb4286622a14642a82a4f045d93a42c",
    }
    for arm in ARMS.values():
        assert arm.revision != UNPUBLISHED_REVISION
        assert arm.hf_repo.startswith("marin-dna/phylop-uniform-v1-")
        assert arm.resolved_revision() == arm.revision


def test_phylop_tokenization_reads_only_immutable_hugging_face_input(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "exp517_functional_specialists.phylop_uniform_experiment."
        "PUBLICATION_PRODUCER_COMMIT",
        "b" * 40,
    )
    arm = _published_arm()
    step = tokenized_dataset(arm)
    config = step.build_config(StepContext.for_fingerprint(deps=step.deps))
    assert config.id == "marin-dna/phylop-uniform-v1-cds"
    assert config.revision == "a" * 40
    assert config.format.text_key == "sequence"
    assert config.format.uppercase_weight == 1.0
    assert config.format.lowercase_weight == 0.01
    assert f"source={SOURCE_PRODUCER_COMMIT}" in config.tags
    assert f"source_config={SOURCE_CONFIG_SHA256}" in config.tags
    assert "strict-selector-control" in config.tags
    assert "s3://" not in repr(config)


def test_phylop_fixed_specialist_recipe(monkeypatch) -> None:
    _set_required_env(monkeypatch)
    monkeypatch.setattr(
        "exp517_functional_specialists.phylop_uniform_experiment."
        "PUBLICATION_PRODUCER_COMMIT",
        "b" * 40,
    )
    arm = _published_arm("background")
    step = build_training(arm)
    pod = step.build_config(
        StepContext.for_fingerprint(
            runtime_arg_keys=step.runtime_args,
            deps=step.deps,
        )
    )
    train = pod.train_config
    assert DATA_VERSION == "2026.08.27"
    assert train.trainer.train_batch_size == BATCH_SIZE == 8_192
    assert train.trainer.num_train_steps == TRAIN_STEPS == 5_000
    assert train.trainer.seed == SEED == 0
    assert train.trainer.steps_per_eval == HF_SAVE_STEPS == 500
    assert train.trainer.per_device_parallelism == PER_DEVICE_PARALLELISM == 1_024
    assert train.trainer.checkpointer.keep == [{"every": NATIVE_CHECKPOINT_STEPS}]
    assert train.train_seq_len == SEQUENCE_LENGTH == 256
    assert train.data_seed == SEED
    assert train.hf_save_steps == HF_SAVE_STEPS
    assert train.data.tokenizer == MODEL_TOKENIZER_PATH
    assert step.runtime_args["train_resources"].regions == [DEFAULT_TPU_REGION]
    assert step.runtime_args["train_resources"].device.variant == DEFAULT_TPU_VARIANT
    assert step.runtime_args["train_resources"].ram == DEFAULT_TPU_RAM
    assert step.runtime_args["train_resources"].preemptible is DEFAULT_TPU_PREEMPTIBLE
    assert len(step.deps) == 1
    assert all("s3://" not in repr(dep) for dep in step.deps)
    assert "WANDB_API_KEY" not in pod.env_vars
    assert "test-key" not in repr(pod)
    tags = training_tags(arm, DEFAULT_TPU_REGION)
    assert "phylop" in tags
    assert "strict-selector-control" in tags
    assert f"tpu_region={DEFAULT_TPU_REGION}" in tags
