from __future__ import annotations

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
)
from exp517_functional_specialists.phylop_enhancer_order_experiment import (
    DATA_VERSION,
    EFFECTIVE_ROW_EPOCHS,
    HF_REPO,
    HF_REVISION,
    PUBLICATION_CONFIG_SHA256,
    PUBLICATION_PRODUCER_COMMIT,
    RUN_ID,
    SOURCE_ROWS,
    TRAIN_ROWS,
    VALIDATION_ROWS,
    build_training,
    resolved_hf_revision,
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
        "gs://marin-us-east5/MarinDNA/exp517_phylop_enhancer_order",
    )


def test_order_control_pins_public_release_and_exposure() -> None:
    assert HF_REPO == "marin-dna/phylop-uniform-v1-enhancer-arm-a-vertebrate-order"
    assert HF_REVISION == "6a592fffcdd155d19e6c8e0986eab606aab19606"
    assert PUBLICATION_PRODUCER_COMMIT == (
        "90b86f6426c919470f0eb26e1b1aa2cab6a261ed"
    )
    assert PUBLICATION_CONFIG_SHA256 == (
        "a5d7ff16ecc2b4574e4803e4858392ffa00aefba17da1e155c4859355ad7b437"
    )
    assert resolved_hf_revision() == HF_REVISION
    assert resolved_publication_commit() == PUBLICATION_PRODUCER_COMMIT
    assert SOURCE_ROWS == 7_876_044
    assert TRAIN_ROWS == 15_719_320
    assert VALIDATION_ROWS == 16_384
    assert EFFECTIVE_ROW_EPOCHS == 40_960_000 / 15_719_320


def test_order_control_tokenization_reads_only_immutable_hub_input() -> None:
    step = tokenized_dataset()
    config = step.build_config(StepContext.for_fingerprint(deps=step.deps))
    assert config.id == HF_REPO
    assert config.revision == HF_REVISION
    assert config.format.text_key == "sequence"
    assert config.format.uppercase_weight == 1.0
    assert config.format.lowercase_weight == 0.01
    assert f"publication={PUBLICATION_PRODUCER_COMMIT}" in config.tags
    assert f"publication_config={PUBLICATION_CONFIG_SHA256}" in config.tags
    assert "one-per-ncbi-order" in config.tags
    assert config.num_shards == 64
    assert "s3://" not in repr(config)


def test_order_control_fixed_training_recipe(monkeypatch) -> None:
    _set_required_env(monkeypatch)
    step = build_training()
    pod = step.build_config(
        StepContext.for_fingerprint(
            runtime_arg_keys=step.runtime_args,
            deps=step.deps,
        )
    )
    train = pod.train_config
    assert DATA_VERSION == "2026.09.04"
    assert RUN_ID == "dna-exp517-phylop-uniform-0p25b-enhancer-order-v1"
    assert train.trainer.id == RUN_ID
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
    assert (
        step.runtime_args["train_resources"].preemptible
        is DEFAULT_TPU_PREEMPTIBLE
    )
    assert len(step.deps) == 1
    assert all("s3://" not in repr(dep) for dep in step.deps)
    assert "WANDB_API_KEY" not in pod.env_vars
    assert "test-key" not in repr(pod)
    tags = training_tags(DEFAULT_TPU_REGION)
    assert "one-per-ncbi-order" in tags
    assert "order-exposure-control" in tags
    assert f"hf_revision={HF_REVISION}" in tags
