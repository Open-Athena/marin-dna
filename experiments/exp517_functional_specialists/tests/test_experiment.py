from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from exp517_functional_specialists.experiment import (
    ARMS,
    BATCH_SIZE,
    DEFAULT_TPU_PREEMPTIBLE,
    DEFAULT_TPU_RAM,
    DEFAULT_TPU_REGION,
    DEFAULT_TPU_VARIANT,
    HF_SAVE_STEPS,
    MODEL,
    MODEL_TOKENIZER_PATH,
    NATIVE_CHECKPOINT_STEPS,
    OPTIMIZER,
    PER_DEVICE_PARALLELISM,
    PRODUCER_COMMIT,
    SEED,
    SEQUENCE_LENGTH,
    TOKENIZER_PATH,
    TOKENIZER_SHA256,
    TOKENIZER_SOURCE_REVISION,
    TRAIN_STEPS,
    UNPUBLISHED_REVISION,
    DnaTokenizedCache,
    build_training,
    tokenized_dataset,
    training_tags,
    validate_vendored_tokenizer,
)
from exp517_functional_specialists.tokenizer_preflight import (
    TokenizerWorkerPreflightConfig,
    verify_tokenizer_on_worker,
)
from marin.execution.artifact import ArtifactRecord, write_record
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
        "gs://marin-us-east5/MarinDNA/exp517_functional_specialists",
    )


def _published_arm(key: str = "cds"):
    return replace(ARMS[key], revision="a" * 40)


def test_five_hub_arms_fail_closed_until_publication() -> None:
    assert {key: arm.hf_repo for key, arm in ARMS.items()} == {
        "cds": "marin-dna/functional-cds",
        "utr3": "marin-dna/functional-utr3",
        "tss_region": "marin-dna/functional-tss",
        "ncrna": "marin-dna/functional-ncrna",
        "enhancer": "marin-dna/functional-enhancer",
    }
    assert {arm.revision for arm in ARMS.values()} == {UNPUBLISHED_REVISION}
    for arm in ARMS.values():
        with pytest.raises(ValueError, match="40-character hexadecimal"):
            arm.resolved_revision()
        with pytest.raises(ValueError, match="40-character hexadecimal"):
            replace(arm, revision="z" * 40).resolved_revision()


def test_tokenization_reads_only_immutable_hugging_face_input() -> None:
    arm = _published_arm()
    step = tokenized_dataset(arm)
    config = step.build_config(StepContext.for_fingerprint(deps=step.deps))
    assert config.id == "marin-dna/functional-cds"
    assert config.revision == "a" * 40
    assert config.tokenizer == TOKENIZER_PATH
    assert config.format.text_key == "sequence"
    assert config.format.uppercase_weight == 1.0
    assert config.format.lowercase_weight == 0.01
    assert f"tokenizer_revision={TOKENIZER_SOURCE_REVISION}" in config.tags
    assert f"producer={PRODUCER_COMMIT}" in config.tags
    assert "s3://" not in repr(config)
    assert step.run.env_vars == {
        "HF_HUB_DOWNLOAD_TIMEOUT": "120",
        "UV_LOCK_TIMEOUT": "7200",
    }


def test_fixed_specialist_recipe(monkeypatch) -> None:
    _set_required_env(monkeypatch)
    arm = _published_arm()
    step = build_training(arm)
    pod = step.build_config(
        StepContext.for_fingerprint(
            runtime_arg_keys=step.runtime_args,
            deps=step.deps,
        )
    )
    train = pod.train_config
    assert MODEL.max_seq_len == SEQUENCE_LENGTH == 256
    assert MODEL.hidden_dim == 1_152
    assert MODEL.intermediate_dim == 4_608
    assert MODEL.num_layers == 12
    assert MODEL.num_heads == MODEL.num_kv_heads == 9
    assert MODEL.head_dim == 128
    assert MODEL.use_sliding_window is False
    assert MODEL.tie_word_embeddings is False
    assert MODEL.tokenizer == MODEL_TOKENIZER_PATH
    assert (
        Path(MODEL_TOKENIZER_PATH) == Path(__file__).resolve().parents[1] / "tokenizer"
    )
    assert 250_000_000 < MODEL.total_trainable_params(7) < 260_000_000
    assert OPTIMIZER.learning_rate == 0.00430097
    assert OPTIMIZER.weight_decay == 0.1
    assert OPTIMIZER.beta1 == 0.66756
    assert OPTIMIZER.beta2 == 0.952222
    assert OPTIMIZER.epsilon == 6.77142e-15
    assert OPTIMIZER.max_grad_norm == 0.995188
    assert OPTIMIZER.warmup == 0.1
    assert OPTIMIZER.decay == 0.2
    assert OPTIMIZER.lr_schedule == "linear"
    assert train.trainer.train_batch_size == BATCH_SIZE == 8_192
    assert train.trainer.num_train_steps == TRAIN_STEPS == 5_000
    assert train.trainer.seed == SEED == 0
    assert train.trainer.steps_per_eval == HF_SAVE_STEPS == 500
    assert train.trainer.per_device_parallelism == PER_DEVICE_PARALLELISM == 1_024
    assert train.trainer.checkpointer.keep == [{"every": NATIVE_CHECKPOINT_STEPS}]
    assert train.train_seq_len == SEQUENCE_LENGTH
    assert train.data_seed == SEED
    assert train.hf_save_steps == HF_SAVE_STEPS
    assert step.runtime_args["train_resources"].regions == [DEFAULT_TPU_REGION]
    assert step.runtime_args["train_resources"].device.variant == DEFAULT_TPU_VARIANT
    assert step.runtime_args["train_resources"].ram == DEFAULT_TPU_RAM
    assert step.runtime_args["train_resources"].preemptible is DEFAULT_TPU_PREEMPTIBLE
    assert len(step.deps) == 1
    assert all("s3://" not in repr(dep) for dep in step.deps)
    assert f"hf_revision={'a' * 40}" in training_tags(arm)


def test_tpu_resource_overrides_are_bounded(monkeypatch) -> None:
    _set_required_env(monkeypatch)
    arm = _published_arm()
    monkeypatch.setenv("EXP517_TPU_VARIANT", "v5p-8,v6e-4")
    step = build_training(arm)
    resources = step.runtime_args["train_resources"]
    assert resources.device.variant == "v5p-8"
    assert resources.device_alternatives == ["v6e-4"]

    monkeypatch.setenv("EXP517_TPU_PREEMPTIBLE", "false")
    assert build_training(arm).runtime_args["train_resources"].preemptible is False
    monkeypatch.setenv("EXP517_TPU_PREEMPTIBLE", "sometimes")
    with pytest.raises(ValueError, match="EXP517_TPU_PREEMPTIBLE"):
        build_training(arm)

    monkeypatch.setenv("EXP517_TPU_PREEMPTIBLE", "true")
    monkeypatch.setenv("EXP517_TPU_RAM", "128g")
    with pytest.raises(ValueError, match="EXP517_TPU_RAM"):
        build_training(arm)

    monkeypatch.setenv("EXP517_TPU_RAM", DEFAULT_TPU_RAM)
    monkeypatch.setenv("EXP517_TPU_REGION", "us-east1")
    monkeypatch.setenv("EXP517_TPU_VARIANT", "v6e-4")
    monkeypatch.setenv(
        "MARIN_PREFIX",
        "gs://marin-us-east5/MarinDNA/exp517_functional_specialists",
    )
    with pytest.raises(ValueError, match="MARIN_PREFIX"):
        build_training(arm)


def test_vendored_tokenizer_and_worker_contract(monkeypatch) -> None:
    assert set(TOKENIZER_SHA256) == {
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    validate_vendored_tokenizer()

    class StubHfTokenizer:
        vocab_size = 7
        bos_token_id = 2
        pad_token_id = 0
        unk_token_id = 1
        eos_token_id = None

        def __call__(self, text, **_kwargs):
            assert text == "ACGTacgt"
            return {"input_ids": [2, 3, 4, 5, 6, 3, 4, 5, 6]}

    class StubTokenizer:
        def as_hf_tokenizer(self):
            return StubHfTokenizer()

    monkeypatch.setattr(
        "exp517_functional_specialists.tokenizer_preflight.load_tokenizer",
        lambda _: StubTokenizer(),
    )
    verify_tokenizer_on_worker(
        TokenizerWorkerPreflightConfig(TOKENIZER_PATH, dict(TOKENIZER_SHA256))
    )


def test_realized_cache_reloads_exact_case_aware_format(tmp_path) -> None:
    write_record(
        ArtifactRecord(
            output_path=str(tmp_path),
            config={
                "tokenizer": TOKENIZER_PATH,
                "format": {
                    "text_key": "sequence",
                    "uppercase_weight": 1.0,
                    "lowercase_weight": 0.01,
                },
                "tags": ["exp517", "hf-only"],
            },
        )
    )
    cache = DnaTokenizedCache.raw_load(str(tmp_path))
    component = cache.as_component()
    assert cache.tokenizer == TOKENIZER_PATH
    assert cache.tags == ["exp517", "hf-only"]
    assert component.format.text_key == "sequence"
    assert component.format.uppercase_weight == 1.0
    assert component.format.lowercase_weight == 0.01

    bad_path = tmp_path / "bad"
    bad_path.mkdir()
    write_record(
        ArtifactRecord(
            output_path=str(bad_path),
            config={
                "tokenizer": TOKENIZER_PATH,
                "format": {
                    "text_key": "sequence",
                    "uppercase_weight": 1.0,
                    "lowercase_weight": 1.0,
                },
            },
        )
    )
    with pytest.raises(ValueError, match="tokenized cache format"):
        _ = DnaTokenizedCache.raw_load(str(bad_path)).format
