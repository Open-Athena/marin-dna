from __future__ import annotations

import pytest
from exp473_center_seeded_projection.experiment import (
    ARMS,
    BATCH_SIZE,
    HF_SAVE_STEPS,
    MODEL,
    NATIVE_CHECKPOINT_STEPS,
    OPTIMIZER,
    PER_DEVICE_PARALLELISM,
    SEED,
    SEQUENCE_LENGTH,
    TOKENIZER_PATH,
    TOKENIZER_SHA256,
    TOKENIZER_SOURCE_REVISION,
    TRAIN_STEPS,
    DnaTokenizedCache,
    build_training,
    validate_vendored_tokenizer,
)
from exp473_center_seeded_projection.tokenizer_preflight import (
    TokenizerWorkerPreflightConfig,
    verify_tokenizer_on_worker,
)
from marin.execution.artifact import ArtifactRecord, write_record
from marin.execution.lazy import StepContext


def _set_required_env(monkeypatch) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setenv("WANDB_ENTITY", "test-entity")
    monkeypatch.setenv("WANDB_PROJECT", "marin")
    monkeypatch.setenv(
        "MARIN_PREFIX",
        "gs://marin-us-east5/MarinDNA/exp473_center_seeded_projection",
    )
    for name in [
        "EXP473_CENTER1_CDS_REVISION",
        "EXP473_FULLWINDOW_ENHANCER_REVISION",
        "EXP473_CENTER1_ENHANCER_REVISION",
    ]:
        monkeypatch.setenv(name, "e" * 40)


def test_all_four_arms_materialize_the_matched_recipe(monkeypatch) -> None:
    _set_required_env(monkeypatch)
    assert MODEL.max_seq_len == SEQUENCE_LENGTH
    assert MODEL.hidden_dim == 1_152
    assert MODEL.intermediate_dim == 4_608
    assert MODEL.num_layers == 12
    assert MODEL.num_heads == MODEL.num_kv_heads == 9
    assert MODEL.head_dim == 128
    assert MODEL.use_sliding_window is False
    assert MODEL.tie_word_embeddings is False
    assert MODEL.tokenizer == TOKENIZER_PATH
    assert MODEL.initializer_range == 0.02
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
    assert OPTIMIZER.min_lr_ratio == 0.0

    for key, arm in ARMS.items():
        step = build_training(arm)
        pod = step.build_config(
            StepContext.for_fingerprint(
                runtime_arg_keys=step.runtime_args,
                deps=step.deps,
            )
        )
        train = pod.train_config
        assert train.trainer.train_batch_size == BATCH_SIZE
        assert train.trainer.num_train_steps == TRAIN_STEPS
        assert train.trainer.seed == SEED
        assert train.trainer.steps_per_eval == HF_SAVE_STEPS
        assert train.trainer.per_device_parallelism == PER_DEVICE_PARALLELISM
        assert train.trainer.checkpointer.keep == [{"every": NATIVE_CHECKPOINT_STEPS}]
        assert train.train_seq_len == SEQUENCE_LENGTH
        assert train.data_seed == SEED
        assert train.hf_save_steps == HF_SAVE_STEPS
        assert train.z_loss_weight == 4.312883184368223e-06
        assert len(step.deps) == 1
        assert key in step.name


def test_tokenized_handles_pin_hf_revisions(monkeypatch) -> None:
    _set_required_env(monkeypatch)
    for arm in ARMS.values():
        step = build_training(arm)
        (cache,) = step.deps
        config = cache.build_config(
            StepContext.for_fingerprint(
                deps=cache.deps,
            ),
        )
        assert config.id == arm.hf_repo
        assert config.revision == arm.resolved_revision()
        assert len(config.revision) == 40
        assert config.tokenizer == TOKENIZER_PATH
        assert config.format.text_key == "sequence"
        assert config.format.uppercase_weight == 1.0
        assert config.format.lowercase_weight == 0.01
        assert f"tokenizer_revision={TOKENIZER_SOURCE_REVISION}" in config.tags
        for name, digest in TOKENIZER_SHA256.items():
            assert f"{name}_sha256={digest}" in config.tags
        assert cache.run.env_vars == {
            "HF_HUB_DOWNLOAD_TIMEOUT": "120",
            "UV_LOCK_TIMEOUT": "7200",
        }


def test_vendored_tokenizer_matches_issue_417() -> None:
    assert set(TOKENIZER_SHA256) == {
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    validate_vendored_tokenizer()


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
                "tags": ["exp473", "case-aware"],
            },
        )
    )
    cache = DnaTokenizedCache.raw_load(str(tmp_path))
    component = cache.as_component()
    assert cache.tokenizer == TOKENIZER_PATH
    assert cache.tags == ["exp473", "case-aware"]
    assert component.format.text_key == "sequence"
    assert component.format.uppercase_weight == 1.0
    assert component.format.lowercase_weight == 0.01

    for wrong_format in (
        {
            "text_key": "text",
            "uppercase_weight": 1.0,
            "lowercase_weight": 0.01,
        },
        {
            "text_key": "sequence",
            "uppercase_weight": 1.0,
            "lowercase_weight": 1.0,
        },
    ):
        bad_path = (
            tmp_path / wrong_format["text_key"] / str(wrong_format["lowercase_weight"])
        )
        bad_path.mkdir(parents=True)
        write_record(
            ArtifactRecord(
                output_path=str(bad_path),
                config={"tokenizer": TOKENIZER_PATH, "format": wrong_format},
            )
        )
        bad_cache = DnaTokenizedCache.raw_load(str(bad_path))
        with pytest.raises(ValueError, match="tokenized cache format"):
            _ = bad_cache.format


def test_tokenizer_worker_preflight_exercises_exact_character_contract(
    monkeypatch,
) -> None:
    class StubHfTokenizer:
        def __call__(self, text, **_kwargs):
            assert text == "ACGTacgt"
            return {"input_ids": [2, 3, 4, 5, 6, 3, 4, 5, 6]}

    class StubTokenizer:
        vocab_size = 7
        bos_token_id = 2
        pad_token_id = 0
        unk_token_id = 1
        eos_token_id = None

        def as_hf_tokenizer(self):
            return StubHfTokenizer()

    monkeypatch.setattr(
        "exp473_center_seeded_projection.tokenizer_preflight.load_tokenizer",
        lambda _: StubTokenizer(),
    )
    verify_tokenizer_on_worker(
        TokenizerWorkerPreflightConfig(TOKENIZER_PATH, dict(TOKENIZER_SHA256))
    )
