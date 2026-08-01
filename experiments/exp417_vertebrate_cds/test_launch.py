"""Focused invariants for the issue #417 matched CDS experiment."""

import math
import tempfile

import pytest
from levanter.tokenizers import load_tokenizer
from marin.execution.artifact import ArtifactRecord, result_type_name, write_record
from marin.execution.build_context import BuildContext, VersionCodex, build_context
from marin.execution.lazy import StepContext, materialized_config
from marin_dna.levanter.formats import DNALmDatasetFormat

from launch import (
    ACTUAL_TOKENS,
    ARM_ENV,
    ARMS,
    BETA1,
    BETA2,
    DATASET_REPOS,
    DATASET_REVISION_ENVS,
    DNA_BASE_SEQ_LEN,
    EPSILON,
    HEAD_DIM,
    HF_SAVE_EVERY,
    HIDDEN_DIM,
    INTERMEDIATE_DIM,
    LEARNING_RATE,
    MARIN_DNA_REVISION,
    MAX_GRAD_NORM,
    MODEL,
    NATIVE_CHECKPOINT_EVERY,
    NUM_HEADS,
    NUM_LAYERS,
    OPTIMIZER,
    PER_DEVICE_PARALLELISM,
    PROJECTION_PIPELINE_REVISION,
    REPEAT_MASK_LOSS_WEIGHT,
    RUN_IDS,
    SEED,
    SEQ_LEN,
    TOKENIZER_PATH,
    TOKENIZER_SHA256,
    TOKENIZER_SOURCE_REVISION,
    TRAIN_BATCH_SIZE,
    TRAIN_HOST_CPU,
    TRAIN_HOST_RAM,
    TRAIN_REGIONS,
    TRAIN_STEPS,
    TRAIN_TPU,
    VALIDATION_EVERY,
    VOCAB_SIZE,
    WANDB_GROUP,
    Z_LOSS_WEIGHT,
    VertebrateCDSTokenizedCache,
    build,
    dataset_revision,
    selected_arms,
    tokenized_dataset,
    validate_vendored_tokenizer,
)

FAKE_REVISIONS = {
    "mammals_only": "1" * 40,
    "combined_vertebrates": "2" * 40,
}


def _set_revisions(monkeypatch: pytest.MonkeyPatch) -> None:
    for arm, revision in FAKE_REVISIONS.items():
        monkeypatch.setenv(DATASET_REVISION_ENVS[arm], revision)


def test_source_revisions_and_arm_names_are_explicit() -> None:
    assert MARIN_DNA_REVISION == "eaac2efffb73d33b87ba75bcf5521809af74fec7"
    assert PROJECTION_PIPELINE_REVISION == "06549d8f7f3ba76151b9c54a5e52d3e3f4402a2d"
    assert TOKENIZER_SOURCE_REVISION == "a73e9d9ee636f722b4c378703c9e2997857809b2"
    assert ARMS == ("mammals_only", "combined_vertebrates")
    assert set(DATASET_REPOS) == set(ARMS)
    assert set(RUN_IDS) == set(ARMS)
    assert all("dna-exp417" in run_id for run_id in RUN_IDS.values())
    assert WANDB_GROUP == "dna-exp417-v1"


def test_dataset_revisions_must_be_immutable_commits(monkeypatch: pytest.MonkeyPatch) -> None:
    env_name = DATASET_REVISION_ENVS["mammals_only"]
    monkeypatch.delenv(env_name, raising=False)
    with pytest.raises(AssertionError, match=env_name):
        dataset_revision("mammals_only")
    monkeypatch.setenv(env_name, "main")
    with pytest.raises(AssertionError, match=env_name):
        dataset_revision("mammals_only")
    monkeypatch.setenv(env_name, FAKE_REVISIONS["mammals_only"])
    assert dataset_revision("mammals_only") == FAKE_REVISIONS["mammals_only"]


def test_arm_selection_is_ordered_and_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ARM_ENV, raising=False)
    assert selected_arms() == ARMS
    monkeypatch.setenv(ARM_ENV, "combined_vertebrates,mammals_only")
    assert selected_arms() == ARMS
    monkeypatch.setenv(ARM_ENV, "mammals_only")
    assert selected_arms() == ("mammals_only",)
    monkeypatch.setenv(ARM_ENV, "unknown")
    with pytest.raises(AssertionError, match="unknown"):
        selected_arms()


def test_repeat_mask_weights_are_target_aligned() -> None:
    validate_vendored_tokenizer()
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    fmt = DNALmDatasetFormat(
        text_key="sequence",
        uppercase_weight=1.0,
        lowercase_weight=REPEAT_MASK_LOSS_WEIGHT,
    )
    row = fmt.build_preprocessor(tokenizer)([{"sequence": "AaCcGgTt"}])[0]

    assert row["input_ids"].shape == (9,)
    assert row["input_ids"][0] == tokenizer.bos_token_id == 2
    assert row["loss_weight"].tolist() == pytest.approx(
        [
            1.0,
            REPEAT_MASK_LOSS_WEIGHT,
            1.0,
            REPEAT_MASK_LOSS_WEIGHT,
            1.0,
            REPEAT_MASK_LOSS_WEIGHT,
            1.0,
            REPEAT_MASK_LOSS_WEIGHT,
            0.0,
        ]
    )


def test_tokenize_config_applies_one_repeat_aware_format_to_both_splits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_revisions(monkeypatch)
    for arm in ARMS:
        config = materialized_config(tokenized_dataset(arm), "gs://example-prefix")
        assert config.id == DATASET_REPOS[arm]
        assert config.revision == FAKE_REVISIONS[arm]
        assert config.tokenizer == TOKENIZER_PATH
        assert config.format.text_key == "sequence"
        assert config.format.uppercase_weight == 1.0
        assert config.format.lowercase_weight == REPEAT_MASK_LOSS_WEIGHT == 0.01
        assert config.max_workers == 32
        assert f"repeat-mask-lowercase-weight={REPEAT_MASK_LOSS_WEIGHT}" in config.tags


def test_tokenized_cache_reload_preserves_repeat_aware_format() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        write_record(
            ArtifactRecord(
                name="datasets/dna-exp417-cds-mammals_only-tokenized",
                output_path=tmpdir,
                result_type=result_type_name(VertebrateCDSTokenizedCache),
                config={
                    "tokenizer": TOKENIZER_PATH,
                    "format": {
                        "text_key": "sequence",
                        "uppercase_weight": 1.0,
                        "lowercase_weight": REPEAT_MASK_LOSS_WEIGHT,
                    },
                },
            )
        )
        cache = VertebrateCDSTokenizedCache.raw_load(tmpdir)
        component = cache.as_component()

    assert isinstance(component.format, DNALmDatasetFormat)
    assert component.format.text_key == "sequence"
    assert component.format.uppercase_weight == 1.0
    assert component.format.lowercase_weight == REPEAT_MASK_LOSS_WEIGHT


def test_model_and_optimizer_match_exp353_recipe() -> None:
    assert DNA_BASE_SEQ_LEN == 255
    assert MODEL.max_seq_len == SEQ_LEN == 256
    assert MODEL.hidden_dim == HIDDEN_DIM == 1_152
    assert MODEL.intermediate_dim == INTERMEDIATE_DIM == 4_608
    assert MODEL.num_layers == NUM_LAYERS == 12
    assert MODEL.num_heads == MODEL.num_kv_heads == NUM_HEADS == 9
    assert MODEL.head_dim == HEAD_DIM == 128
    assert MODEL.initializer_range == 0.02
    assert 250_000_000 < MODEL.total_trainable_params(VOCAB_SIZE) < 260_000_000

    assert math.isclose(OPTIMIZER.learning_rate, LEARNING_RATE)
    assert math.isclose(OPTIMIZER.beta1, BETA1)
    assert math.isclose(OPTIMIZER.beta2, BETA2)
    assert math.isclose(OPTIMIZER.epsilon, EPSILON)
    assert math.isclose(OPTIMIZER.max_grad_norm, MAX_GRAD_NORM)
    assert OPTIMIZER.weight_decay == 0.1
    assert OPTIMIZER.warmup == 0.1
    assert OPTIMIZER.decay == 0.2
    assert OPTIMIZER.lr_schedule == "linear"
    assert OPTIMIZER.min_lr_ratio == 0.0


def test_both_lowered_training_arms_are_matched(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_revisions(monkeypatch)
    monkeypatch.delenv(ARM_ENV, raising=False)
    versions = VersionCodex(default="2026.08.01")
    with build_context(BuildContext(versions=versions)):
        arms = build()

    assert tuple(arms) == ARMS
    for arm, training in arms.items():
        ctx = StepContext.for_fingerprint(training.runtime_args.keys(), training.deps)
        pod_config = training.build_config(ctx)
        config = pod_config.train_config
        trainer = config.trainer

        assert len(training.deps) == 1
        assert training.deps[0].name == f"datasets/dna-exp417-cds-{arm}-tokenized"
        assert config.model == MODEL
        assert config.optimizer == OPTIMIZER
        assert config.train_seq_len == SEQ_LEN
        assert config.z_loss_weight == Z_LOSS_WEIGHT
        assert config.eval_harness is None
        assert config.data_seed == trainer.seed == SEED == 0
        assert trainer.train_batch_size == TRAIN_BATCH_SIZE
        assert trainer.num_train_steps == TRAIN_STEPS
        assert trainer.steps_per_eval == VALIDATION_EVERY
        assert trainer.per_device_parallelism == PER_DEVICE_PARALLELISM
        assert trainer.checkpointer.keep == [{"every": NATIVE_CHECKPOINT_EVERY}]
        assert config.hf_save_steps == HF_SAVE_EVERY
        assert trainer.id == RUN_IDS[arm]

    assert TRAIN_TPU == ("v6e-4", "v5p-8")
    assert TRAIN_REGIONS == ("us-east5",)
    assert TRAIN_HOST_CPU == 16
    assert TRAIN_HOST_RAM == "56g"
    assert TRAIN_BATCH_SIZE == 8_192
    assert TRAIN_STEPS == 5_000
    assert ACTUAL_TOKENS == 10_485_760_000
    assert VALIDATION_EVERY == NATIVE_CHECKPOINT_EVERY == HF_SAVE_EVERY == 500


def test_tokenizer_digests_are_complete() -> None:
    assert set(TOKENIZER_SHA256) == {
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    validate_vendored_tokenizer()
