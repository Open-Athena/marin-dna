"""Focused invariants for the issue #402 experiment recipe."""

import math
import tempfile

import pytest
from levanter.tokenizers import load_tokenizer
from marin.execution.artifact import ArtifactRecord, result_type_name, write_record
from marin.execution.build_context import BuildContext, VersionCodex, build_context
from marin.execution.lazy import StepContext, materialized_config
from marin_dna.levanter.formats import RAGDNALmDatasetFormat
from marin_dna.pipelines.rag_glm.dataset import MISSING_SEQUENCE, assemble_document

from launch import (
    ACTUAL_TOKENS,
    CHECKPOINT_NAME,
    HF_SAVE_EVERY,
    MARIN_DNA_REVISION,
    MENDELIAN_TRAITS_RAG_255,
    MODEL,
    ONLINE_EVAL_ENV,
    OPTIMIZER,
    RAG_EVAL_EVERY,
    SEQ_LEN,
    TARGET_TOKENS,
    TOKENIZER_PATH,
    TOKENIZER_SHA256,
    TRAIN_BATCH_SIZE,
    TRAIN_HOST_CPU,
    TRAIN_HOST_RAM,
    TRAIN_REGIONS,
    TRAIN_STEPS,
    TRAIN_TPU,
    VOCAB_SIZE,
    RAGTokenizedCache,
    build,
    online_eval_enabled,
    rag_tokenized_dataset,
)


def test_online_eval_is_the_pinned_mendelian_rag_task() -> None:
    assert MARIN_DNA_REVISION == "eaac2efffb73d33b87ba75bcf5521809af74fec7"
    assert MENDELIAN_TRAITS_RAG_255.name == "mendelian_traits_rag_255"
    assert MENDELIAN_TRAITS_RAG_255.num_fewshot == 0
    assert RAG_EVAL_EVERY == 1_000
    assert TRAIN_STEPS // RAG_EVAL_EVERY == 7
    assert TRAIN_TPU == ("v6e-4", "v5p-8")
    assert TRAIN_REGIONS == ("us-east5",)
    assert TRAIN_HOST_CPU == 16
    assert TRAIN_HOST_RAM == "64g"


def test_online_eval_can_be_disabled_for_offline_scoring(monkeypatch) -> None:
    monkeypatch.delenv(ONLINE_EVAL_ENV, raising=False)
    assert online_eval_enabled()
    monkeypatch.setenv(ONLINE_EVAL_ENV, "0")
    assert not online_eval_enabled()
    monkeypatch.setenv(ONLINE_EVAL_ENV, "invalid")
    with pytest.raises(AssertionError, match="must be 0 or 1"):
        online_eval_enabled()


def test_offline_mode_exports_every_thousand_steps(monkeypatch) -> None:
    monkeypatch.setenv(ONLINE_EVAL_ENV, "0")
    versions = VersionCodex(
        default="2026.07.24",
        overrides={CHECKPOINT_NAME: "test-dev"},
    )
    with build_context(BuildContext(versions=versions)):
        training = build()
    ctx = StepContext.for_fingerprint(
        training.runtime_args.keys(),
        training.deps,
    )
    pod_config = training.build_config(ctx)
    assert pod_config.train_config.hf_save_steps == HF_SAVE_EVERY == 1_000
    assert pod_config.train_config.eval_harness is None


def test_model_is_the_46m_rung_at_full_document_length() -> None:
    assert MODEL.max_seq_len == SEQ_LEN == 2_048
    assert MODEL.hidden_dim == 640
    assert MODEL.intermediate_dim == 2_560
    assert MODEL.num_layers == 7
    assert MODEL.num_heads == MODEL.num_kv_heads == 5
    assert 45_000_000 < MODEL.total_trainable_params(VOCAB_SIZE) < 47_000_000


def test_resolved_training_horizon_and_adamh_values() -> None:
    assert TRAIN_STEPS == 7_629
    assert ACTUAL_TOKENS == 999_948_288
    assert abs(ACTUAL_TOKENS - TARGET_TOKENS) < TRAIN_BATCH_SIZE * SEQ_LEN
    assert math.isclose(OPTIMIZER.learning_rate, 0.008293207887305696)
    assert math.isclose(OPTIMIZER.adam_lr, 0.0010372270725352284)
    assert math.isclose(OPTIMIZER.epsilon, 1.1700427342623003e-08)
    assert OPTIMIZER.beta1 == 0.9
    assert OPTIMIZER.beta2 == 0.9999
    assert OPTIMIZER.max_grad_norm == 0.1


def test_tokenize_config_pins_data_and_fixed_layout() -> None:
    handle = rag_tokenized_dataset()
    config = materialized_config(handle, "gs://example-prefix")
    assert config.id == "bolinas-dna/zoonomia-rag-v1-v1"
    assert config.revision == "5e6b30cf878b61c99e6432ad8ab7865b18cbe0e7"
    assert config.tokenizer == TOKENIZER_PATH == "tokenizer"
    assert config.format.text_key == "seq"
    assert config.max_workers == 32
    assert all(
        f"{filename}-sha256={digest}" in config.tags
        for filename, digest in TOKENIZER_SHA256.items()
    )


def test_tokenized_cache_reload_preserves_rag_loss_format() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        write_record(
            ArtifactRecord(
                name="datasets/dna-exp402-rag-tokenized",
                output_path=tmpdir,
                result_type=result_type_name(RAGTokenizedCache),
                config={
                    "tokenizer": TOKENIZER_PATH,
                    "format": {
                        "text_key": "seq",
                        "uppercase_weight": 1.0,
                        "lowercase_weight": 1.0,
                    },
                },
            )
        )
        cache = RAGTokenizedCache.raw_load(tmpdir)
        component = cache.as_component()

    assert isinstance(component.format, RAGDNALmDatasetFormat)
    assert component.format.text_key == "seq"


def test_vendored_tokenizer_runs_fixed_layout_preprocessor() -> None:
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    document = assemble_document(
        (
            "A" * 255,
            "C" * 255,
            "G" * 255,
            "T" * 255,
            MISSING_SEQUENCE,
            "A" * 255,
            "C" * 255,
            "G" * 255,
        )
    )
    row = RAGDNALmDatasetFormat().build_preprocessor(tokenizer)([{"seq": document}])[0]
    assert row["input_ids"].shape == (2_048,)
    assert row["loss_weight"].shape == (2_048,)
    assert row["input_ids"][0] == tokenizer.bos_token_id == 2
    assert row["input_ids"][256] == tokenizer.convert_tokens_to_ids("[SEQ]") == 3
    assert row["loss_weight"].tolist() == [1.0] * 2_047 + [0.0]
