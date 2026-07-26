"""Tests for issue #402 Hugging Face publication metadata."""

from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

from marin_dna.pipelines.rag_glm.hf_publication import (
    RAG_ROPE_PARAMETERS,
    normalize_rag_hf_export_metadata,
)
from marin_dna.pipelines.rag_glm.offline_eval import load_rag_model_config_hf
from marin_dna.pipelines.rag_glm.tokenizer import create_rag_char_tokenizer


def _write_transformers_5_export(tmp_path: Path) -> None:
    tokenizer = create_rag_char_tokenizer()
    tokenizer.save_pretrained(tmp_path)
    tokenizer_config_path = tmp_path / "tokenizer_config.json"
    tokenizer_config = json.loads(tokenizer_config_path.read_text())
    tokenizer_config["tokenizer_class"] = "TokenizersBackend"
    tokenizer_config["extra_special_tokens"] = ["[SEQ]"]
    tokenizer_config.pop("additional_special_tokens", None)
    tokenizer_config["model_max_length"] = 10**30
    tokenizer_config_path.write_text(json.dumps(tokenizer_config))

    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "vocab_size": 8,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "max_position_embeddings": 2_048,
        "pad_token_id": 0,
        "bos_token_id": 2,
        "eos_token_id": None,
        "rope_parameters": RAG_ROPE_PARAMETERS,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))


def test_normalize_transformers_5_export_for_standard_loaders(tmp_path: Path) -> None:
    _write_transformers_5_export(tmp_path)

    normalize_rag_hf_export_metadata(tmp_path)

    tokenizer = AutoTokenizer.from_pretrained(tmp_path)
    assert len(tokenizer) == 8
    assert tokenizer.model_max_length == 2_048
    assert tokenizer("ACGT", add_special_tokens=False)["input_ids"] == [4, 5, 6, 7]
    assert tokenizer.bos_token_id == 2
    assert tokenizer.convert_tokens_to_ids("[SEQ]") == 3

    standard_config = AutoConfig.from_pretrained(tmp_path)
    standard_scaling = dict(standard_config.rope_scaling)
    standard_theta = getattr(standard_config, "rope_theta", None)
    if standard_theta is None:
        standard_theta = standard_scaling.pop("rope_theta")
    assert standard_theta == 500_000
    expected_scaling = {
        key: value for key, value in RAG_ROPE_PARAMETERS.items() if key != "rope_theta"
    }
    assert standard_scaling == expected_scaling
    adapted_config = load_rag_model_config_hf(tmp_path)
    adapted_scaling = dict(adapted_config.rope_scaling)
    adapted_theta = getattr(adapted_config, "rope_theta", None)
    if adapted_theta is None:
        adapted_theta = adapted_scaling.pop("rope_theta")
    assert adapted_theta == 500_000
    assert adapted_scaling == expected_scaling


def test_publication_normalization_is_idempotent(tmp_path: Path) -> None:
    _write_transformers_5_export(tmp_path)
    normalize_rag_hf_export_metadata(tmp_path)
    first_config = (tmp_path / "config.json").read_bytes()
    first_tokenizer_config = (tmp_path / "tokenizer_config.json").read_bytes()

    normalize_rag_hf_export_metadata(tmp_path)

    assert (tmp_path / "config.json").read_bytes() == first_config
    assert (tmp_path / "tokenizer_config.json").read_bytes() == first_tokenizer_config
