"""Normalize issue #402 Levanter exports before public Hugging Face upload."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RAG_VOCAB = {
    "[PAD]": 0,
    "[UNK]": 1,
    "[BOS]": 2,
    "[SEQ]": 3,
    "a": 4,
    "c": 5,
    "g": 6,
    "t": 7,
}
RAG_ROPE_PARAMETERS = {
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8_192,
    "rope_type": "llama3",
    "rope_theta": 500_000,
}


def _read_json_object(path: Path) -> dict[str, Any]:
    assert path.is_file(), f"missing required HF export metadata: {path}"
    value = json.loads(path.read_text())
    assert isinstance(value, dict), f"expected a JSON object: {path}"
    return value


def _write_json_object(path: Path, value: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(path)


def normalize_rag_hf_export_metadata(model_dir: str | Path) -> None:
    """Make one staged RAG checkpoint safe for standard Transformers 4 and 5.

    The running experiment uses Transformers 5, whose Levanter exporter writes
    ``TokenizersBackend`` and ``rope_parameters``. Transformers 4 cannot resolve
    that tokenizer class and ignores the RoPE settings unless the equivalent
    legacy fields are present. This function changes metadata only; model
    weights remain byte-for-byte untouched.
    """
    model_path = Path(model_dir)
    assert model_path.is_dir(), f"model directory does not exist: {model_path}"

    tokenizer_json_path = model_path / "tokenizer.json"
    tokenizer_json = _read_json_object(tokenizer_json_path)
    model = tokenizer_json.get("model")
    assert isinstance(model, dict)
    assert model.get("type") == "WordLevel"
    assert model.get("vocab") == RAG_VOCAB
    assert model.get("unk_token") == "[UNK]"
    added_tokens = tokenizer_json.get("added_tokens")
    assert isinstance(added_tokens, list)
    assert {token["content"]: token["id"] for token in added_tokens} == {
        token: token_id
        for token, token_id in RAG_VOCAB.items()
        if token.startswith("[")
    }
    assert all(token.get("special") is True for token in added_tokens)

    tokenizer_config_path = model_path / "tokenizer_config.json"
    tokenizer_config = _read_json_object(tokenizer_config_path)
    observed_extra = tokenizer_config.pop("extra_special_tokens", None)
    observed_additional = tokenizer_config.get("additional_special_tokens")
    if observed_extra is not None:
        assert observed_extra == ["[SEQ]"]
    if observed_additional is not None:
        assert observed_additional == ["[SEQ]"]
    for token_field, expected in (
        ("bos_token", "[BOS]"),
        ("cls_token", "[BOS]"),
        ("pad_token", "[PAD]"),
        ("unk_token", "[UNK]"),
    ):
        observed = tokenizer_config.get(token_field)
        assert observed in (None, expected), f"unexpected {token_field}: {observed!r}"
        tokenizer_config[token_field] = expected
    tokenizer_config.update(
        {
            "additional_special_tokens": ["[SEQ]"],
            "clean_up_tokenization_spaces": False,
            "model_max_length": 2_048,
            "tokenizer_class": "PreTrainedTokenizerFast",
        }
    )
    _write_json_object(tokenizer_config_path, tokenizer_config)

    config_path = model_path / "config.json"
    config = _read_json_object(config_path)
    assert config.get("model_type") == "qwen3"
    assert config.get("vocab_size") == 8
    assert config.get("max_position_embeddings") == 2_048
    assert config.get("pad_token_id") == 0
    assert config.get("bos_token_id") == 2
    assert config.get("eos_token_id") is None
    assert config.get("rope_parameters") == RAG_ROPE_PARAMETERS

    legacy_rope_scaling = dict(RAG_ROPE_PARAMETERS)
    legacy_rope_theta = legacy_rope_scaling.pop("rope_theta")
    observed_theta = config.get("rope_theta")
    if observed_theta is not None:
        assert float(observed_theta) == float(legacy_rope_theta)
    observed_scaling = config.get("rope_scaling")
    if observed_scaling is not None:
        assert observed_scaling == legacy_rope_scaling
    config["rope_theta"] = legacy_rope_theta
    config["rope_scaling"] = legacy_rope_scaling
    _write_json_object(config_path, config)
