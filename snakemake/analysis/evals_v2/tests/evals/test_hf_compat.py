"""Regression tests for the Transformers 4/5 HF checkpoint boundary."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from marin_dna_evals.hf_compat import (
    HfCheckpointCompatibilityError,
    load_hf_base_model_and_tokenizer,
    load_hf_causal_lm_and_tokenizer,
    load_hf_checkpoint_config,
    load_hf_checkpoint_tokenizer,
)

LLAMA3_SCALING = {
    "rope_type": "llama3",
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}
LLAMA3_PARAMETERS = {"rope_theta": 500000, **LLAMA3_SCALING}


def _write_config(tmp_path: Path, **rope_fields: object) -> None:
    config = {"model_type": "qwen3", **rope_fields}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")


def test_load_config_preserves_transformers4_rope(tmp_path: Path) -> None:
    _write_config(tmp_path, rope_theta=500000, rope_scaling=LLAMA3_SCALING)

    config = load_hf_checkpoint_config(tmp_path)

    assert config.rope_theta == 500000
    assert config.rope_scaling == LLAMA3_SCALING


def test_load_config_translates_transformers5_rope(tmp_path: Path) -> None:
    _write_config(tmp_path, rope_parameters=LLAMA3_PARAMETERS)

    config = load_hf_checkpoint_config(tmp_path)

    assert config.rope_theta == 500000
    assert config.rope_scaling == LLAMA3_SCALING


def test_load_config_accepts_consistent_dual_schema(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        rope_parameters=LLAMA3_PARAMETERS,
        rope_theta=500000.0,
        rope_scaling=LLAMA3_SCALING,
    )

    config = load_hf_checkpoint_config(tmp_path)

    assert config.rope_theta == 500000
    assert config.rope_scaling == LLAMA3_SCALING


def test_load_config_accepts_semantically_equivalent_default_dual_schema(
    tmp_path: Path,
) -> None:
    _write_config(
        tmp_path,
        rope_parameters={"rope_theta": 500000, "rope_type": "default"},
        rope_theta=500000,
        rope_scaling={"rope_type": "default"},
    )

    config = load_hf_checkpoint_config(tmp_path)

    assert config.rope_theta == 500000
    assert config.rope_scaling is None


@pytest.mark.parametrize(
    "legacy_fields",
    [
        {"rope_theta": 10000, "rope_scaling": LLAMA3_SCALING},
        {
            "rope_theta": 500000,
            "rope_scaling": {**LLAMA3_SCALING, "high_freq_factor": 2.0},
        },
    ],
)
def test_load_config_rejects_conflicting_dual_schema(
    tmp_path: Path, legacy_fields: dict[str, object]
) -> None:
    _write_config(tmp_path, rope_parameters=LLAMA3_PARAMETERS, **legacy_fields)

    with pytest.raises(HfCheckpointCompatibilityError, match="conflicting"):
        load_hf_checkpoint_config(tmp_path)


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"rope_type": "llama3", **LLAMA3_SCALING}, "missing rope_theta"),
        ({"rope_theta": 500000, "rope_type": "llama3"}, "is missing"),
        (["not", "an", "object"], "must be a JSON object"),
        (
            {
                "full_attention": {"rope_type": "default", "rope_theta": 10000},
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000},
            },
            "per-layer or incomplete",
        ),
    ],
)
def test_load_config_rejects_incomplete_or_malformed_transformers5_schema(
    tmp_path: Path, parameters: object, message: str
) -> None:
    _write_config(tmp_path, rope_parameters=parameters)

    with pytest.raises(HfCheckpointCompatibilityError, match=message):
        load_hf_checkpoint_config(tmp_path)


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"rope_theta": 500000, "rope_type": "future_rope"}, "cannot validate"),
        (
            {"rope_theta": 500000, **LLAMA3_SCALING, "producer_only_field": 1},
            "unsupported fields",
        ),
        (
            {"rope_theta": 500000, "rope_type": "linear", "factor": 0.5},
            "at least 1",
        ),
        (
            {
                "rope_theta": 500000,
                "rope_type": "yarn",
                "factor": 8.0,
                "truncate": "false",
            },
            "must be a boolean",
        ),
    ],
)
def test_load_config_rejects_unrepresentable_transformers5_schema(
    tmp_path: Path, parameters: object, message: str
) -> None:
    _write_config(tmp_path, rope_parameters=parameters)

    with pytest.raises(HfCheckpointCompatibilityError, match=message):
        load_hf_checkpoint_config(tmp_path)


def test_load_config_rejects_transformers5_longrope_as_unrepresentable(
    tmp_path: Path,
) -> None:
    _write_config(
        tmp_path,
        hidden_size=8,
        num_attention_heads=2,
        head_dim=4,
        rope_parameters={
            "rope_theta": 500000,
            "rope_type": "longrope",
            "short_factor": [1.0, 1.0],
            "long_factor": [2.0, 2.0],
            "factor": 2.0,
        },
    )

    with pytest.raises(HfCheckpointCompatibilityError, match="cannot be translated"):
        load_hf_checkpoint_config(tmp_path)


def test_load_config_rejects_transformers4_longrope_factor_length_mismatch(
    tmp_path: Path,
) -> None:
    _write_config(
        tmp_path,
        hidden_size=8,
        num_attention_heads=2,
        head_dim=4,
        max_position_embeddings=8,
        original_max_position_embeddings=4,
        rope_theta=500000,
        rope_scaling={
            "rope_type": "longrope",
            "short_factor": [1.0],
            "long_factor": [2.0],
            "factor": 2.0,
        },
    )

    with pytest.raises(HfCheckpointCompatibilityError, match="must contain 2 values"):
        load_hf_checkpoint_config(tmp_path)


def test_load_config_translates_unscaled_transformers5_rope(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        rope_parameters={"rope_type": "default", "rope_theta": 500000},
    )

    config = load_hf_checkpoint_config(tmp_path)

    assert config.rope_theta == 500000
    assert config.rope_scaling is None


def test_load_config_requires_local_config_json(tmp_path: Path) -> None:
    with pytest.raises(HfCheckpointCompatibilityError, match="config.json"):
        load_hf_checkpoint_config(tmp_path)


def test_load_tokenizer_uses_auto_for_native_export(tmp_path: Path) -> None:
    tokenizer = object()
    with (
        patch(
            "marin_dna_evals.hf_compat.AutoTokenizer.from_pretrained",
            return_value=tokenizer,
        ) as auto_load,
        patch(
            "marin_dna_evals.hf_compat.PreTrainedTokenizerFast.from_pretrained"
        ) as fast_load,
    ):
        assert load_hf_checkpoint_tokenizer(tmp_path) is tokenizer
    auto_load.assert_called_once_with(tmp_path)
    fast_load.assert_not_called()


def test_load_tokenizer_handles_transformers5_backend(tmp_path: Path) -> None:
    (tmp_path / "tokenizer_config.json").write_text(
        '{"tokenizer_class": "TokenizersBackend"}', encoding="utf-8"
    )
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    tokenizer = object()
    with (
        patch("marin_dna_evals.hf_compat.AutoTokenizer.from_pretrained") as auto_load,
        patch(
            "marin_dna_evals.hf_compat.PreTrainedTokenizerFast.from_pretrained",
            return_value=tokenizer,
        ) as fast_load,
    ):
        assert load_hf_checkpoint_tokenizer(tmp_path) is tokenizer
    auto_load.assert_not_called()
    fast_load.assert_called_once_with(tmp_path)


def test_load_tokenizer_requires_json_for_transformers5_backend(tmp_path: Path) -> None:
    (tmp_path / "tokenizer_config.json").write_text(
        '{"tokenizer_class": "TokenizersBackend"}', encoding="utf-8"
    )
    with pytest.raises(HfCheckpointCompatibilityError, match="missing tokenizer.json"):
        load_hf_checkpoint_tokenizer(tmp_path)


@pytest.mark.parametrize("class_name", ["PreTrainedTokenizerFast", "TokenizersBackend"])
@pytest.mark.parametrize("serialized_added_token", [False, True])
def test_load_tokenizer_translates_transformers5_special_tokens(
    tmp_path: Path, class_name: str, serialized_added_token: bool
) -> None:
    from tokenizers import (
        Regex,
        Tokenizer,
        models,
        normalizers,
        pre_tokenizers,
        processors,
    )
    from transformers import PreTrainedTokenizerFast

    vocab = {
        token: index
        for index, token in enumerate(
            ["[PAD]", "[UNK]", "[BOS]", "[SEQ]", "a", "c", "g", "t"]
        )
    }
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.normalizer = normalizers.Lowercase()
    backend.pre_tokenizer = pre_tokenizers.Split(Regex(".{1}"), behavior="isolated")
    backend.post_processor = processors.TemplateProcessing(
        single="[BOS] $A", special_tokens=[("[BOS]", 2)]
    )
    original = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="[BOS]",
        pad_token="[PAD]",
        unk_token="[UNK]",
        additional_special_tokens=["[SEQ]"],
        model_max_length=10240,
    )
    original.save_pretrained(tmp_path)
    config_path = tmp_path / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config["tokenizer_class"] = class_name
    config.pop("additional_special_tokens")
    config["extra_special_tokens"] = (
        [
            {
                "content": "[SEQ]",
                "special": True,
                "normalized": False,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "__type": "AddedToken",
            }
        ]
        if serialized_added_token
        else ["[SEQ]"]
    )
    config_path.write_text(json.dumps(config))
    (tmp_path / "special_tokens_map.json").unlink()
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    loaded = load_hf_checkpoint_tokenizer(tmp_path)

    assert loaded.get_vocab() == original.get_vocab() == vocab
    assert (
        loaded.encode("AC[SEQ]GT") == original.encode("AC[SEQ]GT") == [2, 4, 5, 3, 6, 7]
    )
    assert loaded.all_special_ids == original.all_special_ids
    assert loaded.additional_special_tokens == ["[SEQ]"]
    assert loaded.decode([2, 4, 3, 5, 0], skip_special_tokens=True) == original.decode(
        [2, 4, 3, 5, 0], skip_special_tokens=True
    )
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_load_tokenizer_preserves_transformers4_named_tokens(tmp_path: Path) -> None:
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "extra_special_tokens": {"image_token": "[IMAGE]"},
            }
        )
    )
    with patch("marin_dna_evals.hf_compat.AutoTokenizer.from_pretrained") as loader:
        load_hf_checkpoint_tokenizer(tmp_path)
    loader.assert_called_once_with(tmp_path)


def test_load_tokenizer_rejects_conflicting_special_token_lists(tmp_path: Path) -> None:
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "extra_special_tokens": ["[SEQ]"],
                "additional_special_tokens": ["[OTHER]"],
            }
        )
    )
    with pytest.raises(
        HfCheckpointCompatibilityError, match="conflicting special-token lists"
    ):
        load_hf_checkpoint_tokenizer(tmp_path)


@pytest.mark.parametrize(
    ("function_name", "loader_name"),
    [
        ("load_hf_causal_lm_and_tokenizer", "AutoModelForCausalLM"),
        ("load_hf_base_model_and_tokenizer", "AutoModel"),
    ],
)
def test_model_loaders_pass_validated_config_explicitly(
    function_name: str, loader_name: str
) -> None:
    checkpoint_path = Path("/checkpoint")
    config = object()
    tokenizer = object()
    model = object()
    function = {
        "load_hf_causal_lm_and_tokenizer": load_hf_causal_lm_and_tokenizer,
        "load_hf_base_model_and_tokenizer": load_hf_base_model_and_tokenizer,
    }[function_name]
    with (
        patch(
            "marin_dna_evals.hf_compat.load_hf_checkpoint_config", return_value=config
        ),
        patch(
            "marin_dna_evals.hf_compat.load_hf_checkpoint_tokenizer",
            return_value=tokenizer,
        ),
        patch(
            f"marin_dna_evals.hf_compat.{loader_name}.from_pretrained",
            return_value=model,
        ) as load,
    ):
        assert function(checkpoint_path) == (tokenizer, model)
    load.assert_called_once_with(
        checkpoint_path,
        config=config,
        trust_remote_code=True,
    )


def test_invalid_config_stops_before_tokenizer_or_weight_loading(
    tmp_path: Path,
) -> None:
    _write_config(
        tmp_path,
        rope_parameters=LLAMA3_PARAMETERS,
        rope_theta=10000,
        rope_scaling=LLAMA3_SCALING,
    )
    with (
        patch(
            "marin_dna_evals.hf_compat.load_hf_checkpoint_tokenizer"
        ) as tokenizer_load,
        patch(
            "marin_dna_evals.hf_compat.AutoModelForCausalLM.from_pretrained"
        ) as model_load,
        pytest.raises(HfCheckpointCompatibilityError, match="conflicting"),
    ):
        load_hf_causal_lm_and_tokenizer(tmp_path)
    tokenizer_load.assert_not_called()
    model_load.assert_not_called()


def test_production_auto_model_loading_is_centralized() -> None:
    package_root = Path(__file__).parents[2] / "src" / "marin_dna_evals"
    offenders: list[str] = []
    for path in package_root.rglob("*.py"):
        if path.name == "hf_compat.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(
                node.func, ast.Attribute
            ):
                continue
            owner = node.func.value
            if (
                node.func.attr == "from_pretrained"
                and isinstance(owner, ast.Name)
                and owner.id in {"AutoModel", "AutoModelForCausalLM"}
            ):
                offenders.append(f"{path.relative_to(package_root)}:{node.lineno}")
    assert offenders == []
