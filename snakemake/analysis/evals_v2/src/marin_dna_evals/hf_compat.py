"""Fail-closed Hugging Face checkpoint loading across Transformers 4 and 5."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)


class HfCheckpointCompatibilityError(ValueError):
    """The raw HF checkpoint metadata cannot be loaded without semantic drift."""


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise HfCheckpointCompatibilityError(
            f"required checkpoint metadata is missing: {path}"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HfCheckpointCompatibilityError(
            f"cannot read checkpoint metadata {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise HfCheckpointCompatibilityError(f"{path} must contain a JSON object")
    return value


def _positive_number(value: Any, *, field: str, path: Path) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HfCheckpointCompatibilityError(
            f"{path}: {field} must be a positive number, got {value!r}"
        )
    if not math.isfinite(float(value)) or value <= 0:
        raise HfCheckpointCompatibilityError(
            f"{path}: {field} must be a positive number, got {value!r}"
        )
    return value


def _canonical_rope_scaling(
    value: Any, *, field: str, path: Path
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise HfCheckpointCompatibilityError(
            f"{path}: {field} must be a JSON object or null"
        )

    scaling = dict(value)
    old_type = scaling.pop("type", None)
    rope_type = scaling.get("rope_type")
    if rope_type is None:
        rope_type = old_type
        if rope_type is not None:
            scaling["rope_type"] = rope_type
    elif old_type is not None and old_type != rope_type:
        raise HfCheckpointCompatibilityError(
            f"{path}: {field} has conflicting type={old_type!r} and rope_type={rope_type!r}"
        )
    if not isinstance(rope_type, str) or not rope_type:
        raise HfCheckpointCompatibilityError(
            f"{path}: {field} is missing a string rope_type"
        )

    required_by_type = {
        "default": set(),
        "linear": {"factor"},
        "dynamic": {"factor"},
        "yarn": {"factor"},
        "llama3": {
            "factor",
            "low_freq_factor",
            "high_freq_factor",
            "original_max_position_embeddings",
        },
        "longrope": {"short_factor", "long_factor"},
    }
    optional_by_type = {
        "default": set(),
        "linear": set(),
        "dynamic": {"original_max_position_embeddings"},
        "yarn": {
            "attention_factor",
            "beta_fast",
            "beta_slow",
            "original_max_position_embeddings",
            "mscale",
            "mscale_all_dim",
            "truncate",
        },
        "llama3": set(),
        "longrope": {
            "attention_factor",
            "factor",
            "original_max_position_embeddings",
        },
    }
    if rope_type not in required_by_type:
        raise HfCheckpointCompatibilityError(
            f"{path}: {field} uses rope_type={rope_type!r}, which Transformers 4 cannot validate"
        )
    missing = required_by_type[rope_type] - scaling.keys()
    if missing:
        raise HfCheckpointCompatibilityError(
            f"{path}: {field} for rope_type={rope_type!r} is missing {sorted(missing)}"
        )
    unexpected = (
        scaling.keys()
        - required_by_type[rope_type]
        - optional_by_type[rope_type]
        - {"rope_type"}
    )
    if unexpected:
        raise HfCheckpointCompatibilityError(
            f"{path}: {field} for rope_type={rope_type!r} has unsupported fields {sorted(unexpected)}"
        )

    for numeric_field in (
        "factor",
        "low_freq_factor",
        "high_freq_factor",
        "attention_factor",
        "beta_fast",
        "beta_slow",
        "mscale",
        "mscale_all_dim",
    ):
        if numeric_field in scaling:
            _positive_number(
                scaling[numeric_field], field=f"{field}.{numeric_field}", path=path
            )
    if "factor" in scaling and scaling["factor"] < 1:
        raise HfCheckpointCompatibilityError(
            f"{path}: {field}.factor must be at least 1"
        )
    if "truncate" in scaling and not isinstance(scaling["truncate"], bool):
        raise HfCheckpointCompatibilityError(
            f"{path}: {field}.truncate must be a boolean"
        )
    if "original_max_position_embeddings" in scaling:
        original_max = scaling["original_max_position_embeddings"]
        if (
            isinstance(original_max, bool)
            or not isinstance(original_max, int)
            or original_max <= 0
        ):
            raise HfCheckpointCompatibilityError(
                f"{path}: {field}.original_max_position_embeddings must be a positive integer"
            )
    if (
        rope_type == "llama3"
        and scaling["high_freq_factor"] <= scaling["low_freq_factor"]
    ):
        raise HfCheckpointCompatibilityError(
            f"{path}: {field}.high_freq_factor must exceed low_freq_factor"
        )
    if rope_type == "yarn" and scaling.get("beta_fast", 32) < scaling.get(
        "beta_slow", 1
    ):
        raise HfCheckpointCompatibilityError(
            f"{path}: {field}.beta_fast must be at least beta_slow"
        )
    for factors_field in ("short_factor", "long_factor"):
        if factors_field not in scaling:
            continue
        factors = scaling[factors_field]
        if not isinstance(factors, list) or not factors:
            raise HfCheckpointCompatibilityError(
                f"{path}: {field}.{factors_field} must be a non-empty list of positive numbers"
            )
        for index, factor in enumerate(factors):
            _positive_number(
                factor,
                field=f"{field}.{factors_field}[{index}]",
                path=path,
            )
    if rope_type == "default":
        return None
    return scaling


def _legacy_rope_from_parameters(
    value: Any, *, path: Path
) -> tuple[int | float, dict[str, Any] | None]:
    if not isinstance(value, dict):
        raise HfCheckpointCompatibilityError(
            f"{path}: rope_parameters must be a JSON object"
        )
    parameters = dict(value)
    if "rope_theta" not in parameters:
        raise HfCheckpointCompatibilityError(
            f"{path}: rope_parameters is missing rope_theta; per-layer or incomplete RoPE cannot be translated"
        )
    theta = _positive_number(
        parameters.pop("rope_theta"), field="rope_parameters.rope_theta", path=path
    )
    scaling = _canonical_rope_scaling(parameters, field="rope_parameters", path=path)
    if scaling is not None and scaling["rope_type"] == "longrope":
        raise HfCheckpointCompatibilityError(
            f"{path}: Transformers-5 longrope cannot be translated to Transformers 4 "
            "without changing its original-context semantics"
        )
    return theta, scaling


def _validate_longrope_dimensions(
    config: PretrainedConfig,
    scaling: dict[str, Any],
    *,
    path: Path,
) -> None:
    max_position_embeddings = getattr(config, "max_position_embeddings", None)
    if (
        isinstance(max_position_embeddings, bool)
        or not isinstance(max_position_embeddings, int)
        or max_position_embeddings <= 0
    ):
        raise HfCheckpointCompatibilityError(
            f"{path}: max_position_embeddings must be a positive integer for longrope"
        )
    original_max = getattr(config, "original_max_position_embeddings", None)
    nested_original_max = scaling.get("original_max_position_embeddings")
    if original_max is not None:
        if (
            isinstance(original_max, bool)
            or not isinstance(original_max, int)
            or original_max <= 0
        ):
            raise HfCheckpointCompatibilityError(
                f"{path}: original_max_position_embeddings must be a positive integer"
            )
        if nested_original_max is not None and nested_original_max != original_max:
            raise HfCheckpointCompatibilityError(
                f"{path}: top-level and nested original_max_position_embeddings conflict"
            )
        if "factor" in scaling and not math.isclose(
            float(scaling["factor"]),
            max_position_embeddings / original_max,
        ):
            raise HfCheckpointCompatibilityError(
                f"{path}: longrope factor conflicts with the effective context-length ratio"
            )
    elif nested_original_max is not None:
        raise HfCheckpointCompatibilityError(
            f"{path}: Transformers 4 ignores nested longrope original_max_position_embeddings"
        )
    elif "factor" not in scaling and "attention_factor" not in scaling:
        raise HfCheckpointCompatibilityError(
            f"{path}: longrope requires factor when no original context length or "
            "attention_factor is available"
        )

    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        hidden_size = getattr(config, "hidden_size", None)
        num_attention_heads = getattr(config, "num_attention_heads", None)
        if (
            isinstance(hidden_size, bool)
            or not isinstance(hidden_size, int)
            or hidden_size <= 0
            or isinstance(num_attention_heads, bool)
            or not isinstance(num_attention_heads, int)
            or num_attention_heads <= 0
        ):
            raise HfCheckpointCompatibilityError(
                f"{path}: cannot derive the rotary dimension for longrope"
            )
        head_dim = hidden_size // num_attention_heads
    if isinstance(head_dim, bool) or not isinstance(head_dim, int) or head_dim <= 0:
        raise HfCheckpointCompatibilityError(
            f"{path}: head_dim must be a positive integer for longrope"
        )
    partial_rotary_factor = _positive_number(
        getattr(config, "partial_rotary_factor", 1.0),
        field="partial_rotary_factor",
        path=path,
    )
    expected_length = int(head_dim * partial_rotary_factor) // 2
    if expected_length <= 0:
        raise HfCheckpointCompatibilityError(
            f"{path}: longrope rotary dimension must be positive"
        )
    for field in ("short_factor", "long_factor"):
        actual_length = len(scaling[field])
        if actual_length != expected_length:
            raise HfCheckpointCompatibilityError(
                f"{path}: rope_scaling.{field} must contain {expected_length} values, "
                f"got {actual_length}"
            )


def _validate_effective_rope(
    config: PretrainedConfig,
    *,
    expected_theta: float | None,
    expected_scaling: dict[str, Any] | None,
    check_scaling: bool,
    path: Path,
) -> None:
    if (
        expected_theta is not None
        and getattr(config, "rope_theta", None) != expected_theta
    ):
        raise HfCheckpointCompatibilityError(
            f"{path}: Transformers resolved rope_theta={getattr(config, 'rope_theta', None)!r}, "
            f"expected {expected_theta!r}"
        )
    if check_scaling:
        effective_scaling = _canonical_rope_scaling(
            getattr(config, "rope_scaling", None),
            field="effective rope_scaling",
            path=path,
        )
        if effective_scaling != expected_scaling:
            raise HfCheckpointCompatibilityError(
                f"{path}: Transformers resolved rope_scaling={effective_scaling!r}, "
                f"expected {expected_scaling!r}"
            )
        if (
            effective_scaling is not None
            and effective_scaling["rope_type"] == "longrope"
        ):
            _validate_longrope_dimensions(config, effective_scaling, path=path)


def load_hf_checkpoint_config(checkpoint_path: str | Path) -> PretrainedConfig:
    """Load and validate a local HF config without Transformers-major drift.

    Historical Transformers 5 exports are translated in memory. Consistent
    dual-schema exports are accepted. Conflicting or incomplete schemas raise
    before a tokenizer or model weight loader is called.
    """
    checkpoint_path = Path(checkpoint_path)
    config_path = checkpoint_path / "config.json"
    raw_config = _read_json_object(config_path)

    has_parameters = "rope_parameters" in raw_config
    has_legacy_theta = "rope_theta" in raw_config
    has_legacy_scaling = "rope_scaling" in raw_config

    expected_theta: int | float | None = None
    expected_scaling: dict[str, Any] | None = None
    check_scaling = False

    if has_parameters:
        expected_theta, expected_scaling = _legacy_rope_from_parameters(
            raw_config["rope_parameters"], path=config_path
        )
        has_any_legacy = has_legacy_theta or has_legacy_scaling
        if has_any_legacy:
            if not has_legacy_theta:
                raise HfCheckpointCompatibilityError(
                    f"{config_path}: dual-schema RoPE is missing top-level rope_theta"
                )
            legacy_theta = _positive_number(
                raw_config["rope_theta"], field="rope_theta", path=config_path
            )
            legacy_scaling = _canonical_rope_scaling(
                raw_config.get("rope_scaling"), field="rope_scaling", path=config_path
            )
            if legacy_theta != expected_theta or legacy_scaling != expected_scaling:
                raise HfCheckpointCompatibilityError(
                    f"{config_path}: conflicting Transformers 4 and 5 RoPE schemas"
                )
        check_scaling = True
        config = AutoConfig.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
            rope_theta=expected_theta,
            rope_scaling=expected_scaling,
        )
    else:
        if has_legacy_theta:
            expected_theta = _positive_number(
                raw_config["rope_theta"], field="rope_theta", path=config_path
            )
        if has_legacy_scaling:
            expected_scaling = _canonical_rope_scaling(
                raw_config["rope_scaling"], field="rope_scaling", path=config_path
            )
            check_scaling = True
        config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)

    _validate_effective_rope(
        config,
        expected_theta=expected_theta,
        expected_scaling=expected_scaling,
        check_scaling=check_scaling,
        path=config_path,
    )
    return config


def load_hf_checkpoint_tokenizer(
    checkpoint_path: str | Path,
) -> PreTrainedTokenizerBase:
    """Load native Transformers 4 and Transformers 5 tokenizer exports."""
    checkpoint_path = Path(checkpoint_path)
    tokenizer_config_path = checkpoint_path / "tokenizer_config.json"
    if tokenizer_config_path.is_file():
        tokenizer_config = _read_json_object(tokenizer_config_path)
        if tokenizer_config.get("tokenizer_class") == "TokenizersBackend":
            tokenizer_json_path = checkpoint_path / "tokenizer.json"
            if not tokenizer_json_path.is_file():
                raise HfCheckpointCompatibilityError(
                    f"TokenizersBackend export is missing tokenizer.json: {tokenizer_json_path}"
                )
            return PreTrainedTokenizerFast.from_pretrained(checkpoint_path)
    return AutoTokenizer.from_pretrained(checkpoint_path)


def _load_hf_model_and_tokenizer(
    checkpoint_path: str | Path, model_loader: Any
) -> tuple[PreTrainedTokenizerBase, Any]:
    checkpoint_path = Path(checkpoint_path)
    config = load_hf_checkpoint_config(checkpoint_path)
    tokenizer = load_hf_checkpoint_tokenizer(checkpoint_path)
    model = model_loader.from_pretrained(
        checkpoint_path,
        config=config,
        trust_remote_code=True,
    )
    return tokenizer, model


def load_hf_causal_lm_and_tokenizer(
    checkpoint_path: str | Path,
) -> tuple[PreTrainedTokenizerBase, Any]:
    """Load a validated causal LM and its compatible tokenizer."""
    return _load_hf_model_and_tokenizer(checkpoint_path, AutoModelForCausalLM)


def load_hf_base_model_and_tokenizer(
    checkpoint_path: str | Path,
) -> tuple[PreTrainedTokenizerBase, Any]:
    """Load a validated base model and its compatible tokenizer."""
    return _load_hf_model_and_tokenizer(checkpoint_path, AutoModel)
