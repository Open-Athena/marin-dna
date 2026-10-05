"""Contracts for global inference semantics and checkpoint execution sizing."""

import hashlib
import json
import re
from pathlib import Path

import pytest
import yaml
from marin_dna_evals.workflow_config import (
    GLOBAL_INFERENCE_SWITCHES,
    inference_precision_params,
    resolve_model_batch_size,
    resolve_model_eval_accumulation_steps,
    validate_inference_config,
)

PROJECT_ROOT = Path(__file__).parents[2]


def _config() -> dict[str, object]:
    with (PROJECT_ROOT / "config" / "config.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_checked_in_config_uses_global_on_semantic_switches() -> None:
    config = _config()
    inference = config["inference"]
    models = config["models"]

    validate_inference_config(inference, models)
    assert {field: inference[field] for field in GLOBAL_INFERENCE_SWITCHES} == {
        "return_embeddings": True,
        "torch_compile": True,
        "bf16": True,
    }
    assert all(GLOBAL_INFERENCE_SWITCHES.isdisjoint(model) for model in models)


@pytest.mark.parametrize("field", sorted(GLOBAL_INFERENCE_SWITCHES))
def test_checkpoint_semantic_override_is_rejected(field: str) -> None:
    inference = {
        "return_embeddings": True,
        "torch_compile": True,
        "bf16": True,
        "rc": True,
        "batch_size": 128,
        "eval_accumulation_steps": None,
    }
    with pytest.raises(ValueError, match="cannot override"):
        validate_inference_config(inference, [{"name": "model", field: True}])


def test_checkpoint_execution_matrix_and_fallbacks() -> None:
    config = _config()
    inference = config["inference"]
    models = {model["name"]: model for model in config["models"]}

    cases = {
        "exp232-v4_cds-step-500": (128, None),
        "exp166-v0.1-p1B-step-27329": (64, 8),
        "scaling-v0.5-h2432-p2B-step-215573": (16, 8),
        "exp166-v0.1-p4B-step-27329": (16, 8),
        "exp21-promoters-yolo-step-22000": (64, 8),
    }
    for model_name, expected in cases.items():
        model = models[model_name]
        observed = (
            resolve_model_batch_size(model, inference),
            resolve_model_eval_accumulation_steps(model, inference),
        )
        assert observed == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("batch_size", 0),
        ("batch_size", True),
        ("eval_accumulation_steps", 0),
        ("eval_accumulation_steps", False),
    ],
)
def test_invalid_checkpoint_execution_value_is_rejected(
    field: str, value: object
) -> None:
    inference = {
        "return_embeddings": True,
        "torch_compile": True,
        "bf16": True,
        "rc": True,
        "batch_size": 128,
        "eval_accumulation_steps": None,
    }
    with pytest.raises(ValueError, match=field):
        validate_inference_config(
            inference,
            [{"name": "model", field: value}],
        )


def test_documented_fp32_fallback_is_accepted() -> None:
    inference = dict(_config()["inference"])
    inference.update(
        bf16=False, tf32=False, precision_reason="Synthetic bf16 parity failed."
    )
    validate_inference_config(inference, [])


@pytest.mark.parametrize(
    "fields",
    [
        {"bf16": False},
        {"bf16": False, "tf32": False},
        {"bf16": False, "tf32": False, "precision_reason": " "},
        {"bf16": False, "tf32": True, "precision_reason": "Parity failure"},
        {"bf16": "false"},
        {"bf16": True, "tf32": "false"},
    ],
)
def test_incomplete_or_malformed_precision_fallback_is_rejected(fields) -> None:
    inference = dict(_config()["inference"])
    inference.update(fields)
    with pytest.raises(ValueError, match="bf16|tf32|fp32"):
        validate_inference_config(inference, [])


@pytest.mark.parametrize("field", ["tf32", "precision_reason"])
def test_precision_policy_cannot_be_overridden_by_checkpoint(field) -> None:
    with pytest.raises(ValueError, match="cannot override"):
        validate_inference_config(
            _config()["inference"], [{"name": "model", field: False}]
        )


def test_default_precision_preserves_existing_snakemake_parameter_identity() -> None:
    assert inference_precision_params(_config()["inference"]) == {}


def test_explicit_precision_override_enters_snakemake_provenance() -> None:
    settings = dict(_config()["inference"])
    settings.update(bf16=False, tf32=False, precision_reason="Recorded parity failure")
    assert inference_precision_params(settings) == {
        "precision": {
            "bf16": False,
            "tf32": False,
            "precision_reason": "Recorded parity failure",
        }
    }


def test_existing_scoring_run_source_is_unchanged() -> None:
    # Snakemake hashes run_func_src verbatim: even default-equivalent edits
    # invalidate existing artifacts via the independent code rerun trigger.
    fixture = json.loads(
        (PROJECT_ROOT / "tests/fixtures/legacy_scoring_run_hashes.json").read_text()
    )
    for filename, expected in fixture["run_blocks"].items():
        text = (PROJECT_ROOT / "workflow/rules" / filename).read_text()
        blocks = re.findall(
            r"^    run:\n(.*?)(?=^rule |\Z)", text, re.MULTILINE | re.DOTALL
        )
        assert [
            hashlib.sha256(block.encode()).hexdigest() for block in blocks
        ] == expected
