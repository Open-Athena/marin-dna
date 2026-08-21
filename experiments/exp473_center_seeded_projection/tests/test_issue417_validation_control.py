from __future__ import annotations

import json

import pytest
from exp473_center_seeded_projection.issue417_validation_control import (
    translate_legacy_rope_config,
)
from exp473_center_seeded_projection.issue417_validation_control_config import (
    STEPS,
    build_issue417_validation_control_config,
)
from exp473_center_seeded_projection.native_validation_replay import (
    EXPECTED_ROPE_SCALING,
    EXPECTED_ROPE_THETA,
)


def test_issue417_control_is_complete_and_fail_closed() -> None:
    commit = "a" * 40
    config = build_issue417_validation_control_config(diagnostic_commit=commit)
    assert config["purpose"] == "damage_control_issue417_validation_control"
    assert config["interpretation_allowed"] is False
    assert config["vep_held_out_access"] is False
    assert len(config["models"]) == 19
    assert {model["arm"] for model in config["models"]} == set(STEPS)
    assert {
        model["step"]
        for model in config["models"]
        if model["arm"] == "issue417_mammals_only"
    } == set(STEPS["issue417_mammals_only"])
    with pytest.raises(ValueError, match="full lowercase commit SHA"):
        build_issue417_validation_control_config(diagnostic_commit="dev")


def test_legacy_rope_translation_is_ephemeral_and_exact(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config_path = checkpoint / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "transformers_version": "5.12.1",
                "rope_parameters": {
                    **EXPECTED_ROPE_SCALING,
                    "rope_theta": EXPECTED_ROPE_THETA,
                },
            }
        ),
        encoding="utf-8",
    )
    receipt = translate_legacy_rope_config(checkpoint)
    translated = json.loads(config_path.read_text(encoding="utf-8"))
    assert receipt["mode"] == "ephemeral_legacy_to_dual_schema"
    assert translated["rope_theta"] == EXPECTED_ROPE_THETA
    assert translated["rope_scaling"] == EXPECTED_ROPE_SCALING
    assert receipt["original_config_sha256"] != receipt["translated_config_sha256"]

    second = translate_legacy_rope_config(checkpoint)
    assert second["mode"] == "verified_existing_dual_schema"
    assert second["original_config_sha256"] == second["translated_config_sha256"]


def test_legacy_rope_translation_rejects_wrong_theta(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "rope_parameters": {
                    **EXPECTED_ROPE_SCALING,
                    "rope_theta": 10_000,
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unexpected trained RoPE semantics"):
        translate_legacy_rope_config(checkpoint)
