"""Registry contract for the fixed-checkpoint context ablation in issue #485."""

from __future__ import annotations

from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).parents[2] / "config" / "config.yaml"
BASE_MODEL = "mix-v0.9-p1B-i24-exp135-m5.1-step-59158"
CHECKPOINT = (
    "gs://marin-us-east5/checkpoints/"
    "dna-bolinas-mix-v0.9-p1B-i24-exp135-zoonomia-m5.1-bef41e/hf/step-59158"
)
PROBED_CONTEXT_MODELS = {
    BASE_MODEL: 255,
    f"{BASE_MODEL}-ctx1023": 1023,
    f"{BASE_MODEL}-ctx511": 511,
    f"{BASE_MODEL}-ctx127": 127,
    f"{BASE_MODEL}-ctx63": 63,
    f"{BASE_MODEL}-ctx31": 31,
}
CONTEXT_MODELS = PROBED_CONTEXT_MODELS


def test_issue_485_context_models_share_checkpoint_and_cover_expected_windows():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    models = {model["name"]: model for model in config["models"]}

    assert set(CONTEXT_MODELS) <= set(models)
    for name, window_size in CONTEXT_MODELS.items():
        model = models[name]
        assert model["gcs_path"] == CHECKPOINT
        assert model["window_size"] == window_size
        assert window_size % 2 == 1
        assert "mendelian_traits" in model["datasets"]

    for name in set(CONTEXT_MODELS) - {BASE_MODEL}:
        assert models[name]["datasets"] == ["mendelian_traits"]

    assert models[f"{BASE_MODEL}-ctx511"]["batch_size"] == 24
    assert models[f"{BASE_MODEL}-ctx1023"]["batch_size"] == 6


def test_issue_485_context_models_are_all_registered_for_mendelian_probe():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    probe_models = {
        model["name"]: model["datasets"] for model in config["probe"]["models"]
    }

    for name in PROBED_CONTEXT_MODELS:
        assert probe_models[name] == ["mendelian_traits"]
