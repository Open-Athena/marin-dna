from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
import torch
from torch import nn
from transformers import PreTrainedModel

from exp479_mntp.causal_longrun import (
    LONGRUN_CHECKPOINT_STEPS,
    LONGRUN_EVALUATION_ARTIFACT,
    LONGRUN_LEARNING_RATE,
    LONGRUN_MODEL_ARTIFACT_PREFIX,
    LONGRUN_RUN_NAME,
    AdamWLongRunConfig,
    CausalLongRunModule,
    _write_retention_manifest,
    plot_longrun_stability,
    plot_macro_validation,
    projected_longrun_cost,
    summarize_macro_trajectory,
)
from exp479_mntp.config import wsd_multiplier


def test_longrun_is_the_selected_single_configuration() -> None:
    config = AdamWLongRunConfig()
    assert LONGRUN_LEARNING_RATE == 1e-5
    assert config.learning_rate == 1e-5
    assert config.train_steps == 1_000
    assert config.warmup_steps == 100
    assert config.cooldown_start_step == 800
    assert config.weight_decay == 0
    assert "corrected" in LONGRUN_RUN_NAME
    assert "corrected" in LONGRUN_MODEL_ARTIFACT_PREFIX
    assert "corrected" in LONGRUN_EVALUATION_ARTIFACT
    assert LONGRUN_CHECKPOINT_STEPS == (
        0,
        25,
        50,
        100,
        200,
        300,
        400,
        500,
        600,
        700,
        800,
        900,
        1_000,
    )


def test_longrun_schedule_is_ten_percent_warmup_and_twenty_percent_decay() -> None:
    assert wsd_multiplier(0) == pytest.approx(0.0)
    assert wsd_multiplier(1) == pytest.approx(0.01)
    assert wsd_multiplier(50) == pytest.approx(0.5)
    assert wsd_multiplier(100) == pytest.approx(1.0)
    assert wsd_multiplier(800) == pytest.approx(1.0)
    assert wsd_multiplier(900) == pytest.approx(0.5)
    assert wsd_multiplier(999) == pytest.approx(0.005)
    assert wsd_multiplier(1_000) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="expected"):
        AdamWLongRunConfig(warmup_steps=0)


def test_longrun_module_builds_one_plain_adamw_group() -> None:
    model = cast(PreTrainedModel, cast(Any, nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))))
    module = CausalLongRunModule(model=model, batch_size=2)
    configuration = module.configure_optimizers()
    optimizer = configuration["optimizer"]
    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["betas"] == (0.9, 0.95)
    assert optimizer.param_groups[0]["eps"] == 1e-8
    assert optimizer.param_groups[0]["weight_decay"] == 0
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)


def _macro_losses(*, final_delta: float = -0.001) -> pd.DataFrame:
    rows = []
    for step in LONGRUN_CHECKPOINT_STEPS:
        fraction = step / 1_000
        rows.append(
            {
                "step": step,
                "component": "macro",
                "loss": 0.23 + final_delta * fraction,
                "accuracy": 0.5,
                "n_rows": 640,
            }
        )
    return pd.DataFrame(rows)


def test_macro_gate_uses_only_final_macro_loss() -> None:
    passing = summarize_macro_trajectory(_macro_losses())
    assert passing["passed"]
    assert passing["delta"] == pytest.approx(-0.001)

    failing = summarize_macro_trajectory(_macro_losses(final_delta=0.001))
    assert not failing["passed"]
    assert failing["delta"] == pytest.approx(0.001)


def test_longrun_plots_write_reviewable_svg_and_png(tmp_path: Path) -> None:
    validation_output = tmp_path / "validation-trajectory"
    plot_macro_validation(_macro_losses(), validation_output)
    assert validation_output.with_suffix(".svg").is_file()
    assert validation_output.with_suffix(".png").is_file()
    validation_svg = validation_output.with_suffix(".svg").read_text(encoding="utf-8")
    assert "Corrected fixed-plan validation macro" in validation_svg
    assert "AdamW 1e-5 causal fine-tuning with corrected loss" in validation_svg

    trace = pd.DataFrame(
        {
            "step": [0, 1, 2],
            "train_loss": [0.3, 0.29, 0.28],
            "learning_rate": [0.0, 1e-7, 2e-7],
            "pre_clip_gradient_norm": [0.9, 0.8, 0.7],
        }
    )
    stability_output = tmp_path / "training-stability"
    plot_longrun_stability(trace, stability_output)
    assert stability_output.with_suffix(".svg").is_file()
    assert stability_output.with_suffix(".png").is_file()
    stability_svg = stability_output.with_suffix(".svg").read_text(encoding="utf-8")
    assert "Warmup-stable-decay schedule" in stability_svg
    assert "Gradient stability" in stability_svg


def test_retention_manifest_explicitly_records_no_deletion(tmp_path: Path) -> None:
    path = tmp_path / "retention-manifest.json"
    records: list[dict[str, int | str | None]] = [
        {
            "kind": "lightning_checkpoint",
            "step": 1_000,
            "artifact_id": "artifact-id",
            "artifact_name": "checkpoint:v0",
            "artifact_version": "v0",
            "qualified_name": "entity/project/checkpoint:v0",
        }
    ]
    _write_retention_manifest(path, records)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["backend"] == "wandb"
    assert payload["deletion_performed"] is False
    assert payload["artifacts"] == records


def test_longrun_prelaunch_projection_reserves_two_hours() -> None:
    assert projected_longrun_cost(25.26241970350875) == pytest.approx(29.84241970350875)
