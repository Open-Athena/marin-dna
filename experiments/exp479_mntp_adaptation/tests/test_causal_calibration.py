from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
import torch
from torch import nn
from transformers import PreTrainedModel

from exp479_mntp.causal_calibration import (
    CALIBRATION_CHECKPOINT_STEPS,
    CALIBRATION_LEARNING_RATE,
    AdamWCalibrationConfig,
    CausalCalibrationModule,
    plot_training_stability,
    plot_validation_trajectories,
    projected_total_cost,
    summarize_validation_gate,
    warmup_constant_multiplier,
)
from exp479_mntp.config import DATA_COMPONENTS


def test_calibration_is_one_conservative_learning_rate() -> None:
    config = AdamWCalibrationConfig()
    assert CALIBRATION_LEARNING_RATE == 1e-6
    assert config.learning_rate == 1e-6
    assert config.train_steps == 200
    assert config.warmup_steps == 10
    assert config.weight_decay == 0
    assert CALIBRATION_CHECKPOINT_STEPS == (0, 1, 10, 25, 50, 100, 200)


def test_warmup_constant_schedule_reaches_and_holds_peak() -> None:
    assert warmup_constant_multiplier(0, warmup_steps=10) == pytest.approx(0.1)
    assert warmup_constant_multiplier(8, warmup_steps=10) == pytest.approx(0.9)
    assert warmup_constant_multiplier(9, warmup_steps=10) == pytest.approx(1.0)
    assert warmup_constant_multiplier(199, warmup_steps=10) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="non-negative"):
        warmup_constant_multiplier(-1, warmup_steps=10)


def test_calibration_module_builds_one_plain_adamw_group() -> None:
    model = cast(PreTrainedModel, cast(Any, nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))))
    module = CausalCalibrationModule(model=model, batch_size=2)
    configuration = module.configure_optimizers()
    optimizer = configuration["optimizer"]
    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["betas"] == (0.9, 0.95)
    assert optimizer.param_groups[0]["eps"] == 1e-8
    assert optimizer.param_groups[0]["weight_decay"] == 0
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-7)


def _loss_table(*, failing_component: str | None = None) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for component_index, component in enumerate(
        ("pooled", *(item.name for item in DATA_COMPONENTS))
    ):
        baseline = 0.2 + component_index * 0.01
        for step in CALIBRATION_CHECKPOINT_STEPS:
            loss = baseline - step * 1e-5
            if component == failing_component and step == 200:
                loss = baseline + 0.01
            rows.append(
                {
                    "step": step,
                    "component": component,
                    "loss": loss,
                    "accuracy": 0.5,
                    "n_rows": 640 if component == "pooled" else 128,
                }
            )
    return pd.DataFrame(rows)


def test_validation_gate_requires_every_component_to_avoid_degradation() -> None:
    passing = summarize_validation_gate(_loss_table())
    assert passing["passed"]
    assert all(check["passed"] for check in passing["checks"])

    failing = summarize_validation_gate(_loss_table(failing_component="enhancer"))
    assert not failing["passed"]
    enhancer = next(check for check in failing["checks"] if check["component"] == "enhancer")
    assert not enhancer["passed"]
    assert enhancer["delta"] > 0


def test_validation_plot_writes_reviewable_svg_and_png(tmp_path: Path) -> None:
    output = tmp_path / "validation-trajectories"
    plot_validation_trajectories(_loss_table(), output)
    svg = output.with_suffix(".svg")
    assert svg.is_file()
    assert output.with_suffix(".png").is_file()
    rendered = svg.read_text(encoding="utf-8")
    assert "Pooled fixed-plan validation" in rendered
    assert "Five validation components" in rendered
    assert "AdamW 1e-6 causal fine-tuning sanity check" in rendered


def test_training_stability_plot_writes_svg_and_png(tmp_path: Path) -> None:
    trace = pd.DataFrame(
        {
            "step": [0, 1, 2],
            "train_loss": [0.3, 0.29, 0.28],
            "pre_clip_gradient_norm": [0.9, 0.8, 0.7],
        }
    )
    output = tmp_path / "training-stability"
    plot_training_stability(trace, output)
    svg = output.with_suffix(".svg")
    assert svg.is_file()
    assert output.with_suffix(".png").is_file()
    rendered = svg.read_text(encoding="utf-8")
    assert "Training-loss stability" in rendered
    assert "Gradient stability" in rendered


def test_prelaunch_projection_reserves_two_hours() -> None:
    assert projected_total_cost(24.7340) == pytest.approx(29.3140)
