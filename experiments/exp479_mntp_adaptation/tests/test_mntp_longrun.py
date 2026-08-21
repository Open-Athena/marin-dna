from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from transformers import PreTrainedModel

from exp479_mntp.causal_longrun import LONGRUN_CHECKPOINT_STEPS
from exp479_mntp.mntp_longrun import (
    MNTP_AUPRC_STEPS,
    MNTP_EVALUATION_ARTIFACT,
    MNTP_MODEL_ARTIFACT_PREFIX,
    MNTP_RUN_NAME,
    MntpLongRunModule,
    plot_mntp_auprc,
    plot_mntp_dependency,
    plot_mntp_validation,
    summarize_mntp_trajectory,
)


def test_mntp_longrun_uses_the_selected_single_configuration() -> None:
    assert "transferred-mntp" in MNTP_RUN_NAME
    assert "corrected" in MNTP_RUN_NAME
    assert "corrected" in MNTP_MODEL_ARTIFACT_PREFIX
    assert "corrected" in MNTP_EVALUATION_ARTIFACT
    assert MNTP_AUPRC_STEPS == (0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1_000)


def test_mntp_longrun_module_uses_full_attention_and_one_adamw_group() -> None:
    model = cast(PreTrainedModel, cast(Any, nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))))
    module = MntpLongRunModule(model=model, batch_size=2)
    assert module.arm == "transferred_mntp"
    assert module.attention_mode == "full"
    configuration = module.configure_optimizers()
    optimizer = configuration["optimizer"]
    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["betas"] == (0.9, 0.95)
    assert optimizer.param_groups[0]["eps"] == 1e-8
    assert optimizer.param_groups[0]["weight_decay"] == 0
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)


def _validation_rows(*, final_delta: float = -0.01) -> pd.DataFrame:
    rows = []
    for mode, initial in (("diffusion", 0.5), ("single", 0.4)):
        for step in LONGRUN_CHECKPOINT_STEPS:
            rows.append(
                {
                    "step": step,
                    "validation_mode": mode,
                    "component": "macro",
                    "loss": initial + final_delta * step / 1_000,
                    "accuracy": 0.3,
                    "n_rows": 640,
                }
            )
    return pd.DataFrame(rows)


def test_mntp_gate_requires_both_macros_to_improve() -> None:
    passing = summarize_mntp_trajectory(_validation_rows())
    assert passing["passed"]
    assert all(check["delta"] == pytest.approx(-0.01) for check in passing["checks"])

    failing_rows = _validation_rows()
    failing_rows.loc[failing_rows["validation_mode"] == "single", "loss"] += (
        failing_rows.loc[failing_rows["validation_mode"] == "single", "step"] / 1_000 * 0.02
    )
    failing = summarize_mntp_trajectory(failing_rows)
    assert not failing["passed"]


def test_mntp_plots_write_reviewable_svg_and_png(tmp_path) -> None:
    validation_path = tmp_path / "validation"
    plot_mntp_validation(_validation_rows(), validation_path)

    metric_rows = []
    for dataset in ("mendelian_traits", "complex_traits", "sge"):
        for step in MNTP_AUPRC_STEPS:
            metric_rows.append(
                {
                    "dataset": dataset,
                    "orientation": "protocol_fwd_rc",
                    "step": step,
                    "auprc": 0.4 + step * 1e-5,
                    "se": 0.01,
                }
            )
    auprc_path = tmp_path / "auprc"
    plot_mntp_auprc(pd.DataFrame(metric_rows), auprc_path)

    dependency_path = tmp_path / "dependency"
    matrix = np.arange(16, dtype=np.float32).reshape(4, 4)
    np.fill_diagonal(matrix, 0)
    plot_mntp_dependency(matrix, dependency_path)

    for output in (validation_path, auprc_path, dependency_path):
        assert output.with_suffix(".svg").is_file()
        assert output.with_suffix(".png").is_file()
