from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from exp479_mntp.source_validation_reproduction import (
    EXPECTED_ROWS_PER_COMPONENT,
    ORIGINAL_WANDB_LOSSES,
    ORIGINAL_WANDB_MACRO,
    _add_validation_statistics,
    _new_total,
    plot_source_parity,
    summarize_source_parity,
)


def _parity_frame(*, delta: float = 0.0) -> pd.DataFrame:
    rows = []
    for metric, original in ORIGINAL_WANDB_LOSSES.items():
        component, slice_name = metric.split("/")
        rows.append(
            {
                "component": component,
                "slice": slice_name,
                "metric": metric,
                "n_rows": EXPECTED_ROWS_PER_COMPONENT,
                "corrected_validation_ce": original + 0.02,
                "original_evaluator_loss": original + delta,
                "original_wandb_loss": original,
                "original_evaluator_delta": delta,
                "original_evaluator_absolute_delta": abs(delta),
            }
        )
    return pd.DataFrame(rows)


def test_original_macro_matches_the_mean_of_nine_slices() -> None:
    assert sum(ORIGINAL_WANDB_LOSSES.values()) / len(ORIGINAL_WANDB_LOSSES) == pytest.approx(
        ORIGINAL_WANDB_MACRO
    )


def test_pinned_evaluator_applies_default_repeat_weights_twice() -> None:
    total = _new_total()
    _add_validation_statistics(
        total,
        per_token_ce=torch.tensor([2.0, 10.0]),
        per_token_z_loss=torch.zeros(2),
        labels=torch.tensor([1, 1]),
        weights=torch.tensor([1.0, 0.01]),
    )
    assert total["weighted_ce_sum"] / total["loss_weight_sum"] == pytest.approx(2.1 / 1.01)
    assert total["squared_weight_ce_sum"] / total["loss_weight_sum"] == pytest.approx(2.001 / 1.01)


def test_source_parity_gate_checks_every_slice_and_macro() -> None:
    passing = summarize_source_parity(_parity_frame(delta=0.001))
    assert passing["passed"]
    assert passing["total_model_rows_evaluated"] == EXPECTED_ROWS_PER_COMPONENT * 3
    assert passing["corrected_macro"] == pytest.approx(ORIGINAL_WANDB_MACRO + 0.02)
    failing = summarize_source_parity(_parity_frame(delta=0.003))
    assert not failing["passed"]


def test_source_parity_plot_writes_svg_and_png(tmp_path: Path) -> None:
    output = tmp_path / "source-parity"
    plot_source_parity(_parity_frame(), output)
    assert output.with_suffix(".svg").is_file()
    assert output.with_suffix(".png").is_file()
