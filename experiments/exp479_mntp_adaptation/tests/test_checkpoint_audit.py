from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from exp479_mntp.checkpoint_audit import (
    LOSS_PARITY_TOLERANCE,
    REPLAY_STEPS,
    _find_sample_id_for_position,
    attach_logged_loss_parity,
    plot_dependency_panel,
    plot_training_stability,
    trajectory_points,
    triangle_summary,
)
from exp479_mntp.masking import sample_seed


def test_deterministic_single_mask_ids_cover_alignment_boundaries() -> None:
    for position in (0, 63, 127, 191, 254):
        sample_id = _find_sample_id_for_position(position)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(sample_seed(0, sample_id, stream=1))
        assert int(torch.randint(255, (), generator=generator)) == position


def test_trajectory_points_include_zero_roundtrip_and_early_clm(tmp_path: Path) -> None:
    points = trajectory_points(tmp_path)
    by_id = {point.point_id: point for point in points}
    assert (
        tuple(point.step for point in points if point.plot_series == "Continued CLM replay")
        == REPLAY_STEPS[1:]
    )
    assert by_id["clm-replay-step0000"].path == tmp_path / "step-0000"
    assert by_id["clm-original-step0400"].kind == "lightning"
    assert by_id["clm-original-step1000"].kind == "hf"


def test_triangle_summary_distinguishes_causal_direction() -> None:
    matrix = np.zeros((255, 255), dtype=np.float32)
    matrix[3, 10] = 2
    matrix[20, 4] = 1
    summary = triangle_summary(matrix, "directed")
    assert summary["upper_mass"] == 2
    assert summary["lower_mass"] == 1
    assert summary["upper_nonzero"] == 1
    assert summary["lower_nonzero"] == 1


def test_logged_loss_parity_only_gates_original_checkpoints(tmp_path: Path) -> None:
    logged = pd.DataFrame(
        {
            "step": [400],
            "transferred_diffusion_loss": [0.4],
            "transferred_single_mask_loss": [0.3],
            "scratch_diffusion_loss": [0.5],
            "scratch_single_mask_loss": [0.4],
            "clm_diffusion_loss": [0.33],
        }
    )
    path = tmp_path / "logged.csv"
    logged.to_csv(path, index=False)
    losses = pd.DataFrame(
        [
            {
                "arm": "clm_continuation",
                "step": 400,
                "kind": "lightning",
                "validation_mode": "causal",
                "loss": 0.33 + LOSS_PARITY_TOLERANCE / 2,
            },
            {
                "arm": "clm_continuation",
                "step": 400,
                "kind": "replay",
                "validation_mode": "causal",
                "loss": 99,
            },
        ]
    )
    result = attach_logged_loss_parity(losses, path)
    assert bool(result.loc[0, "logged_parity_passed"])
    assert np.isnan(result.loc[1, "logged_loss"])


def test_training_stability_plot_writes_svg_and_png(tmp_path: Path) -> None:
    trace = pd.DataFrame(
        {
            "step": [0, 1, 2],
            "train_loss": [1.0, 0.9, 0.8],
            "pre_clip_gradient_norm": [1.2, 0.8, 0.6],
        }
    )
    output = tmp_path / "training-stability"
    plot_training_stability(
        trace,
        output,
        clip_threshold=0.995,
    )
    assert output.with_suffix(".svg").is_file()
    assert output.with_suffix(".png").is_file()


def test_dependency_panel_writes_one_column_header_and_each_locus(tmp_path: Path) -> None:
    matrix = np.eye(8, dtype=np.float32)
    maps = [(f"locus_{index}", matrix, matrix * 0.8, matrix * 0.9) for index in range(5)]
    output = tmp_path / "nucleotide-dependency-panel"
    plot_dependency_panel(maps, output)

    svg = output.with_suffix(".svg")
    assert svg.is_file()
    assert output.with_suffix(".png").is_file()
    rendered = svg.read_text(encoding="utf-8")
    assert rendered.count("Raw reference-directed") == 1
    for name, *_ in maps:
        assert name in rendered
