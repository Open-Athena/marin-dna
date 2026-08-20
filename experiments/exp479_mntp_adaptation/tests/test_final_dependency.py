from __future__ import annotations

from pathlib import Path

import numpy as np

from exp479_mntp.checkpoint_audit import ModelPoint
from exp479_mntp.final_dependency import POINTS, _plot_maps, _summary_rows


def test_final_dependency_points_cover_each_trained_arm() -> None:
    assert {point.arm for point in POINTS} == {
        "transferred_mntp",
        "scratch_mntp",
        "clm_continuation",
    }
    assert {point.step for point in POINTS} == {1000}


def test_directed_regions_and_plot(tmp_path: Path) -> None:
    point = ModelPoint(
        "test-step1000",
        "transferred_mntp",
        1000,
        "mntp",
        "hf",
        "Test MNTP",
    )
    matrix = np.zeros((5, 5), dtype=np.float32)
    matrix[1, 3] = 2.0
    matrix[3, 1] = 4.0
    rows = {row["region"]: row for row in _summary_rows(point, matrix)}
    assert rows["past_context"]["maximum_dependency"] == 2.0
    assert rows["future_context"]["maximum_dependency"] == 4.0

    output = tmp_path / "dependency"
    _plot_maps([(point, matrix), (point, matrix), (point, matrix)], output_path=output)
    assert output.with_suffix(".svg").read_text(encoding="utf-8").startswith("<?xml")
    assert output.with_suffix(".png").stat().st_size > 0
