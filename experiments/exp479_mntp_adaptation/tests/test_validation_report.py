from __future__ import annotations

from pathlib import Path

import pandas as pd

from exp479_mntp.validation_report import (
    ARM_MODES,
    ARM_RUNS,
    COMPONENTS,
    EXPECTED_STEPS,
    plot_validation_components,
)


def test_validation_component_plot_writes_svg_and_png(tmp_path: Path) -> None:
    rows = [
        {
            "arm": arm,
            "mode": mode,
            "step": step,
            "component": component,
            "loss": 0.2 + 0.01 * component_index + step / 100_000,
            "wandb_run_id": ARM_RUNS[arm][0],
        }
        for arm in ARM_RUNS
        for mode in ARM_MODES[arm]
        for component_index, component in enumerate(COMPONENTS)
        for step in EXPECTED_STEPS
    ]
    output = tmp_path / "validation-components"
    plot_validation_components(pd.DataFrame(rows), output)
    assert output.with_suffix(".svg").is_file()
    assert output.with_suffix(".png").is_file()
