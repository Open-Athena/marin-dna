from __future__ import annotations

import pandas as pd
import pytest

from exp479_mntp.config import DATA_COMPONENTS
from exp479_mntp.loss_normalization_audit import (
    AUDIT_STEPS,
    checkpoint_artifact_name,
    macro_trajectory,
    plot_corrected_trajectory,
    plot_source_scale,
)


def _component_rows() -> pd.DataFrame:
    rows = []
    for step in AUDIT_STEPS:
        for index, component in enumerate(DATA_COMPONENTS):
            loss = 0.6 + index * 0.05 + step * 1e-5
            rows.append(
                {
                    "step": step,
                    "component": component.name,
                    "legacy_count_normalized_ce": loss * 0.3,
                    "sequence_balanced_ce": loss,
                    "marin_token_weighted_ce": loss,
                    "marin_loss": loss + 0.001,
                }
            )
    return pd.DataFrame(rows)


def test_artifact_names_are_version_pinned() -> None:
    assert checkpoint_artifact_name(25).endswith("step-0025:v0")
    assert checkpoint_artifact_name(1_000).endswith("step-1000:v0")
    with pytest.raises(ValueError, match="retained"):
        checkpoint_artifact_name(0)


def test_macro_trajectory_separates_original_and_added_validation_scopes() -> None:
    macros = macro_trajectory(_component_rows())
    assert set(macros["scope"]) == {"source_three", "all_five"}
    assert macros.groupby("scope")["step"].apply(list).to_dict() == {
        "all_five": list(AUDIT_STEPS),
        "source_three": list(AUDIT_STEPS),
    }
    step_zero = macros[macros["step"] == 0].set_index("scope")
    assert step_zero.loc["source_three", "n_components"] == 3
    assert step_zero.loc["all_five", "n_components"] == 5


def test_normalization_audit_plots_write_svg_and_png(tmp_path) -> None:
    macros = macro_trajectory(_component_rows())
    trajectory = tmp_path / "trajectory"
    scale = tmp_path / "scale"
    plot_corrected_trajectory(macros, trajectory)
    plot_source_scale(macros, scale)
    for output in (trajectory, scale):
        assert output.with_suffix(".svg").is_file()
        assert output.with_suffix(".png").is_file()
