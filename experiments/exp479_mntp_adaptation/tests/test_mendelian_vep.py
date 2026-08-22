from __future__ import annotations

from pathlib import Path

import pandas as pd

from exp479_mntp.mendelian_vep import (
    paired_mendelian_trajectory,
    plot_mendelian_trajectory,
)


def test_paired_mendelian_trajectory_and_plot(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for subset in ("coding", "splicing"):
        for group in range(30):
            for label in (True, False):
                rows.append(
                    {
                        "label": label,
                        "subset": subset,
                        "match_group": f"{subset}-{group}",
                        "step_0000": float(not label),
                        "step_0100": float(label),
                    }
                )
    comparisons = paired_mendelian_trajectory(
        pd.DataFrame(rows),
        n_bootstrap=20,
    )
    assert comparisons["optimizer_step"].tolist() == [0, 100]
    assert comparisons.loc[0, "delta"] == 0.0
    assert comparisons.loc[1, "delta"] > 0
    assert comparisons.loc[1, "n_subsets"] == 2

    baseline = 0.2
    endpoints = pd.DataFrame(
        {
            "dataset": ["mendelian_traits", "mendelian_traits"],
            "optimizer_step": [0, 100],
            "auprc": [baseline, baseline + float(comparisons.loc[1, "delta"])],
        }
    )
    output_path = tmp_path / "mendelian"
    plot_mendelian_trajectory(endpoints, comparisons, output_path)
    assert output_path.with_suffix(".svg").is_file()
    assert output_path.with_suffix(".png").is_file()
