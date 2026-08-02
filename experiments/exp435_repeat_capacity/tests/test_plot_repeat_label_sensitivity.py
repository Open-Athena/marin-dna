from __future__ import annotations

from pathlib import Path

import polars as pl

from plot_repeat_label_sensitivity import ORIENTATIONS, STRATA, load_plot_tables, plot


def _write_plot_inputs(root: Path) -> None:
    pl.DataFrame(
        {
            "stratum": list(STRATA),
            "target": ["overall"] * len(STRATA),
            "prevalence": [0.10, 0.03, 0.08, 0.11],
        }
    ).write_parquet(root / "stratum_target_counts.parquet")

    summary_rows = []
    for block in (1, 10, 19):
        for orientation in ORIENTATIONS:
            for response in ("abs_delta", "delta"):
                for stratum in STRATA:
                    summary_rows.append(
                        {
                            "block": block,
                            "orientation": orientation,
                            "response": response,
                            "repeat_stratum": stratum,
                            "target": "overall",
                            "eligible_features": 100,
                            "discoveries": block,
                        }
                    )
    pl.DataFrame(summary_rows).write_parquet(root / "target_summary.parquet")

    retention_rows = []
    for block in (1, 10, 19):
        for orientation in ORIENTATIONS:
            for response in ("abs_delta", "delta"):
                retention_rows.append(
                    {
                        "block": block,
                        "orientation": orientation,
                        "response": response,
                        "target": "overall",
                        "retention_fraction": 0.75,
                    }
                )
    pl.DataFrame(retention_rows).write_parquet(root / "repeat_free_retention.parquet")

    feature_rows = []
    for orientation in ORIENTATIONS:
        for stratum, prevalence in zip(STRATA, (0.10, 0.03, 0.08, 0.11)):
            feature_rows.append(
                {
                    "orientation": orientation,
                    "response": "abs_delta",
                    "repeat_stratum": stratum,
                    "target": "overall",
                    "prevalence": prevalence,
                    "best_auprc": prevalence * 2,
                }
            )
    pl.DataFrame(feature_rows).write_parquet(root / "feature9086.parquet")


def test_load_and_plot_repeat_label_sensitivity(tmp_path: Path) -> None:
    _write_plot_inputs(tmp_path)

    tables = load_plot_tables(tmp_path)
    assert set(tables) == {"counts", "summary", "retention", "feature"}

    png, svg = plot(tmp_path, tmp_path / "output")
    assert png.stat().st_size > 10_000
    assert svg.stat().st_size > 10_000
