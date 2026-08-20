from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from marin_dna_evals.classification_478 import (
    RegionScores,
    ScoreSpec,
    _score_specs,
    _score_values,
    analyze_conservation_classification_478,
)


def test_score_specs_include_all_pairs_and_expected_directions() -> None:
    model_order = ["46M", "76M", "4B"]
    specs = _score_specs(model_order)
    assert len(specs) == 9
    assert ScoreSpec("loss_delta", "46M", "76M") in specs
    assert ScoreSpec("loss_delta", "46M", "4B") in specs

    data = RegionScores(
        region="cds",
        labels=np.array([True, False]),
        blocks=np.array([1, 1]),
        nll_by_model={
            "46M": np.array([0.5, 1.0]),
            "76M": np.array([0.2, 0.9]),
            "4B": np.array([0.1, 0.8]),
        },
        entropy_by_model={
            "46M": np.array([0.2, 0.8]),
            "76M": np.array([0.1, 0.7]),
            "4B": np.array([0.05, 0.6]),
        },
        chrom_by_code={1: "NC_000001.11"},
    )
    np.testing.assert_array_equal(
        _score_values(data, ScoreSpec("loss", "46M", "46M")),
        [-0.5, -1.0],
    )
    np.testing.assert_allclose(
        _score_values(data, ScoreSpec("loss_delta", "46M", "76M")),
        [0.3, 0.1],
    )


def _write_atoms(
    path: Path,
    *,
    conserved_loss: float,
    nonconserved_loss: float,
) -> None:
    pd.DataFrame(
        {
            "window_id": ["w0", "w1"],
            "nll": [
                [
                    conserved_loss,
                    nonconserved_loss,
                    conserved_loss,
                    nonconserved_loss,
                ],
                [
                    nonconserved_loss,
                    conserved_loss,
                    nonconserved_loss,
                    conserved_loss,
                ],
            ],
            "entropy_4nuc": [
                [0.2, 0.8, 0.2, 0.8],
                [0.8, 0.2, 0.8, 0.2],
            ],
        }
    ).to_parquet(path, index=False)


def test_analysis_filters_repeats_and_reports_global_and_region(tmp_path: Path) -> None:
    joined_path = tmp_path / "joined.parquet"
    pd.DataFrame(
        {
            "window_id": ["w0", "w1"],
            "region": ["cds", "cds"],
            "chrom": ["NC_000001.11", "NC_000001.11"],
            "start": [0, 20],
            "end": [4, 24],
            "is_conserved": [
                [True, False, True, False],
                [False, True, False, True],
            ],
            "is_repeat": [
                [False, False, True, False],
                [False, False, False, False],
            ],
            "is_ambiguous": [
                [False, False, False, False],
                [False, False, False, True],
            ],
        }
    ).to_parquet(joined_path, index=False)

    atom_paths: dict[tuple[str, str, str], Path] = {}
    for model, conserved_loss, nonconserved_loss in (
        ("46M", 0.5, 1.0),
        ("76M", 0.1, 0.9),
    ):
        for orientation in ("fwd", "rc"):
            path = tmp_path / f"{model}.{orientation}.parquet"
            _write_atoms(
                path,
                conserved_loss=conserved_loss,
                nonconserved_loss=nonconserved_loss,
            )
            atom_paths[(model, "cds", orientation)] = path

    metrics, block_metrics, manifest = analyze_conservation_classification_478(
        {"cds": joined_path},
        atom_paths,
        model_order=["46M", "76M"],
        window_size=4,
        primary_start=0,
        primary_end_exclusive=4,
        block_bp=10,
        orientations=("fwd_rc_mean", "fwd", "rc"),
    )
    assert set(metrics["scope"]) == {"global", "cds"}
    assert len(metrics) == 30
    assert set(metrics["orientation"]) == {"fwd_rc_mean", "fwd", "rc"}
    assert set(block_metrics["scope"]) == {"global", "cds"}
    assert manifest["counts"]["cds"]["n_positions"] == 6
    assert manifest["counts"]["cds"]["n_conserved"] == 2
    delta = metrics[
        (metrics["scope"] == "cds")
        & (metrics["statistic"] == "loss_delta")
        & (metrics["orientation"] == "fwd")
    ].iloc[0]
    assert delta["model_from"] == "46M"
    assert delta["model_to"] == "76M"
    assert delta["auprc"] == 1.0
