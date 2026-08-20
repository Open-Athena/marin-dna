from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from marin_dna_evals.figure_orientation_478 import (
    plot_orientation_sensitivity_478,
)
from marin_dna_evals.orientation_478 import (
    analyze_orientation_sensitivity_478,
    orientation_agreement_row,
)


def test_orientation_agreement_row_identical_scores() -> None:
    values = np.linspace(-1.0, 1.0, 100)
    row = orientation_agreement_row(
        values,
        values.copy(),
        score_kind="endpoint_delta",
        top_fraction=0.1,
        rank_sample_size=50,
        rng=np.random.default_rng(478),
    )

    assert row["pearson"] == pytest.approx(1.0)
    assert row["spearman_sample"] == pytest.approx(1.0)
    assert row["top_fraction_overlap"] == pytest.approx(1.0)
    assert row["bottom_fraction_overlap"] == pytest.approx(1.0)
    assert row["sign_agreement"] == pytest.approx(1.0)
    assert row["mae"] == pytest.approx(0.0)
    assert row["spearman_sample_n"] == 50


def _write_atom(
    path: Path,
    window_ids: list[str],
    nll: np.ndarray,
    entropy: np.ndarray,
) -> None:
    pd.DataFrame(
        {
            "window_id": window_ids,
            "nll": list(nll.astype(np.float32)),
            "entropy_4nuc": list(entropy.astype(np.float32)),
        }
    ).to_parquet(path, index=False)


def test_analyze_orientation_sensitivity_contract(tmp_path: Path) -> None:
    width = 8
    window_ids = [f"NC_1:{start}-{start + width}" for start in (0, 100, 200, 300)]
    conserved = np.asarray(
        [
            np.zeros(width, dtype=bool),
            np.zeros(width, dtype=bool),
            np.ones(width, dtype=bool),
            np.ones(width, dtype=bool),
        ]
    )
    repeat = np.asarray(
        [
            np.zeros(width, dtype=bool),
            np.ones(width, dtype=bool),
            np.zeros(width, dtype=bool),
            np.ones(width, dtype=bool),
        ]
    )
    joined = pd.DataFrame(
        {
            "window_id": window_ids,
            "region": "upstream",
            "chrom": "NC_1",
            "start": [0, 100, 200, 300],
            "end": [8, 108, 208, 308],
            "sequence_upper": ["ACGTACGT"] * 4,
            "is_conserved": list(conserved),
            "is_repeat": list(repeat),
            "is_ambiguous": list(np.zeros((4, width), dtype=bool)),
            "window_gc": np.linspace(0.4, 0.6, 4),
            "kmer7_nll": list(np.full((4, width), 1.4, dtype=np.float32)),
        }
    )
    joined_path = tmp_path / "joined.parquet"
    joined.to_parquet(joined_path, index=False)

    models = ["model-p46M-step", "model-p4B-step"]
    atom_paths: dict[tuple[str, str, str], Path] = {}
    position = np.tile(np.arange(width), (4, 1))
    small = 1.5 + 0.01 * position - 0.15 * conserved + 0.04 * repeat
    gain = 0.1 + 0.2 * conserved - 0.03 * repeat
    for orientation, offset in (("fwd", 0.02), ("rc", -0.02)):
        for model, values in (
            (models[0], small + offset),
            (models[1], small + offset - gain),
        ):
            path = tmp_path / f"{model}.{orientation}.parquet"
            _write_atom(
                path,
                window_ids,
                values,
                np.full((4, width), 0.8 + offset),
            )
            atom_paths[(model, "upstream", orientation)] = path

    summary, controlled, agreement, manifest = analyze_orientation_sensitivity_478(
        {"upstream": joined_path},
        atom_paths,
        model_order=models,
        window_size=width,
        primary_start=1,
        primary_end_exclusive=7,
        block_bp=100,
        bootstrap_replicates=20,
        seed=478,
        top_fraction=0.1,
        rank_sample_size=10,
    )

    assert len(summary) == 80
    assert set(summary["orientation"]) == {"fwd", "rc"}
    assert set(summary["span"]) == {"central_32_222", "all_255"}
    assert len(controlled) == 88
    assert len(agreement) == 45
    assert set(agreement["score_kind"]) == {
        "absolute_nll_46m",
        "predictive_entropy_46m",
        "endpoint_delta",
    }
    assert manifest["primary_analysis"].startswith("FWD/RC mean")
    endpoint = agreement[
        (agreement["score_kind"] == "endpoint_delta")
        & (agreement["conservation"] == "all")
        & (agreement["comparison"] == "fwd_vs_mean")
    ].iloc[0]
    assert endpoint["pearson"] == pytest.approx(1.0)
    assert endpoint["sign_agreement"] == pytest.approx(1.0)


@pytest.mark.parametrize("suffix", ["png", "svg"])
def test_plot_orientation_sensitivity_smoke(tmp_path: Path, suffix: str) -> None:
    models = ["scaling-p46M-step", "scaling-p4B-step"]
    summary_rows: list[dict[str, object]] = []
    controlled_rows: list[dict[str, object]] = []
    averaged_rows: list[dict[str, object]] = []
    agreement_rows: list[dict[str, object]] = []
    for region_index, region in enumerate(("cds", "upstream", "downstream")):
        for orientation_index, orientation in enumerate(("fwd", "rc")):
            for conserved in (False, True):
                for repeat in (False, True):
                    for model_index, model in enumerate(models):
                        mean = (
                            1.4
                            - 0.3 * model_index
                            - 0.1 * conserved
                            + 0.04 * repeat
                            + 0.03 * region_index
                            + 0.01 * orientation_index
                        )
                        summary_rows.append(
                            {
                                "analysis_family": "primary",
                                "span": "central_32_222",
                                "region": region,
                                "conserved": conserved,
                                "repeat": repeat,
                                "score_kind": "absolute_nll",
                                "model_from": model,
                                "mean": mean,
                                "orientation": orientation,
                            }
                        )
            for term_index, term in enumerate(
                ("conserved", "repeat", "conserved_x_repeat")
            ):
                estimate = 0.3 - 0.1 * term_index + 0.01 * orientation_index
                controlled_rows.append(
                    {
                        "orientation": orientation,
                        "region": region,
                        "score_kind": "endpoint_delta",
                        "term": term,
                        "estimate": estimate,
                        "ci_low": estimate - 0.02,
                        "ci_high": estimate + 0.02,
                    }
                )
        for term_index, term in enumerate(
            ("conserved", "repeat", "conserved_x_repeat")
        ):
            estimate = 0.3 - 0.1 * term_index + 0.005
            averaged_rows.append(
                {
                    "region": region,
                    "score_kind": "endpoint_delta",
                    "term": term,
                    "estimate": estimate,
                    "ci_low": estimate - 0.02,
                    "ci_high": estimate + 0.02,
                }
            )
        for comparison in ("fwd_vs_mean", "rc_vs_mean"):
            agreement_rows.append(
                {
                    "region": region,
                    "score_kind": "endpoint_delta",
                    "conservation": "all",
                    "repeat_status": "all",
                    "comparison": comparison,
                    "pearson": 0.8,
                    "spearman_sample": 0.75,
                    "top_fraction_overlap": 0.5,
                    "sign_agreement": 0.9,
                }
            )

    paths = {
        "summary": tmp_path / "orientation_summary.parquet",
        "controlled": tmp_path / "orientation_controlled.parquet",
        "agreement": tmp_path / "orientation_agreement.parquet",
        "averaged": tmp_path / "controlled.parquet",
        "figure": tmp_path / f"orientation.{suffix}",
    }
    pd.DataFrame(summary_rows).to_parquet(paths["summary"], index=False)
    pd.DataFrame(controlled_rows).to_parquet(paths["controlled"], index=False)
    pd.DataFrame(agreement_rows).to_parquet(paths["agreement"], index=False)
    pd.DataFrame(averaged_rows).to_parquet(paths["averaged"], index=False)

    plot_orientation_sensitivity_478(
        paths["summary"],
        paths["controlled"],
        paths["agreement"],
        paths["averaged"],
        paths["figure"],
    )

    assert paths["figure"].exists()
    assert paths["figure"].stat().st_size > 1_000
