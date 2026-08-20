from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd
import pytest
from marin_dna_evals.figure_478 import (
    plot_classification_orientation_478,
    plot_compute_efficiency_478,
    plot_conservation_classification_478,
    plot_loss_delta_classification_478,
    plot_nonrepeat_conservation_loss_478,
    plot_practical_delta_orientation_478,
    plot_predictability_478,
    plot_token_composition_478,
)


@pytest.mark.parametrize("suffix", ["png", "svg"])
def test_plot_predictability_478_smoke(tmp_path: Path, suffix: str) -> None:
    models = [
        "scaling-v0.5-h640-p46M-step-215573",
        "scaling-v0.5-h2944-p4B-step-215573",
    ]
    rows: list[dict[str, object]] = []
    for region_index, region in enumerate(("cds", "upstream", "downstream")):
        for conserved in (False, True):
            for repeat in (False, True):
                for model_index, model in enumerate(models):
                    mean = (
                        1.5
                        - 0.2 * model_index
                        - 0.1 * conserved
                        + 0.03 * repeat
                        + 0.05 * region_index
                    )
                    rows.append(
                        {
                            "analysis_family": "primary",
                            "span": "central_32_222",
                            "region": region,
                            "feature": "all",
                            "conserved": conserved,
                            "repeat": repeat,
                            "score_kind": "absolute_nll",
                            "model_from": model,
                            "model_to": model,
                            "mean": mean,
                            "ci_low": mean - 0.02,
                            "ci_high": mean + 0.02,
                            "n_positions": 100,
                        }
                    )
        for conserved in (False, True):
            for repeat in (False, True):
                mean = 0.2 + 0.05 * conserved - 0.03 * repeat
                rows.append(
                    {
                        "analysis_family": "primary",
                        "span": "central_32_222",
                        "region": region,
                        "feature": "all",
                        "conserved": conserved,
                        "repeat": repeat,
                        "score_kind": "endpoint_delta",
                        "model_from": models[0],
                        "model_to": models[-1],
                        "mean": mean,
                        "ci_low": mean - 0.02,
                        "ci_high": mean + 0.02,
                        "n_positions": 100,
                    }
                )
    for family, features in (
        ("secondary_codon", ("codon_1", "codon_2", "codon_3")),
        (
            "secondary_splice",
            ("splice_donor_2bp", "splice_acceptor_2bp"),
        ),
    ):
        for feature in features:
            for strand in ("plus", "minus"):
                rows.append(
                    {
                        "analysis_family": family,
                        "span": "central_32_222",
                        "region": "cds",
                        "feature": feature,
                        "feature_strand": strand,
                        "conserved": False,
                        "repeat": False,
                        "score_kind": "endpoint_delta",
                        "model_from": models[0],
                        "model_to": models[-1],
                        "mean": 0.2,
                        "ci_low": 0.18,
                        "ci_high": 0.22,
                        "n_positions": 20,
                    }
                )
    controlled = []
    for region in ("cds", "upstream", "downstream"):
        for term, estimate in (
            ("conserved", 0.05),
            ("repeat", -0.03),
            ("conserved_x_repeat", 0.01),
        ):
            controlled.append(
                {
                    "region": region,
                    "score_kind": "endpoint_delta",
                    "term": term,
                    "estimate": estimate,
                    "ci_low": estimate - 0.01,
                    "ci_high": estimate + 0.01,
                }
            )
    summary_path = tmp_path / "summary.parquet"
    controlled_path = tmp_path / "controlled.parquet"
    output_path = tmp_path / f"figure.{suffix}"
    pd.DataFrame(rows).to_parquet(summary_path, index=False)
    pd.DataFrame(controlled).to_parquet(controlled_path, index=False)
    plot_predictability_478(summary_path, controlled_path, output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 10_000


@pytest.mark.parametrize("suffix", ["png", "svg"])
def test_plot_nonrepeat_and_composition_smoke(
    tmp_path: Path,
    suffix: str,
) -> None:
    models = [
        "scaling-v0.5-h640-p46M-step-215573",
        "scaling-v0.5-h768-p76M-step-215573",
        "scaling-v0.5-h896-p128M-step-215573",
        "scaling-v0.5-h1152-p255M-step-215573",
        "scaling-v0.5-h1408-p476M-step-215573",
        "scaling-v0.5-h1920-p1B-step-215573",
        "scaling-v0.5-h2432-p2B-step-215573",
        "scaling-v0.5-h2944-p4B-step-215573",
    ]
    counts = {
        ("cds", False, False): 150,
        ("cds", True, False): 120,
        ("cds", False, True): 20,
        ("cds", True, True): 10,
        ("upstream", False, False): 210,
        ("upstream", True, False): 50,
        ("upstream", False, True): 35,
        ("upstream", True, True): 5,
        ("downstream", False, False): 215,
        ("downstream", True, False): 40,
        ("downstream", False, True): 40,
        ("downstream", True, True): 5,
    }
    rows = []
    for region in ("cds", "upstream", "downstream"):
        for conserved in (False, True):
            for repeat in (False, True):
                for model_index, model in enumerate(models):
                    rows.append(
                        {
                            "analysis_family": "primary",
                            "span": "central_32_222",
                            "region": region,
                            "conserved": conserved,
                            "repeat": repeat,
                            "score_kind": "absolute_nll",
                            "model_from": model,
                            "mean": 1.4 - 0.05 * model_index - 0.2 * conserved,
                            "n_positions": counts[(region, conserved, repeat)],
                        }
                    )
    summary_path = tmp_path / "summary.parquet"
    pd.DataFrame(rows).to_parquet(summary_path, index=False)
    loss_path = tmp_path / f"loss.{suffix}"
    composition_path = tmp_path / f"composition.{suffix}"
    plot_nonrepeat_conservation_loss_478(summary_path, loss_path)
    plot_token_composition_478(summary_path, composition_path)
    assert loss_path.stat().st_size > 5_000
    assert composition_path.stat().st_size > 5_000


@pytest.mark.parametrize("suffix", ["png", "svg"])
def test_plot_conservation_classification_smoke(
    tmp_path: Path,
    suffix: str,
) -> None:
    models = [
        "scaling-v0.5-h640-p46M-step-215573",
        "scaling-v0.5-h768-p76M-step-215573",
        "scaling-v0.5-h896-p128M-step-215573",
        "scaling-v0.5-h1152-p255M-step-215573",
        "scaling-v0.5-h1408-p476M-step-215573",
        "scaling-v0.5-h1920-p1B-step-215573",
        "scaling-v0.5-h2432-p2B-step-215573",
        "scaling-v0.5-h2944-p4B-step-215573",
    ]
    prevalence = {
        "global": 0.28,
        "cds": 0.45,
        "upstream": 0.20,
        "downstream": 0.17,
    }
    rows: list[dict[str, object]] = []
    for scope, baseline in prevalence.items():
        for statistic_index, statistic in enumerate(("loss", "entropy")):
            for model_index, model in enumerate(models):
                rows.append(
                    {
                        "scope": scope,
                        "statistic": statistic,
                        "model_from": model,
                        "model_to": model,
                        "orientation": "fwd_rc_mean",
                        "auprc": baseline + 0.03 * model_index + 0.01 * statistic_index,
                        "prevalence": baseline,
                        "auprc_minus_prevalence": (
                            0.03 * model_index + 0.01 * statistic_index
                        ),
                    }
                )
        for pair_index, (smaller, larger) in enumerate(combinations(models, 2)):
            lift = 0.02 + 0.003 * pair_index
            rows.append(
                {
                    "scope": scope,
                    "statistic": "loss_delta",
                    "model_from": smaller,
                    "model_to": larger,
                    "orientation": "fwd_rc_mean",
                    "auprc": baseline + lift,
                    "prevalence": baseline,
                    "auprc_minus_prevalence": lift,
                }
            )
    metrics_path = tmp_path / "metrics.parquet"
    pd.DataFrame(rows).to_parquet(metrics_path, index=False)
    averaged = pd.DataFrame(rows)
    single_orientations = pd.concat(
        [
            averaged.assign(
                orientation=orientation,
                auprc=averaged["auprc"] - offset,
                auprc_minus_prevalence=(averaged["auprc_minus_prevalence"] - offset),
            )
            for orientation, offset in (("fwd", 0.01), ("rc", 0.015))
        ],
        ignore_index=True,
    )
    orientation_path = tmp_path / "orientation_metrics.parquet"
    single_orientations.to_parquet(orientation_path, index=False)
    absolute_path = tmp_path / f"absolute.{suffix}"
    delta_path = tmp_path / f"delta.{suffix}"
    orientation_plot_path = tmp_path / f"orientation.{suffix}"
    practical_delta_path = tmp_path / f"practical_delta.{suffix}"
    compute_path = tmp_path / f"compute.{suffix}"
    plot_conservation_classification_478(metrics_path, absolute_path)
    plot_loss_delta_classification_478(metrics_path, delta_path)
    plot_classification_orientation_478(
        metrics_path,
        orientation_path,
        orientation_plot_path,
        statistic="loss",
    )
    plot_practical_delta_orientation_478(
        metrics_path,
        orientation_path,
        practical_delta_path,
    )
    plot_compute_efficiency_478(
        orientation_path,
        compute_path,
    )
    assert absolute_path.stat().st_size > 5_000
    assert delta_path.stat().st_size > 5_000
    assert orientation_plot_path.stat().st_size > 5_000
    assert practical_delta_path.stat().st_size > 5_000
    assert compute_path.stat().st_size > 5_000
