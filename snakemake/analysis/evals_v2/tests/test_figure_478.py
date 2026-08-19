from __future__ import annotations

from pathlib import Path

import pandas as pd
from marin_dna_evals.figure_478 import plot_predictability_478


def test_plot_predictability_478_smoke(tmp_path: Path) -> None:
    models = [
        "scaling-v0.5-h640-p46M-step-215573",
        "scaling-v0.5-h2944-p4B-step-215573",
    ]
    rows: list[dict[str, object]] = []
    for region_index, region in enumerate(("cds", "upstream", "downstream")):
        for conserved in (False, True):
            for model_index, model in enumerate(models):
                mean = 1.5 - 0.2 * model_index - 0.1 * conserved + 0.05 * region_index
                rows.append(
                    {
                        "analysis_family": "primary",
                        "span": "central_32_222",
                        "region": region,
                        "feature": "all",
                        "conserved": conserved,
                        "repeat": False,
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
    output_path = tmp_path / "figure.png"
    pd.DataFrame(rows).to_parquet(summary_path, index=False)
    pd.DataFrame(controlled).to_parquet(controlled_path, index=False)
    plot_predictability_478(summary_path, controlled_path, output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 10_000
