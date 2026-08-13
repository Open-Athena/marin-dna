"""Pure tests for leaderboard ranking and projection helpers."""

from pathlib import Path

import pandas as pd
from marin_dna_evals.soft_vep_leaderboard import (
    _rank_summary,
    add_macro_average,
    leave_one_experiment_out_projection,
    load_marin_dna_models,
)
from marin_dna_evals.soft_vep_metrics import AUPRC, MEAN_GAP_GLOBAL


def test_load_marin_dna_models_filters_family_and_dataset(tmp_path: Path):
    registry = tmp_path / "models.yaml"
    registry.write_text(
        """
- id: a
  display: A
  family: marin_dna
  experiment: 1
  datasets: [mendelian_traits]
- id: b
  display: B
  family: conservation
  datasets: [mendelian_traits]
- id: c
  display: C
  family: marin_dna
  datasets: [complex_traits]
"""
    )
    models = load_marin_dna_models(registry)
    assert models[["model", "experiment_group"]].to_dict("records") == [
        {"model": "a", "experiment_group": "exp1"}
    ]


def _points() -> pd.DataFrame:
    rows = []
    for subset_offset, subset in enumerate(["s1", "s2"]):
        for model_index, model in enumerate(["a", "b", "c", "d"]):
            auprc = 0.1 * model_index + 0.01 * subset_offset
            rows.extend(
                [
                    {
                        "model": model,
                        "subset": subset,
                        "metric": AUPRC,
                        "value": auprc,
                        "higher_is_better": True,
                    },
                    {
                        "model": model,
                        "subset": subset,
                        "metric": MEAN_GAP_GLOBAL,
                        "value": 2.0 * auprc,
                        "higher_is_better": True,
                    },
                ]
            )
    return pd.DataFrame(rows)


def test_rank_and_macro_helpers_preserve_exact_order():
    points = add_macro_average(_points())
    summary, pairs = _rank_summary(points)
    gap = summary[summary["metric"] == MEAN_GAP_GLOBAL]
    assert gap["spearman"].round(12).eq(1.0).all()
    assert gap["pairwise_reversals"].eq(0).all()
    assert not pairs[pairs["metric"] == MEAN_GAP_GLOBAL]["reversal"].any()


def test_projection_holds_out_whole_experiments():
    points = add_macro_average(_points())
    models = pd.DataFrame(
        {
            "model": ["a", "b", "c", "d"],
            "experiment_group": ["e1", "e1", "e2", "e3"],
        }
    )
    predictions, summary = leave_one_experiment_out_projection(points, models)
    assert set(predictions["model"]) == {"a", "b", "c", "d"}
    assert set(summary["metric"]) == {MEAN_GAP_GLOBAL}
    assert predictions["absolute_error"].ge(0).all()
