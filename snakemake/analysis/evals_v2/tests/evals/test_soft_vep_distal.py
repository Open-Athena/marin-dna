import pandas as pd

from marin_dna_evals.soft_vep_distal import DISTAL_KEY, history_rows


def test_history_rows_preserves_every_finite_point_in_scan_order():
    history = [
        {"_step": 500, DISTAL_KEY: 0.11},
        {"_step": 1000, DISTAL_KEY: None},
        {"_step": 1500, DISTAL_KEY: 0.15},
        {"_step": 1500, DISTAL_KEY: 0.16},
    ]
    result = history_rows(
        history,
        arm="arm",
        experiment="exp",
        run_path="entity/project/run",
    )
    assert result[["step", "history_index", "auprc"]].to_dict("records") == [
        {"step": 500, "history_index": 0, "auprc": 0.11},
        {"step": 1500, "history_index": 2, "auprc": 0.15},
        {"step": 1500, "history_index": 3, "auprc": 0.16},
    ]
    assert result["source"].eq(
        "https://wandb.ai/entity/project/run"
    ).all()
    assert pd.api.types.is_integer_dtype(result["step"])
