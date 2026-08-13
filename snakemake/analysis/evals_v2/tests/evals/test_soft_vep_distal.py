import pandas as pd

from marin_dna_evals.soft_vep_distal import (
    DISTAL_KEY,
    history_rows,
    patched_panel_summary,
)


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
    assert result["source"].eq("https://wandb.ai/entity/project/run").all()
    assert pd.api.types.is_integer_dtype(result["step"])


def test_patched_panel_summary_combines_non_distal_and_distal_readiness():
    specialist_wins = pd.DataFrame(
        {
            "metric": ["auprc"] * 7,
            "earliest_persistent_step": [500, 1000, 1500, 2000, 2500, 3000, 500],
        }
    )
    rows = []
    arms = {
        "exp232 cCRE baseline": [0.1, 0.1, 0.1],
        "exp326 A: no exon overlap": [0.1, 0.2, 0.3],
        "exp326 B: no exon overlap, enhancer-only": [0.2, 0.1, 0.2],
        "exp351 tiled": [0.2, 0.1, 0.2],
        "exp351 centered": [0.1, 0.2, 0.3],
    }
    for arm, values in arms.items():
        experiment = "exp351" if arm.startswith("exp351") else "exp326"
        rows.extend(
            {
                "experiment": experiment,
                "arm": arm,
                "step": step,
                "history_index": index,
                "auprc": value,
            }
            for index, (step, value) in enumerate(zip((500, 1000, 1500), values))
        )
    result = patched_panel_summary(pd.DataFrame(rows), specialist_wins)
    assert result["non_distal_ready_step"].eq(3000).all()
    assert result["distal_first_two_win_step"].eq(1000).all()
    assert result["composite_ready_step"].eq(3000).all()
    assert not result["distal_soft_metrics_available"].any()
