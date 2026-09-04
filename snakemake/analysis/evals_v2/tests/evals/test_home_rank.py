import numpy as np
import pandas as pd
import pytest
from marin_dna_evals.home_rank import (
    first_persistent_checkpoint,
    joint_auprc_home_rank_probability,
)


def _paired_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label": [1, 0, 1, 0, 1, 0],
            "match_group": [10, 10, 20, 20, 30, 30],
        }
    )


def test_joint_home_rank_probability_uses_shared_group_resamples() -> None:
    dataset = _paired_dataset()
    result = joint_auprc_home_rank_probability(
        dataset,
        {
            "home": pd.Series([0.9, 0.1, 0.8, 0.2, 0.7, 0.3]),
            "away": pd.Series([0.1, 0.9, 0.2, 0.8, 0.3, 0.7]),
        },
        "home",
        n_bootstrap=50,
        rng=517,
    )

    assert result["point_auprc"] == {
        "home": 1.0,
        "away": pytest.approx(0.38333333333333336),
    }
    assert result["point_winners"] == ("home",)
    assert result["home_is_point_winner"] is True
    assert result["home_rank_first_probability"] == 1.0
    assert result["n_bootstrap_valid"] == 50
    assert result["n_groups"] == 3


def test_joint_home_rank_probability_counts_exact_ties_as_first() -> None:
    dataset = _paired_dataset()
    score = np.asarray([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
    result = joint_auprc_home_rank_probability(
        dataset,
        {"home": score, "away": score.copy()},
        "home",
        n_bootstrap=20,
        rng=0,
    )

    assert result["point_winners"] == ("home", "away")
    assert result["home_rank_first_probability"] == 1.0


def test_first_persistent_checkpoint_returns_start_of_first_run() -> None:
    checkpoints = [500, 1000, 1500, 2000, 2500]
    assert (
        first_persistent_checkpoint(
            checkpoints, [0.96, 0.90, 0.95, 0.97, 0.99], threshold=0.95
        )
        == 1500
    )
    assert (
        first_persistent_checkpoint(
            checkpoints, [0.96, 0.90, 0.95, 0.94, 0.99], threshold=0.95
        )
        is None
    )
