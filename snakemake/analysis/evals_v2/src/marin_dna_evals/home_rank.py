"""Joint-bootstrap utilities for specialist home-rank trajectories."""

from collections.abc import Mapping, Sequence
from itertools import pairwise

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def joint_auprc_home_rank_probability(
    dataset: pd.DataFrame,
    arm_scores: Mapping[str, pd.Series | np.ndarray],
    home_arm: str,
    *,
    n_bootstrap: int = 1000,
    rng: np.random.Generator | int | None = 0,
) -> dict[str, object]:
    """Estimate ``P(home ranks first)`` with one group draw shared by all arms.

    The input rows must be identical across arms. Each bootstrap iteration samples
    ``match_group`` clusters once, then evaluates every arm on that same resample.
    This preserves the paired cross-arm covariance that independent per-arm
    bootstraps would discard. Exact AUPRC ties count as ranking first.
    """
    required = {"label", "match_group"}
    missing = required - set(dataset.columns)
    assert not missing, f"dataset missing columns: {sorted(missing)}"
    assert len(arm_scores) >= 2, "home-rank analysis requires at least two arms"
    assert home_arm in arm_scores, f"home arm {home_arm!r} is absent from arm_scores"
    assert n_bootstrap > 0, "n_bootstrap must be positive"

    labels = np.asarray(dataset["label"]).astype(int)
    assert set(np.unique(labels)) <= {0, 1}, "label must be binary"
    n_pos = int(labels.sum())
    assert 0 < n_pos < len(labels), f"AUPRC undefined: n_pos={n_pos} of n={len(labels)}"

    arms = tuple(arm_scores)
    scores = np.column_stack([np.asarray(arm_scores[arm], dtype=float) for arm in arms])
    assert scores.shape == (len(dataset), len(arms)), (
        f"score shape {scores.shape} does not match "
        f"dataset/arms {(len(dataset), len(arms))}"
    )
    assert np.isfinite(scores).all(), "arm scores contain NaN or infinity"

    point = {
        arm: float(average_precision_score(labels, scores[:, i]))
        for i, arm in enumerate(arms)
    }
    point_max = max(point.values())
    point_winners = tuple(arm for arm in arms if point[arm] == point_max)

    match_group = np.asarray(dataset["match_group"])
    assert not pd.isna(match_group).any(), "match_group contains null values"
    group_to_rows = list(
        pd.Series(np.arange(len(dataset)))
        .groupby(match_group, sort=True)
        .apply(np.asarray)
    )
    n_groups = len(group_to_rows)
    assert n_groups > 0, "no match groups"

    generator = np.random.default_rng(rng)
    home_index = arms.index(home_arm)
    home_first = 0
    n_valid = 0
    for _ in range(n_bootstrap):
        sampled_groups = generator.integers(0, n_groups, size=n_groups)
        row_index = np.concatenate([group_to_rows[i] for i in sampled_groups])
        sampled_labels = labels[row_index]
        sampled_n_pos = int(sampled_labels.sum())
        if sampled_n_pos == 0 or sampled_n_pos == len(sampled_labels):
            continue
        aps = np.asarray(
            [
                average_precision_score(sampled_labels, scores[row_index, i])
                for i in range(len(arms))
            ]
        )
        home_first += int(aps[home_index] == aps.max())
        n_valid += 1

    assert n_valid > 0, "every bootstrap resample was single-class"
    return {
        "point_auprc": point,
        "point_winners": point_winners,
        "home_is_point_winner": home_arm in point_winners,
        "home_rank_first_probability": home_first / n_valid,
        "n_bootstrap": n_bootstrap,
        "n_bootstrap_valid": n_valid,
        "n_groups": n_groups,
        "n_rows": len(dataset),
        "n_pos": n_pos,
    }


def first_persistent_checkpoint(
    checkpoints: Sequence[int],
    probabilities: Sequence[float],
    *,
    threshold: float = 0.95,
    consecutive: int = 2,
) -> int | None:
    """Return the first checkpoint in the first persistent threshold run."""
    assert len(checkpoints) == len(probabilities), "checkpoint/probability mismatch"
    assert consecutive > 0, "consecutive must be positive"
    assert 0 <= threshold <= 1, "threshold must be in [0, 1]"
    assert all(a < b for a, b in pairwise(checkpoints)), (
        "checkpoints must be strictly increasing"
    )

    run = 0
    for index, probability in enumerate(probabilities):
        run = run + 1 if probability >= threshold else 0
        if run == consecutive:
            return int(checkpoints[index - consecutive + 1])
    return None
