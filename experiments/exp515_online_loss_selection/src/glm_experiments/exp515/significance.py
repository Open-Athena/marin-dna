"""Matched-group significance gates for issue #515 evaluations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

KEY_COLUMNS = ["chrom", "pos", "ref", "alt", "label", "match_group"]


def _batched_average_precision(
    score_matrix: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Compute exact AP for score rows, including tied prediction scores."""

    order = np.argsort(-score_matrix, axis=1, kind="stable")
    sorted_scores = np.take_along_axis(score_matrix, order, axis=1)
    sorted_labels = labels[order]
    ranks = np.arange(1, score_matrix.shape[1] + 1)
    positives = int(labels.sum())
    result = np.empty(len(score_matrix), dtype=np.float64)
    for row in range(len(score_matrix)):
        tie_ends = np.r_[sorted_scores[row, :-1] != sorted_scores[row, 1:], True]
        cumulative = np.cumsum(sorted_labels[row])
        positive_at_ends = cumulative[tie_ends]
        previous = np.r_[0, positive_at_ends[:-1]]
        block_positives = positive_at_ends - previous
        result[row] = (
            np.sum(block_positives * positive_at_ends / ranks[tie_ends]) / positives
        )
    return result


def paired_group_swap_p_worse(
    candidate_csv: Path,
    bridge_csv: Path,
    *,
    permutations: int,
    seed: int,
    batch_size: int = 250,
) -> dict[str, Any]:
    """Test whether candidate AP is worse by swapping model identity per match group."""

    if permutations <= 0 or batch_size <= 0:
        raise ValueError("permutations and batch size must be positive")
    candidate = pd.read_csv(candidate_csv)
    bridge = pd.read_csv(bridge_csv)
    missing = set(KEY_COLUMNS + ["minus_llr_score"]) - set(candidate.columns)
    missing |= set(KEY_COLUMNS + ["minus_llr_score"]) - set(bridge.columns)
    if missing:
        raise ValueError(f"evaluation CSV lacks columns {sorted(missing)}")
    if not candidate[KEY_COLUMNS].equals(bridge[KEY_COLUMNS]):
        raise ValueError("candidate and bridge evaluation rows are not aligned")

    labels = candidate["label"].astype(bool).to_numpy()
    groups, group_index = np.unique(
        candidate["match_group"].to_numpy(),
        return_inverse=True,
    )
    group_sizes = np.bincount(group_index)
    positive_counts = np.bincount(group_index, weights=labels.astype(np.int8))
    if (
        len(groups) == 0
        or not np.all(group_sizes == group_sizes[0])
        or not np.all(positive_counts == 1)
    ):
        raise ValueError(
            "evaluation must have equal matched groups with one positive each"
        )

    candidate_scores = candidate["minus_llr_score"].to_numpy(dtype=np.float64)
    bridge_scores = bridge["minus_llr_score"].to_numpy(dtype=np.float64)
    if not np.isfinite(candidate_scores).all() or not np.isfinite(bridge_scores).all():
        raise ValueError("evaluation scores must be finite")
    observed_candidate = float(average_precision_score(labels, candidate_scores))
    observed_bridge = float(average_precision_score(labels, bridge_scores))
    observed_delta = observed_candidate - observed_bridge

    rng = np.random.default_rng(seed)
    worse_or_equal = 0
    for start in range(0, permutations, batch_size):
        stop = min(start + batch_size, permutations)
        swap_group = rng.integers(
            0,
            2,
            size=(stop - start, len(groups)),
            dtype=np.int8,
        ).astype(bool)
        swap_row = swap_group[:, group_index]
        permuted_candidate = np.where(
            swap_row,
            bridge_scores,
            candidate_scores,
        )
        permuted_bridge = np.where(
            swap_row,
            candidate_scores,
            bridge_scores,
        )
        deltas = _batched_average_precision(
            permuted_candidate,
            labels,
        ) - _batched_average_precision(permuted_bridge, labels)
        worse_or_equal += int(np.count_nonzero(deltas <= observed_delta))

    return {
        "candidate_auprc": observed_candidate,
        "bridge_auprc": observed_bridge,
        "delta_auprc": observed_delta,
        "p_worse_one_sided": (worse_or_equal + 1) / (permutations + 1),
        "permutations": permutations,
        "seed": seed,
        "randomization_unit": "match_group",
    }


def paired_group_swap_p_two_sided(
    candidate_csv: Path,
    reference_csv: Path,
    *,
    permutations: int,
    seed: int,
    batch_size: int = 250,
) -> dict[str, Any]:
    """Test either-direction AP difference by swapping model identity per group."""

    if permutations <= 0 or batch_size <= 0:
        raise ValueError("permutations and batch size must be positive")
    candidate = pd.read_csv(candidate_csv)
    reference = pd.read_csv(reference_csv)
    missing = set(KEY_COLUMNS + ["minus_llr_score"]) - set(candidate.columns)
    missing |= set(KEY_COLUMNS + ["minus_llr_score"]) - set(reference.columns)
    if missing:
        raise ValueError(f"evaluation CSV lacks columns {sorted(missing)}")
    if not candidate[KEY_COLUMNS].equals(reference[KEY_COLUMNS]):
        raise ValueError("candidate and reference evaluation rows are not aligned")

    labels = candidate["label"].astype(bool).to_numpy()
    groups, group_index = np.unique(
        candidate["match_group"].to_numpy(),
        return_inverse=True,
    )
    group_sizes = np.bincount(group_index)
    positive_counts = np.bincount(group_index, weights=labels.astype(np.int8))
    if (
        len(groups) == 0
        or not np.all(group_sizes == group_sizes[0])
        or not np.all(positive_counts == 1)
    ):
        raise ValueError(
            "evaluation must have equal matched groups with one positive each"
        )

    candidate_scores = candidate["minus_llr_score"].to_numpy(dtype=np.float64)
    reference_scores = reference["minus_llr_score"].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(candidate_scores).all()
        or not np.isfinite(reference_scores).all()
    ):
        raise ValueError("evaluation scores must be finite")
    observed_candidate = float(average_precision_score(labels, candidate_scores))
    observed_reference = float(average_precision_score(labels, reference_scores))
    observed_delta = observed_candidate - observed_reference

    rng = np.random.default_rng(seed)
    at_least_as_extreme = 0
    for start in range(0, permutations, batch_size):
        stop = min(start + batch_size, permutations)
        swap_group = rng.integers(
            0,
            2,
            size=(stop - start, len(groups)),
            dtype=np.int8,
        ).astype(bool)
        swap_row = swap_group[:, group_index]
        permuted_candidate = np.where(
            swap_row,
            reference_scores,
            candidate_scores,
        )
        permuted_reference = np.where(
            swap_row,
            candidate_scores,
            reference_scores,
        )
        deltas = _batched_average_precision(
            permuted_candidate,
            labels,
        ) - _batched_average_precision(permuted_reference, labels)
        at_least_as_extreme += int(
            np.count_nonzero(np.abs(deltas) >= abs(observed_delta))
        )

    return {
        "candidate_auprc": observed_candidate,
        "reference_auprc": observed_reference,
        "delta_auprc": observed_delta,
        "p_two_sided": (at_least_as_extreme + 1) / (permutations + 1),
        "permutations": permutations,
        "seed": seed,
        "randomization_unit": "match_group",
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Return Holm familywise adjusted p-values by key."""

    if not p_values:
        return {}
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running_maximum = 0.0
    hypotheses = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        if not 0.0 <= value <= 1.0:
            raise ValueError("p-values must be between zero and one")
        running_maximum = max(
            running_maximum,
            min(1.0, value * (hypotheses - rank)),
        )
        adjusted[name] = running_maximum
    return {name: adjusted[name] for name in p_values}


def statistically_not_worse_gate(
    bridge_csv: Path,
    candidates: dict[str, Path],
    *,
    permutations: int = 20_000,
    alpha: float = 0.05,
    seed: int = 515,
) -> dict[str, Any]:
    """Drop only candidates significantly worse than bridge after Holm correction."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    tests = {
        name: paired_group_swap_p_worse(
            path,
            bridge_csv,
            permutations=permutations,
            seed=seed + index,
        )
        for index, (name, path) in enumerate(candidates.items())
    }
    adjusted = holm_adjust(
        {name: float(test["p_worse_one_sided"]) for name, test in tests.items()}
    )
    for name, test in tests.items():
        test["p_worse_holm"] = adjusted[name]
        test["continue_to_endpoint"] = adjusted[name] >= alpha
    return {
        "alpha": alpha,
        "correction": "Holm familywise",
        "alternative": "candidate AUPRC is worse than bridge AUPRC",
        "decision_rule": "drop only when Holm-adjusted p_worse is below alpha",
        "tests": tests,
    }
