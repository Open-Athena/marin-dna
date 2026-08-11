"""Tests for the pairwise-accuracy metric (matched-pair within-group)."""

import math

import pandas as pd
import pytest
from marin_dna_evals.metrics import (
    GLOBAL_SUBSET,
    MACRO_AVG_SUBSET,
    compute_pairwise_metrics,
    pairwise_accuracy,
)

# ---- pairwise_accuracy ----------------------------------------------------


def _frame(label, score, match_group):
    return (
        pd.Series(label),
        pd.Series(score, dtype=float),
        pd.Series(match_group),
    )


def test_pairwise_accuracy_all_wins():
    label, score, mg = _frame([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2], [0, 0, 1, 1])
    res = pairwise_accuracy(label, score, mg)
    assert res["value"] == 1.0
    assert res["se"] == 0.0
    assert res["n_pairs"] == 2
    assert res["n_ties"] == 0


def test_pairwise_accuracy_all_losses():
    label, score, mg = _frame([1, 0, 1, 0], [0.1, 0.9, 0.2, 0.8], [0, 0, 1, 1])
    res = pairwise_accuracy(label, score, mg)
    assert res["value"] == 0.0
    assert res["se"] == 0.0
    assert res["n_pairs"] == 2
    assert res["n_ties"] == 0


def test_pairwise_accuracy_fifty_fifty():
    # 4 pairs: 2 wins, 2 losses.
    label = [1, 0, 1, 0, 1, 0, 1, 0]
    score = [0.9, 0.1, 0.1, 0.9, 0.8, 0.2, 0.2, 0.8]
    mg = [0, 0, 1, 1, 2, 2, 3, 3]
    res = pairwise_accuracy(*_frame(label, score, mg))
    assert res["value"] == 0.5
    assert res["se"] == pytest.approx(math.sqrt(0.25 / 4))
    assert res["n_pairs"] == 4
    assert res["n_ties"] == 0


def test_pairwise_accuracy_pure_ties():
    label = [1, 0, 1, 0, 1, 0]
    score = [0.5, 0.5, 0.7, 0.7, 0.0, 0.0]
    mg = [0, 0, 1, 1, 2, 2]
    res = pairwise_accuracy(*_frame(label, score, mg))
    assert res["value"] == 0.5  # ties counted as 0.5
    assert res["n_pairs"] == 3
    assert res["n_ties"] == 3
    assert res["se"] == pytest.approx(math.sqrt(0.5 * 0.5 / 3))


def test_pairwise_accuracy_mixed_wins_and_ties():
    # 4 pairs: 2 wins, 1 tie, 1 loss => value = (2 + 0.5*1) / 4 = 0.625
    label = [1, 0, 1, 0, 1, 0, 1, 0]
    score = [0.9, 0.1, 0.5, 0.5, 0.8, 0.2, 0.1, 0.9]
    mg = [0, 0, 1, 1, 2, 2, 3, 3]
    res = pairwise_accuracy(*_frame(label, score, mg))
    expected = (2 + 0.5) / 4
    assert res["value"] == pytest.approx(expected)
    assert res["se"] == pytest.approx(math.sqrt(expected * (1 - expected) / 4))
    assert res["n_pairs"] == 4
    assert res["n_ties"] == 1


def test_pairwise_accuracy_handles_bool_label():
    label = [True, False, True, False]
    score = [0.9, 0.1, 0.8, 0.2]
    mg = [0, 0, 1, 1]
    res = pairwise_accuracy(*_frame(label, score, mg))
    assert res["value"] == 1.0
    assert res["n_pairs"] == 2


def test_pairwise_accuracy_se_decreases_with_n():
    """SE for 50/50 must shrink as 1/sqrt(n)."""
    se_small = pairwise_accuracy(*_frame([1, 0, 1, 0], [1, 0, 0, 1], [0, 0, 1, 1]))[
        "se"
    ]
    n_big = 100
    label = [1, 0] * n_big
    score = ([1, 0] * (n_big // 2)) + ([0, 1] * (n_big // 2))
    mg = [i for i in range(n_big) for _ in range(2)]
    se_big = pairwise_accuracy(*_frame(label, score, mg))["se"]
    assert se_big < se_small


def test_pairwise_accuracy_rejects_extra_positive_in_group():
    # match_group 0 has 2 positives, 1 negative.
    label = [1, 1, 0, 1, 0]
    score = [0.9, 0.5, 0.1, 0.8, 0.2]
    mg = [0, 0, 0, 1, 1]
    with pytest.raises(AssertionError, match="exactly 1 positive"):
        pairwise_accuracy(*_frame(label, score, mg))


def test_pairwise_accuracy_rejects_missing_negative_in_group():
    # match_group 1 has only a positive.
    label = [1, 0, 1]
    score = [0.9, 0.1, 0.8]
    mg = [0, 0, 1]
    with pytest.raises(AssertionError, match="exactly 1 positive"):
        pairwise_accuracy(*_frame(label, score, mg))


def test_pairwise_accuracy_rejects_length_mismatch():
    with pytest.raises(AssertionError, match="length mismatch"):
        pairwise_accuracy(
            pd.Series([1, 0, 1, 0]),
            pd.Series([0.9, 0.1, 0.8]),
            pd.Series([0, 0, 1, 1]),
        )


def test_pairwise_accuracy_rejects_nan_score():
    """NaN in score is a silent-corruption risk (NaN > NaN and NaN == NaN both
    False -> NaN-vs-NaN silently counts as a loss). Caller must fill upstream."""
    label = [1, 0, 1, 0]
    score = [float("nan"), 0.1, 0.8, 0.2]
    mg = [0, 0, 1, 1]
    with pytest.raises(AssertionError, match="NaN"):
        pairwise_accuracy(*_frame(label, score, mg))


# ---- compute_pairwise_metrics --------------------------------------------


def test_compute_pairwise_metrics_per_subset():
    """Stratifies by ``subset`` and returns one row per (subset, score), plus
    aggregate ``_global_`` and ``_macro_avg_`` rows."""
    dataset = pd.DataFrame(
        {
            "label": [1, 0, 1, 0, 1, 0, 1, 0],
            "subset": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "match_group": [0, 0, 1, 1, 2, 2, 3, 3],
        }
    )
    # Subset A: both pairs are wins -> 1.0
    # Subset B: one win, one loss -> 0.5
    scores = pd.DataFrame(
        {
            "score": [0.9, 0.1, 0.8, 0.2, 0.9, 0.1, 0.1, 0.9],
        }
    )
    metrics = compute_pairwise_metrics(
        dataset=dataset, scores=scores, score_columns=["score"], n_min=2
    )
    assert set(metrics.columns) == {
        "score_type",
        "subset",
        "value",
        "se",
        "n_pairs",
        "n_ties",
    }
    # 2 subsets + _global_ + _macro_avg_, all for one score_column.
    assert len(metrics) == 4
    by_subset = metrics.set_index("subset")["value"]
    assert by_subset["A"] == 1.0
    assert by_subset["B"] == 0.5


def test_compute_pairwise_metrics_multi_score_columns():
    dataset = pd.DataFrame(
        {
            "label": [1, 0, 1, 0],
            "subset": ["A", "A", "A", "A"],
            "match_group": [0, 0, 1, 1],
        }
    )
    scores = pd.DataFrame(
        {
            "score_a": [0.9, 0.1, 0.8, 0.2],  # all wins -> 1.0
            "score_b": [0.1, 0.9, 0.2, 0.8],  # all losses -> 0.0
        }
    )
    metrics = compute_pairwise_metrics(dataset=dataset, scores=scores, n_min=1)
    # One per-subset row per score_type, plus _global_ and _macro_avg_ each.
    assert len(metrics) == 6
    per_subset = metrics[metrics["subset"] == "A"].set_index("score_type")["value"]
    assert per_subset["score_a"] == 1.0
    assert per_subset["score_b"] == 0.0


def test_compute_pairwise_metrics_rejects_subset_straddle():
    """A match_group spanning two subsets is a silent-corruption risk."""
    dataset = pd.DataFrame(
        {
            "label": [1, 0, 1, 0],
            "subset": ["A", "B", "A", "A"],  # group 0 straddles A/B
            "match_group": [0, 0, 1, 1],
        }
    )
    scores = pd.DataFrame({"score": [0.9, 0.1, 0.8, 0.2]})
    with pytest.raises(AssertionError, match="span multiple subsets"):
        compute_pairwise_metrics(
            dataset=dataset, scores=scores, score_columns=["score"]
        )


def test_compute_pairwise_metrics_default_score_columns():
    dataset = pd.DataFrame(
        {
            "label": [1, 0],
            "subset": ["A", "A"],
            "match_group": [0, 0],
        }
    )
    scores = pd.DataFrame({"foo": [0.9, 0.1], "bar": [0.1, 0.9]})
    metrics = compute_pairwise_metrics(dataset=dataset, scores=scores, n_min=1)
    assert set(metrics["score_type"]) == {"foo", "bar"}


def test_compute_pairwise_metrics_requires_match_group_column():
    dataset = pd.DataFrame({"label": [1, 0], "subset": ["A", "A"]})
    scores = pd.DataFrame({"score": [0.9, 0.1]})
    with pytest.raises(AssertionError, match="match_group"):
        compute_pairwise_metrics(
            dataset=dataset, scores=scores, score_columns=["score"]
        )


# ---- aggregate rows (_global_ + _macro_avg_) ------------------------------


def _two_subset_dataset(big_n: int = 40, small_n: int = 4):
    """Synthetic 2-subset matched-pair frame with one large and one small
    subset. Subset A: all wins. Subset B: alternating wins/losses (n must be
    even for a clean 50/50 split; tests pass even n)."""
    label, score, mg, subset = [], [], [], []
    g = 0
    for _ in range(big_n):  # subset A — all wins
        label += [1, 0]
        score += [0.9, 0.1]
        mg += [g, g]
        subset += ["A", "A"]
        g += 1
    for i in range(small_n):  # subset B — alternating wins/losses
        win = i % 2 == 0
        label += [1, 0]
        score += [0.9, 0.1] if win else [0.1, 0.9]
        mg += [g, g]
        subset += ["B", "B"]
        g += 1
    return pd.DataFrame(
        {"label": label, "subset": subset, "match_group": mg}
    ), pd.DataFrame({"score": score})


def test_global_row_matches_direct_pairwise_accuracy():
    """The _global_ row must equal a direct pairwise_accuracy call on the
    whole merged frame — same value, same SE, same n_pairs, same n_ties."""
    dataset, scores = _two_subset_dataset(big_n=40, small_n=4)
    metrics = compute_pairwise_metrics(
        dataset=dataset, scores=scores, score_columns=["score"], n_min=30
    )
    direct = pairwise_accuracy(
        label=dataset["label"],
        score=scores["score"],
        match_group=dataset["match_group"],
    )
    g = metrics[metrics["subset"] == GLOBAL_SUBSET].iloc[0]
    assert g["value"] == pytest.approx(direct["value"])
    assert g["se"] == pytest.approx(direct["se"])
    assert int(g["n_pairs"]) == direct["n_pairs"]
    assert int(g["n_ties"]) == direct["n_ties"]


def test_macro_avg_unweighted_mean_when_all_subsets_qualify():
    """When every subset has n_pairs >= n_min, macro_avg == arithmetic mean of
    per-subset values (not n-weighted)."""
    dataset, scores = _two_subset_dataset(big_n=40, small_n=30)
    metrics = compute_pairwise_metrics(
        dataset=dataset, scores=scores, score_columns=["score"], n_min=30
    )
    per_subset = metrics[~metrics["subset"].str.startswith("_")]
    assert len(per_subset) == 2
    macro = metrics[metrics["subset"] == MACRO_AVG_SUBSET].iloc[0]
    assert macro["value"] == pytest.approx(per_subset["value"].mean())
    # SE = sqrt(Σ SE²) / K
    k = len(per_subset)
    expected_se = math.sqrt((per_subset["se"] ** 2).sum()) / k
    assert macro["se"] == pytest.approx(expected_se)
    assert int(macro["n_pairs"]) == k


def test_macro_avg_excludes_small_subset_but_global_includes_it():
    """A subset with n < n_min is dropped from macro but still feeds global."""
    dataset, scores = _two_subset_dataset(big_n=40, small_n=4)
    metrics = compute_pairwise_metrics(
        dataset=dataset, scores=scores, score_columns=["score"], n_min=30
    )
    macro = metrics[metrics["subset"] == MACRO_AVG_SUBSET].iloc[0]
    a_row = metrics[metrics["subset"] == "A"].iloc[0]
    # Only A qualifies (n=40); B (n=4) excluded.
    assert int(macro["n_pairs"]) == 1
    assert macro["value"] == pytest.approx(a_row["value"])
    assert macro["se"] == pytest.approx(a_row["se"])
    # Global covers all 44 pairs.
    g = metrics[metrics["subset"] == GLOBAL_SUBSET].iloc[0]
    assert int(g["n_pairs"]) == 44


def test_macro_avg_asserts_when_no_subset_qualifies():
    dataset = pd.DataFrame(
        {
            "label": [1, 0, 1, 0],
            "subset": ["A", "A", "B", "B"],
            "match_group": [0, 0, 1, 1],
        }
    )
    scores = pd.DataFrame({"score": [0.9, 0.1, 0.8, 0.2]})
    with pytest.raises(AssertionError, match="no subsets meet n_min"):
        compute_pairwise_metrics(
            dataset=dataset, scores=scores, score_columns=["score"], n_min=30
        )


def test_aggregates_emitted_per_score_column():
    """Each score_col gets its own _global_ and _macro_avg_ row."""
    dataset, _ = _two_subset_dataset(big_n=40, small_n=30)
    # Two score columns: one all-wins, one all-losses.
    n_total = len(dataset)
    scores = pd.DataFrame(
        {
            "all_wins": [
                0.9 if dataset["label"].iloc[i] == 1 else 0.1 for i in range(n_total)
            ],
            "all_losses": [
                0.1 if dataset["label"].iloc[i] == 1 else 0.9 for i in range(n_total)
            ],
        }
    )
    metrics = compute_pairwise_metrics(dataset=dataset, scores=scores, n_min=30)
    by_score_and_subset = metrics.set_index(["score_type", "subset"])["value"]
    assert by_score_and_subset[("all_wins", GLOBAL_SUBSET)] == 1.0
    assert by_score_and_subset[("all_wins", MACRO_AVG_SUBSET)] == 1.0
    assert by_score_and_subset[("all_losses", GLOBAL_SUBSET)] == 0.0
    assert by_score_and_subset[("all_losses", MACRO_AVG_SUBSET)] == 0.0


def test_global_n_pairs_equals_total_pairs():
    """Sanity: _global_'s n_pairs == sum of per-subset n_pairs."""
    dataset, scores = _two_subset_dataset(big_n=40, small_n=4)
    metrics = compute_pairwise_metrics(
        dataset=dataset, scores=scores, score_columns=["score"], n_min=2
    )
    per_subset = metrics[~metrics["subset"].str.startswith("_")]
    g = metrics[metrics["subset"] == GLOBAL_SUBSET].iloc[0]
    assert int(g["n_pairs"]) == int(per_subset["n_pairs"].sum())
