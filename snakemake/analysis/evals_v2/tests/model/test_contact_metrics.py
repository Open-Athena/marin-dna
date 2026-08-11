"""Tests for ``marin_dna_evals.model.contact_metrics`` (Fig-7b contact prediction)."""

import numpy as np
import pytest

from marin_dna_evals.model.contact_metrics import (
    base_pairs_to_contact_matrix,
    contact_prediction_metrics,
    parse_trnascan_ss,
)

# tRNA-Arg-TCT-4-1 (gtRNAdb GRCh38) tRNAscan-SE nested-bp structure, 74 nt.
TRNA_SS = ">>>>>>>..>>>>.........<<<<.>>>>>.......<<<<<.....>>>>>.......<<<<<<<<<<<<."


def test_parse_trnascan_ss_trna_cloverleaf():
    pairs = parse_trnascan_ss(TRNA_SS)
    assert len(TRNA_SS) == 74
    # 7 acceptor + 4 D-arm + 5 anticodon + 5 T-arm = 21 base pairs.
    assert len(pairs) == 21
    # Outermost acceptor-stem pair: position 0 pairs with position 72.
    assert (0, 72) in pairs
    # All pairs are ordered and in range.
    assert all(0 <= i < j < 74 for i, j in pairs)


def test_parse_trnascan_ss_small_and_unbalanced():
    assert parse_trnascan_ss(">>..<<") == [(0, 5), (1, 4)]
    assert parse_trnascan_ss("") == []
    with pytest.raises(AssertionError):
        parse_trnascan_ss(">>.<")  # one unclosed '>'
    with pytest.raises(AssertionError):
        parse_trnascan_ss(">.<<")  # stray '<'


def test_base_pairs_to_contact_matrix_symmetric():
    c = base_pairs_to_contact_matrix([(0, 4), (1, 3)], 5)
    assert c.shape == (5, 5)
    assert c[0, 4] and c[4, 0] and c[1, 3] and c[3, 1]
    assert np.array_equal(c, c.T)
    assert c.sum() == 4  # two pairs, each counted twice


def test_contact_metrics_perfect_ranking():
    """A dependency map equal to the contact matrix ranks true pairs first →
    AUROC = AUPRC = PPV = MCC = 1."""
    pairs = [(0, 9), (1, 8), (2, 7)]
    n = 10
    dep = base_pairs_to_contact_matrix(pairs, n).astype(float)
    # add a tiny diagonal-decaying background so non-contacts aren't all tied
    dep = dep + 1e-3 * np.maximum(
        0, 1 - np.abs(np.subtract.outer(range(n), range(n))) / n
    )
    m = contact_prediction_metrics(dep, pairs, min_sep=1)
    assert m["n_true_pairs"] == 3
    assert m["auroc"] == pytest.approx(1.0)
    assert m["auprc"] == pytest.approx(1.0)
    assert m["ppv"] == pytest.approx(1.0)
    assert m["mcc"] == pytest.approx(1.0)


def test_contact_metrics_random_is_chance():
    rng = np.random.RandomState(0)
    n = 40
    pairs = [(i, n - 1 - i) for i in range(8)]  # 8 long-range pairs
    A = rng.rand(n, n)
    dep = (A + A.T) / 2
    m = contact_prediction_metrics(dep, pairs, min_sep=1)
    # A random map has no signal → AUROC near chance.
    assert 0.3 < m["auroc"] < 0.7
    assert m["n_true_pairs"] == 8


def test_contact_metrics_min_sep_excludes_near_diagonal():
    pairs = [(0, 1), (0, 9)]  # one adjacent, one long-range
    n = 10
    dep = base_pairs_to_contact_matrix(pairs, n).astype(float)
    # min_sep=3 drops the (0,1) adjacent pair from the candidate set.
    m = contact_prediction_metrics(dep, pairs, min_sep=3)
    assert m["n_true_pairs"] == 1  # only (0,9) survives the separation filter
