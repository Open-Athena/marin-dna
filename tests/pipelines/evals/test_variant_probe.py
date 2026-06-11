"""Tests for the frozen-embedding variant probe (issue #314).

Covers the correctness guarantees the investigation relies on: ref↔alt
symmetry of the swap-invariant features (required for complex/qtl + transfer),
pooling reductions, leak-proof chromosome-grouped OOF coverage, and the AUPRC
wiring.
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score

from marin_dna.pipelines.evals.variant_probe import (
    SYMMETRIC_COMBOS,
    chrom_grouped_oof,
    cov_delta_feature,
    innerprod_feature,
    make_linear_probe,
    pair_feature,
    pool_tokens,
    probe_auprc,
    random_projection,
)


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------
# Feature construction + symmetry
# --------------------------------------------------------------------------


@pytest.mark.parametrize("combo", ["abs_delta", "prod", "sum_absdiff"])
def test_symmetric_combos_invariant_under_swap(combo: str) -> None:
    rng = _rng()
    ref, alt = rng.standard_normal((20, 8)), rng.standard_normal((20, 8))
    assert combo in SYMMETRIC_COMBOS
    np.testing.assert_allclose(
        pair_feature(ref, alt, combo), pair_feature(alt, ref, combo)
    )


def test_signed_combos_change_under_swap() -> None:
    rng = _rng()
    ref, alt = rng.standard_normal((20, 8)), rng.standard_normal((20, 8))
    # delta negates; concat swaps its two halves.
    np.testing.assert_allclose(
        pair_feature(ref, alt, "delta"), -pair_feature(alt, ref, "delta")
    )
    swapped = pair_feature(alt, ref, "concat")
    np.testing.assert_allclose(pair_feature(ref, alt, "concat")[:, :8], swapped[:, 8:])


def test_pair_feature_shapes() -> None:
    ref, alt = np.zeros((5, 8)), np.ones((5, 8))
    assert pair_feature(ref, alt, "delta").shape == (5, 8)
    assert pair_feature(ref, alt, "abs_delta").shape == (5, 8)
    assert pair_feature(ref, alt, "prod").shape == (5, 8)
    assert pair_feature(ref, alt, "concat").shape == (5, 16)
    assert pair_feature(ref, alt, "sum_absdiff").shape == (5, 16)


def test_pair_feature_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        pair_feature(np.zeros((2, 3)), np.zeros((2, 3)), "nope")


# --------------------------------------------------------------------------
# Pooling
# --------------------------------------------------------------------------


def test_pool_tokens_extents() -> None:
    # states[n, l, d] = l (broadcast over d) so pooled values are predictable.
    length, d = 10, 4
    states = np.tile(np.arange(length, dtype=float)[None, :, None], (3, 1, d))
    np.testing.assert_allclose(pool_tokens(states, "entire_window"), 4.5)  # mean 0..9
    np.testing.assert_allclose(pool_tokens(states, "max"), 9.0)
    np.testing.assert_allclose(pool_tokens(states, "variant_token", var_index=3), 3.0)
    # center 4 of 10 -> positions [3,4,5,6], mean 4.5
    np.testing.assert_allclose(pool_tokens(states, "center", n_center=4), 4.5)
    assert pool_tokens(states, "entire_window").shape == (3, d)


def test_pool_center_must_fit() -> None:
    states = np.zeros((2, 5, 3))
    with pytest.raises(AssertionError):
        pool_tokens(states, "center", n_center=6)


def test_pool_tokens_returns_copies_not_views() -> None:
    # Pooled features accumulate across shards; a view would pin the whole parent
    # array and leak GBs (the #314 OOM). Every extent must return a fresh array.
    states = np.zeros((3, 10, 4))
    for extent, kw in [
        ("entire_window", {}),
        ("max", {}),
        ("center", {"n_center": 4}),
        ("variant_token", {"var_index": 3}),
    ]:
        out = pool_tokens(states, extent, **kw)
        assert not np.shares_memory(out, states), f"{extent} returns a view"


# --------------------------------------------------------------------------
# Per-token features (innerprod, cov_delta)
# --------------------------------------------------------------------------


def test_innerprod_value_and_symmetry() -> None:
    rng = _rng(1)
    ref, alt = rng.standard_normal((6, 7, 5)), rng.standard_normal((6, 7, 5))
    ip = innerprod_feature(ref, alt)
    assert ip.shape == (6, 5)
    np.testing.assert_allclose(ip, (ref * alt).sum(axis=1))
    np.testing.assert_allclose(ip, innerprod_feature(alt, ref))  # symmetric


def test_cov_delta_symmetry_and_shape() -> None:
    rng = _rng(2)
    ref, alt = rng.standard_normal((6, 7, 12)), rng.standard_normal((6, 7, 12))
    proj = random_projection(12, 4, seed=3)
    cov = cov_delta_feature(ref, alt, proj)
    assert cov.shape == (6, 16)  # r^2 with r=4
    # Gram of (alt-ref) == Gram of (ref-alt): symmetric under swap.
    np.testing.assert_allclose(cov, cov_delta_feature(alt, ref, proj), atol=1e-5)


# --------------------------------------------------------------------------
# Probe pipeline
# --------------------------------------------------------------------------


def test_make_linear_probe_toggles() -> None:
    lr = make_linear_probe(loss="logistic", c=0.5, n_pca=4, standardize=True)
    assert [s[0] for s in lr.steps] == ["scaler", "pca", "clf"]
    assert isinstance(lr.steps[-1][1], LogisticRegression)
    assert hasattr(lr, "predict_proba")

    rr = make_linear_probe(loss="ridge", c=2.0, n_pca=None, standardize=False)
    assert [s[0] for s in rr.steps] == ["clf"]
    assert isinstance(rr.steps[-1][1], Ridge)
    assert rr.steps[-1][1].alpha == 0.5  # alpha = 1/c
    assert not hasattr(rr, "predict_proba")  # ridge -> ranking score, no proba


def test_make_linear_probe_rejects_unknown_loss() -> None:
    with pytest.raises(ValueError):
        make_linear_probe(loss="svm")


# --------------------------------------------------------------------------
# Chromosome-grouped OOF CV
# --------------------------------------------------------------------------


def _toy_dataset(n_groups: int = 6, per_group: int = 50, signal: float = 3.0):
    rng = _rng(7)
    groups = np.repeat(np.arange(n_groups), per_group)
    label = rng.integers(0, 2, size=n_groups * per_group)
    feats = rng.standard_normal((n_groups * per_group, 6))
    feats[:, 0] += signal * label  # one informative dimension
    return feats, label, groups


def test_oof_covers_every_row_once() -> None:
    feats, label, groups = _toy_dataset()
    oof = chrom_grouped_oof(feats, label, groups, loss="logistic", standardize=True)
    assert oof.shape == label.shape
    assert not np.isnan(oof).any()


def test_oof_recovers_learnable_signal() -> None:
    feats, label, groups = _toy_dataset(signal=3.0)
    oof = chrom_grouped_oof(feats, label, groups, loss="logistic", standardize=True)
    assert average_precision_score(label, oof) > 0.75  # prevalence ~0.5


def test_oof_no_group_leakage() -> None:
    # Features carry information ONLY via group identity (one-hot of group), with
    # each group's label drawn i.i.d. — so a model that memorizes group->label
    # in-fold cannot predict a held-out (unseen) group: OOF AUPRC ~ chance.
    rng = _rng(11)
    n_groups, per_group = 8, 40
    groups = np.repeat(np.arange(n_groups), per_group)
    label = rng.integers(0, 2, size=n_groups * per_group)
    onehot = np.eye(n_groups)[groups]
    oof = chrom_grouped_oof(onehot, label, groups, loss="logistic", standardize=False)
    ap = average_precision_score(label, oof)
    prevalence = label.mean()
    assert abs(ap - prevalence) < 0.12, f"group identity leaked: AP={ap:.3f}"


def test_oof_ridge_runs() -> None:
    feats, label, groups = _toy_dataset()
    oof = chrom_grouped_oof(feats, label, groups, loss="ridge", standardize=True)
    assert not np.isnan(oof).any()
    assert average_precision_score(label, oof) > 0.75


def test_oof_requires_two_groups() -> None:
    feats, label, _ = _toy_dataset()
    with pytest.raises(AssertionError):
        chrom_grouped_oof(feats, label, np.zeros(len(label)))


# --------------------------------------------------------------------------
# AUPRC scoring
# --------------------------------------------------------------------------


def test_probe_auprc_perfect_separation() -> None:
    label = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.8, 0.9])
    mg = np.array([0, 0, 1, 1])
    out = probe_auprc(label, score, mg, n_bootstrap=0)
    assert out["value"] == pytest.approx(1.0)
    assert out["n_rows"] == 4
