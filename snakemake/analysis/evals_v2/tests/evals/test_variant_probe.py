"""Tests for the frozen-embedding variant probe (issues #314/#320).

Covers the correctness guarantees the production rule relies on: ref↔alt symmetry
of the swap-invariant features, the f16→f32 upcast that guards the delta against
cancellation, leak-proof nested chromosome-grouped OOF, the C-edge guard, and the
per-subset orchestration (coverage, threshold skipping, no-subset path, reusable
classifier).
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from marin_dna_evals.variant_probe import (
    SYMMETRIC_COMBOS,
    fit_full_classifier,
    pair_feature,
    pair_feature_from_bundle,
    run_subset_probes,
    summarize_selected_c,
    traitgym_nested_oof,
)

# Small grid / fold count keep the nested-CV tests fast.
C_GRID_TEST = np.logspace(-2, 2, 5)  # [0.01, 0.1, 1, 10, 100]


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
    # concat_ref_delta = [ref, alt-ref]: the delta half negates under swap (signed).
    crd = pair_feature(ref, alt, "concat_ref_delta")
    crd_s = pair_feature(alt, ref, "concat_ref_delta")
    np.testing.assert_allclose(crd[:, 8:], -crd_s[:, 8:])
    assert "concat_ref_delta" not in SYMMETRIC_COMBOS


def test_pair_feature_shapes() -> None:
    ref, alt = np.zeros((5, 8)), np.ones((5, 8))
    assert pair_feature(ref, alt, "delta").shape == (5, 8)
    assert pair_feature(ref, alt, "abs_delta").shape == (5, 8)
    assert pair_feature(ref, alt, "prod").shape == (5, 8)
    assert pair_feature(ref, alt, "concat").shape == (5, 16)
    assert pair_feature(ref, alt, "sum_absdiff").shape == (5, 16)
    assert pair_feature(ref, alt, "concat_ref_delta").shape == (5, 16)


def test_pair_feature_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        pair_feature(np.zeros((2, 3)), np.zeros((2, 3)), "nope")


# --------------------------------------------------------------------------
# pair_feature_from_bundle — the in-bundle f16 path
# --------------------------------------------------------------------------


def _emb_frame(ref: np.ndarray, alt: np.ndarray, **cols) -> pd.DataFrame:
    """Frame with emb_ref/emb_alt stored as the #318 list[f16] columns."""
    data = {
        "emb_ref": list(ref.astype(np.float16)),
        "emb_alt": list(alt.astype(np.float16)),
    }
    data.update(cols)
    return pd.DataFrame(data)


def test_pair_feature_from_bundle_upcasts_f16_before_combining() -> None:
    # 1500/1501 are exact in f16, but their sum 3001 is NOT (f16 spacing is 2 above
    # 2048), so `ref+alt` computed in f16 rounds to 3000 while f32 keeps 3001. Using
    # `sum_absdiff` (which adds) makes the upcast observable — `delta` alone can't,
    # since f16 subtraction of near-equal values is exact (Sterbenz). A missing
    # pre-combine upcast would return f16 [[3000, 1]] and fail both asserts.
    ref = np.array([[1500.0]], dtype=np.float32)
    alt = np.array([[1501.0]], dtype=np.float32)
    feat = pair_feature_from_bundle(_emb_frame(ref, alt), "sum_absdiff")
    assert feat.dtype == np.float32
    np.testing.assert_array_equal(feat, [[3001.0, 1.0]])


def test_pair_feature_from_bundle_matches_manual_stack() -> None:
    rng = _rng(3)
    ref, alt = rng.standard_normal((6, 5)), rng.standard_normal((6, 5))
    df = _emb_frame(ref, alt)
    expected = pair_feature(
        np.stack(df["emb_ref"]).astype(np.float32),
        np.stack(df["emb_alt"]).astype(np.float32),
        "concat_ref_delta",
    )
    np.testing.assert_allclose(
        pair_feature_from_bundle(df, "concat_ref_delta"), expected
    )


def test_pair_feature_from_bundle_requires_emb_columns() -> None:
    with pytest.raises(AssertionError):
        pair_feature_from_bundle(pd.DataFrame({"chrom": ["1"], "label": [1]}), "delta")


# --------------------------------------------------------------------------
# Nested chromosome-grouped OOF
# --------------------------------------------------------------------------


def _toy_dataset(n_groups: int = 6, per_group: int = 60, signal: float = 3.0):
    rng = _rng(7)
    groups = np.repeat(np.arange(n_groups), per_group)
    label = rng.integers(0, 2, size=n_groups * per_group)
    feats = rng.standard_normal((n_groups * per_group, 6))
    feats[:, 0] += signal * label  # one informative dimension
    return feats, label, groups


def test_nested_oof_covers_rows_tunes_C_and_recovers_signal() -> None:
    feats, label, groups = _toy_dataset(signal=3.0)
    oof, selected = traitgym_nested_oof(
        feats, label, groups, c_grid=C_GRID_TEST, inner_splits=3, n_jobs=1
    )
    # every row held out exactly once; one selected C per outer (LOGO) fold
    assert oof.shape == label.shape and not np.isnan(oof).any()
    assert len(selected) == len(np.unique(groups))
    # tuned C is always drawn from the supplied grid
    assert all(any(np.isclose(c, g) for g in C_GRID_TEST) for c in selected)
    # nested CV still recovers the planted signal
    assert average_precision_score(label, oof) > 0.75


def test_nested_oof_no_group_leakage() -> None:
    # Features carry information ONLY via group identity (one-hot of group), labels
    # i.i.d. per group — a model that memorizes group->label in-fold cannot predict
    # a held-out (unseen) group: OOF AUPRC ~ prevalence.
    rng = _rng(11)
    n_groups, per_group = 8, 40
    groups = np.repeat(np.arange(n_groups), per_group)
    label = rng.integers(0, 2, size=n_groups * per_group)
    onehot = np.eye(n_groups)[groups]
    oof, _ = traitgym_nested_oof(
        onehot, label, groups, c_grid=C_GRID_TEST, inner_splits=3, n_jobs=1
    )
    ap = average_precision_score(label, oof)
    assert abs(ap - label.mean()) < 0.12, f"group identity leaked: AP={ap:.3f}"


def test_nested_oof_requires_three_groups() -> None:
    feats, label, _ = _toy_dataset()
    with pytest.raises(AssertionError):
        traitgym_nested_oof(feats, label, np.zeros(len(label)), n_jobs=1)


# --------------------------------------------------------------------------
# Full (reusable) classifier
# --------------------------------------------------------------------------


def test_fit_full_classifier_predicts_and_returns_grid_C_and_scores() -> None:
    feats, label, groups = _toy_dataset(signal=3.0)
    clf, c, c_scores = fit_full_classifier(
        feats, label, groups, c_grid=C_GRID_TEST, inner_splits=3, n_jobs=1
    )
    assert any(np.isclose(c, g) for g in C_GRID_TEST)
    proba = clf.predict_proba(feats[:5])
    assert proba.shape == (5, 2) and np.all((proba >= 0) & (proba <= 1))
    # one inner-CV AUPRC per grid C, ascending in C, finite
    assert c_scores.shape == C_GRID_TEST.shape and np.isfinite(c_scores).all()


# --------------------------------------------------------------------------
# C-edge guard
# --------------------------------------------------------------------------


def test_summarize_selected_c_interior_not_flagged() -> None:
    out = summarize_selected_c([1.0, 10.0, 1.0], full_c=10.0, c_grid=C_GRID_TEST)
    assert out["at_edge"] is False
    assert out["n_at_low_edge"] == 0 and out["n_at_high_edge"] == 0
    assert out["c_min"] == 1.0 and out["c_max"] == 10.0


def test_summarize_selected_c_low_edge_flagged() -> None:
    # two folds pin the 0.01 floor; the shipped full_c also pins it.
    out = summarize_selected_c([0.01, 0.01, 1.0], full_c=0.01, c_grid=C_GRID_TEST)
    assert out["at_edge"] is True
    assert out["n_at_low_edge"] == 2 and out["n_at_high_edge"] == 0
    assert out["full_at_edge"] is True


def test_summarize_selected_c_high_edge_via_full_c() -> None:
    # interior folds, but the all-data fit lands on the 100 ceiling.
    out = summarize_selected_c([1.0, 10.0], full_c=100.0, c_grid=C_GRID_TEST)
    assert out["at_edge"] is True and out["full_at_edge"] is True
    assert out["n_at_low_edge"] == 0 and out["n_at_high_edge"] == 0


def test_summarize_selected_c_high_edge_saturated_is_benign() -> None:
    # full_c pins the 100 cap, but the inner-CV curve is flat at the top (interior
    # neighbor gives the same AUPRC) → saturated, not a truncation.
    scores = np.array([0.30, 0.40, 0.50, 0.55, 0.55])  # ascending in C_GRID_TEST
    out = summarize_selected_c(
        [1.0, 100.0], full_c=100.0, c_grid=C_GRID_TEST, full_c_scores=scores
    )
    assert out["at_edge"] is True
    assert out["high_edge_gain"] == pytest.approx(0.0)
    assert out["truncation_risk"] is False


def test_summarize_selected_c_high_edge_truncation_flagged() -> None:
    # full_c pins the cap and the curve is STILL rising there → optimum may be beyond.
    scores = np.array([0.30, 0.40, 0.50, 0.55, 0.70])
    out = summarize_selected_c(
        [100.0], full_c=100.0, c_grid=C_GRID_TEST, full_c_scores=scores
    )
    assert out["high_edge_gain"] == pytest.approx(0.15)
    assert out["truncation_risk"] is True


def test_summarize_selected_c_low_edge_flat_is_benign() -> None:
    # folds pin the floor but the curve is flat there (the ncRNA case) → benign.
    scores = np.array([0.36, 0.37, 0.40, 0.30, 0.20])  # flat-then-peak-then-drop
    out = summarize_selected_c(
        [0.01, 0.01, 0.1], full_c=0.01, c_grid=C_GRID_TEST, full_c_scores=scores
    )
    assert out["n_at_low_edge"] == 2
    assert out["low_edge_gain"] == pytest.approx(0.36 - 0.37)  # <0 → not improving
    assert out["truncation_risk"] is False


def test_summarize_selected_c_low_edge_truncation_flagged() -> None:
    # full_c pins the floor and the curve is STILL higher there than its interior
    # neighbor (descending) → wants heavier reg than the grid allows → flagged.
    scores = np.array([0.60, 0.40, 0.30, 0.20, 0.10])  # best at the heavy-reg floor
    out = summarize_selected_c(
        [0.01], full_c=0.01, c_grid=C_GRID_TEST, full_c_scores=scores
    )
    assert out["low_edge_gain"] == pytest.approx(0.20)  # 0.60 - 0.40
    assert out["truncation_risk"] is True


def test_summarize_selected_c_nan_edge_gain_flags_risk() -> None:
    # A failed/single-class inner-CV fold leaves mean_test_score NaN at the pinned
    # edge; saturation can't be verified, so it must NOT be reported benign.
    scores = np.array([0.30, 0.40, 0.50, 0.55, np.nan])  # NaN at the pinned high edge
    out = summarize_selected_c(
        [100.0], full_c=100.0, c_grid=C_GRID_TEST, full_c_scores=scores
    )
    assert np.isnan(out["high_edge_gain"])
    assert out["truncation_risk"] is True  # not silently False via `nan > tol`


# --------------------------------------------------------------------------
# run_subset_probes — per-subset orchestration
# --------------------------------------------------------------------------


def _toy_bundle(
    subsets: dict[str, int],
    *,
    n_chrom: int = 4,
    d: int = 8,
    signal: float = 3.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Per-subset bundle frame; signal lives in the alt−ref delta (dim 0)."""
    rng = _rng(seed)
    frames = []
    for name, n in subsets.items():
        chrom = np.array([f"chr{i % n_chrom + 1}" for i in range(n)])
        label = rng.integers(0, 2, size=n)
        ref = rng.standard_normal((n, d)).astype(np.float32)
        delta = (rng.standard_normal((n, d)) * 0.1).astype(np.float32)
        delta[:, 0] += signal * label
        frames.append(
            _emb_frame(ref, ref + delta, chrom=chrom, label=label, subset=name)
        )
    return pd.concat(frames, ignore_index=True)


def test_run_subset_probes_covers_trained_subsets_and_drops_emb() -> None:
    df = _toy_bundle({"a": 80, "b": 80})
    preds, clfs = run_subset_probes(
        df,
        feature_combo="concat_ref_delta",
        c_grid=C_GRID_TEST,
        min_variants=40,
        min_chroms=3,
        inner_splits=3,
        n_jobs=1,
    )
    assert set(clfs) == {"a", "b"}
    # predictions carry one row per variant, probe_score everywhere, no emb cols.
    assert len(preds) == len(df)
    assert "probe_score" in preds.columns
    assert "emb_ref" not in preds.columns and "emb_alt" not in preds.columns
    assert not preds["probe_score"].isna().any()
    # the planted signal is recovered out-of-fold.
    assert average_precision_score(preds["label"], preds["probe_score"]) > 0.75
    # each classifier records provenance + is reusable.
    for entry in clfs.values():
        assert entry["feature"] == "concat_ref_delta"
        assert any(np.isclose(entry["C"], g) for g in C_GRID_TEST)
        assert "c_summary" in entry


def test_run_subset_probes_skips_below_threshold_subset() -> None:
    df = _toy_bundle({"big": 80, "small": 20})
    preds, clfs = run_subset_probes(
        df,
        feature_combo="concat_ref_delta",
        c_grid=C_GRID_TEST,
        min_variants=40,
        min_chroms=3,
        inner_splits=3,
        n_jobs=1,
    )
    assert set(clfs) == {"big"}  # small (n=20 < 40) dropped
    small = preds["subset"] == "small"
    assert preds.loc[small, "probe_score"].isna().all()
    assert not preds.loc[~small, "probe_score"].isna().any()


def test_run_subset_probes_no_subset_column_uses_all_group() -> None:
    df = _toy_bundle({"only": 80}).drop(columns="subset")
    preds, clfs = run_subset_probes(
        df,
        feature_combo="concat_ref_delta",
        c_grid=C_GRID_TEST,
        min_variants=40,
        min_chroms=3,
        inner_splits=3,
        n_jobs=1,
    )
    assert set(clfs) == {"all"}
    assert not preds["probe_score"].isna().any()


def test_run_subset_probes_reusable_classifier_predicts_other_data() -> None:
    train = _toy_bundle({"a": 80}, seed=0)
    _preds, clfs = run_subset_probes(
        train,
        feature_combo="concat_ref_delta",
        c_grid=C_GRID_TEST,
        min_variants=40,
        min_chroms=3,
        inner_splits=3,
        n_jobs=1,
    )
    # apply the saved classifier to a fresh bundle (cross-dataset reuse).
    fresh = _toy_bundle({"a": 30}, seed=99)
    feat_fresh = pair_feature_from_bundle(fresh, "concat_ref_delta")
    proba = clfs["a"]["pipeline"].predict_proba(feat_fresh)
    assert proba.shape == (30, 2) and np.all((proba >= 0) & (proba <= 1))


def test_run_subset_probes_all_below_threshold_raises() -> None:
    df = _toy_bundle({"a": 20})
    with pytest.raises(AssertionError):
        run_subset_probes(
            df,
            feature_combo="concat_ref_delta",
            c_grid=C_GRID_TEST,
            min_variants=1000,
            n_jobs=1,
        )


def test_run_subset_probes_skips_single_class_subset() -> None:
    # a subset that clears the size/chrom gates but is all-one-class must be skipped
    # (LogisticRegression needs both classes), not crash the run.
    df = _toy_bundle({"normal": 80, "allneg": 80})
    df.loc[df["subset"] == "allneg", "label"] = 0  # force single-class
    preds, clfs = run_subset_probes(
        df,
        feature_combo="concat_ref_delta",
        c_grid=C_GRID_TEST,
        min_variants=40,
        min_chroms=3,
        inner_splits=3,
        n_jobs=1,
    )
    assert set(clfs) == {"normal"}
    assert preds.loc[preds["subset"] == "allneg", "probe_score"].isna().all()


def test_run_subset_probes_skips_below_min_chroms() -> None:
    # plenty of variants but only 2 chromosomes (< min_chroms) → skipped, not fed to
    # LeaveOneGroupOut (which needs >=3 groups).
    wide = _toy_bundle({"wide": 80}, n_chrom=4)
    narrow = _toy_bundle({"narrow": 80}, n_chrom=2, seed=1)
    df = pd.concat([wide, narrow], ignore_index=True)
    preds, clfs = run_subset_probes(
        df,
        feature_combo="concat_ref_delta",
        c_grid=C_GRID_TEST,
        min_variants=40,
        min_chroms=3,
        inner_splits=3,
        n_jobs=1,
    )
    assert set(clfs) == {"wide"}
    assert preds.loc[preds["subset"] == "narrow", "probe_score"].isna().all()


def test_run_subset_probes_keeps_going_when_one_subset_errors(monkeypatch) -> None:
    # the per-subset try/except must keep an unattended multi-subset run alive: one
    # erroring subset is skipped (NaN), siblings still train.
    from marin_dna_evals import variant_probe as vp

    df = _toy_bundle({"aaa_bad": 80, "zzz_good": 80})  # sorted() → aaa_bad first
    real = vp.traitgym_nested_oof
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:  # only the first subset errors
            raise RuntimeError("boom")
        return real(*args, **kwargs)

    monkeypatch.setattr(vp, "traitgym_nested_oof", flaky)
    preds, clfs = vp.run_subset_probes(
        df,
        feature_combo="concat_ref_delta",
        c_grid=C_GRID_TEST,
        min_variants=40,
        min_chroms=3,
        inner_splits=3,
        n_jobs=1,
    )
    assert set(clfs) == {"zzz_good"}
    assert preds.loc[preds["subset"] == "aaa_bad", "probe_score"].isna().all()
    assert not preds.loc[preds["subset"] == "zzz_good", "probe_score"].isna().any()
