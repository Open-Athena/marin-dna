"""Tests for the classical (AUPRC/AUROC/Spearman) metrics API and cross-run
aggregation, plus the cluster-bootstrap AUPRC used by ``evals_v2``.
Per-metric tests for ``pairwise_accuracy`` live in
``test_pairwise_accuracy.py``."""

import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from marin_dna.pipelines.evals.metrics import (
    GLOBAL_SUBSET,
    MACRO_AVG_SUBSET,
    METRIC_FUNCTIONS,
    SGE_POOLED_SUBSET,
    aggregate_metrics,
    auprc_with_bootstrap_se,
    compute_auprc_metrics,
    compute_metrics,
    compute_qtl_metrics,
    compute_sge_metrics,
    compute_sge_probe_metrics,
    paired_metric_delta_bootstrap,
    per_chrom_ap_table,
    per_chrom_weighted_ap,
)


def test_metric_functions_auprc():
    labels = pd.Series([1, 1, 0, 0, 1])
    scores = pd.Series([0.9, 0.8, 0.6, 0.3, 0.7])
    auprc = METRIC_FUNCTIONS["AUPRC"](labels, scores)
    assert 0.0 <= auprc <= 1.0
    assert isinstance(auprc, float)


def test_metric_functions_auroc():
    labels = pd.Series([1, 1, 0, 0, 1])
    scores = pd.Series([0.9, 0.8, 0.6, 0.3, 0.7])
    auroc = METRIC_FUNCTIONS["AUROC"](labels, scores)
    assert 0.0 <= auroc <= 1.0
    assert isinstance(auroc, float)


def test_metric_functions_spearman():
    labels = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    scores = pd.Series([1.1, 2.2, 2.9, 4.1, 5.0])
    spearman = METRIC_FUNCTIONS["Spearman"](labels, scores)
    assert -1.0 <= spearman <= 1.0
    assert isinstance(spearman, float)


def test_compute_metrics_without_subsets():
    dataset = pd.DataFrame(
        {
            "chrom": ["chr1"] * 5,
            "pos": [100, 200, 300, 400, 500],
            "ref": ["A", "C", "G", "T", "A"],
            "alt": ["T", "G", "A", "C", "G"],
            "label": [1, 1, 0, 0, 1],
        }
    )
    scores = pd.DataFrame(
        {
            "minus_llr": [0.9, 0.8, 0.6, 0.3, 0.7],
            "abs_llr": [0.9, 0.8, 0.6, 0.3, 0.7],
        }
    )
    metrics = compute_metrics(
        dataset=dataset,
        scores=scores,
        metrics=["AUPRC", "AUROC"],
        score_columns=["minus_llr", "abs_llr"],
    )
    # 2 metrics * 2 score types = 4 rows
    assert len(metrics) == 4
    assert set(metrics["metric"]) == {"AUPRC", "AUROC"}
    assert set(metrics["score_type"]) == {"minus_llr", "abs_llr"}
    assert set(metrics["subset"]) == {"global"}
    assert all(metrics["value"].notna())


def test_compute_metrics_with_subsets():
    dataset = pd.DataFrame(
        {
            "chrom": ["chr1"] * 6,
            "pos": [100, 200, 300, 400, 500, 600],
            "ref": ["A", "C", "G", "T", "A", "C"],
            "alt": ["T", "G", "A", "C", "G", "T"],
            "label": [1, 1, 0, 0, 1, 0],
            "subset": ["5UTR", "5UTR", "3UTR", "3UTR", "5UTR", "3UTR"],
        }
    )
    scores = pd.DataFrame(
        {
            "minus_llr": [0.9, 0.8, 0.6, 0.3, 0.7, 0.4],
            "abs_llr": [0.9, 0.8, 0.6, 0.3, 0.7, 0.4],
        }
    )
    metrics = compute_metrics(
        dataset=dataset,
        scores=scores,
        metrics=["AUPRC"],
        score_columns=["minus_llr"],
    )
    # 1 metric * 1 score type * 3 subsets (global, 5UTR, 3UTR) = 3 rows
    assert len(metrics) == 3
    assert set(metrics["subset"]) == {"global", "5UTR", "3UTR"}


def test_compute_metrics_default_score_columns():
    dataset = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "pos": [100],
            "ref": ["A"],
            "alt": ["T"],
            "label": [1],
        }
    )
    scores = pd.DataFrame({"minus_llr": [0.9], "abs_llr": [0.9]})
    metrics = compute_metrics(dataset=dataset, scores=scores, metrics=["AUPRC"])
    assert set(metrics["score_type"]) == {"minus_llr", "abs_llr"}


def test_aggregate_metrics():
    """Aggregating metrics from multiple files annotates each row with
    ``step`` and ``dataset`` and concatenates."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        m1 = pd.DataFrame(
            {
                "score_type": ["score", "score"],
                "subset": ["A", "B"],
                "value": [0.9, 0.7],
                "se": [0.05, 0.10],
                "n_pairs": [100, 50],
                "n_ties": [0, 2],
            }
        )
        m1.to_parquet(tmpdir / "m1.parquet")

        m2 = pd.DataFrame(
            {
                "score_type": ["score", "score"],
                "subset": ["A", "B"],
                "value": [0.85, 0.65],
                "se": [0.06, 0.11],
                "n_pairs": [100, 50],
                "n_ties": [1, 2],
            }
        )
        m2.to_parquet(tmpdir / "m2.parquet")

        result = aggregate_metrics(
            metric_files=[str(tmpdir / "m1.parquet"), str(tmpdir / "m2.parquet")],
            dataset_names=["dataset1", "dataset1"],
            model_steps=["10000", "20000"],
        )

        assert len(result) == 4
        assert set(result["step"]) == {10000, 20000}
        assert set(result["dataset"]) == {"dataset1"}
        assert set(result["subset"]) == {"A", "B"}
        assert all(result["value"].notna())


# ---------------------------------------------------------------------------
# auprc_with_bootstrap_se
# ---------------------------------------------------------------------------


def _matched_pairs(
    n_pos: int = 50, k: int = 9, separable: bool = True, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthesize matched-pair data: n_pos positives, each with k matched negatives.
    If ``separable=True``, all positives score above all negatives → AUPRC=1.0."""
    rng = np.random.default_rng(seed)
    labels = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_pos * k, dtype=int)])
    match_group = np.concatenate([np.arange(n_pos), np.repeat(np.arange(n_pos), k)])
    if separable:
        scores = np.concatenate(
            [rng.uniform(0.7, 1.0, n_pos), rng.uniform(0.0, 0.3, n_pos * k)]
        )
    else:
        scores = rng.uniform(0.0, 1.0, n_pos * (k + 1))
    return labels, scores, match_group


def test_auprc_perfectly_separable_returns_one():
    labels, scores, mg = _matched_pairs(separable=True)
    res = auprc_with_bootstrap_se(labels, scores, mg, n_bootstrap=100, rng=0)
    assert res["value"] == pytest.approx(1.0, abs=1e-12)
    # Bootstrap samples from perfectly separable data all give AUPRC=1.0 →
    # zero variance.
    assert res["se"] == pytest.approx(0.0, abs=1e-12)
    assert res["n_groups"] == 50
    assert res["n_rows"] == 50 * 10


def test_auprc_random_scores_near_baseline():
    """Random scores → AUPRC near the 1:9 positive prevalence baseline (0.1)."""
    labels, scores, mg = _matched_pairs(separable=False, seed=42)
    res = auprc_with_bootstrap_se(labels, scores, mg, n_bootstrap=200, rng=0)
    # Point estimate should land within a few SDs of the 0.1 baseline.
    assert 0.05 < res["value"] < 0.25
    # SE should be non-trivial.
    assert res["se"] > 0


def test_auprc_n_bootstrap_zero_is_point_only():
    """n_bootstrap=0 → skip the resample loop: identical point AUPRC, se=NaN,
    and n_groups still computed (it feeds the macro-avg n_min gate). This is the
    online in-training hot path (SE is computed offline instead)."""
    labels, scores, mg = _matched_pairs(separable=False, seed=42)
    full = auprc_with_bootstrap_se(labels, scores, mg, n_bootstrap=200, rng=0)
    point = auprc_with_bootstrap_se(labels, scores, mg, n_bootstrap=0)
    assert point["value"] == pytest.approx(full["value"])  # data-only point est.
    assert np.isnan(point["se"])
    assert point["n_groups"] == full["n_groups"]
    assert point["n_rows"] == full["n_rows"]


def test_auprc_seed_reproducibility():
    """Same seed → identical SE; different seeds → SE differs (sanity)."""
    labels, scores, mg = _matched_pairs(separable=False, seed=1)
    a = auprc_with_bootstrap_se(labels, scores, mg, n_bootstrap=100, rng=0)
    b = auprc_with_bootstrap_se(labels, scores, mg, n_bootstrap=100, rng=0)
    c = auprc_with_bootstrap_se(labels, scores, mg, n_bootstrap=100, rng=1)
    assert a["se"] == b["se"]
    assert a["se"] != c["se"]
    # Point estimate is data-only, so it should be identical across seeds.
    assert a["value"] == c["value"]


def test_auprc_global_matches_sklearn():
    """Point-estimate value equals sklearn's average_precision_score over all rows."""
    labels, scores, mg = _matched_pairs(separable=False, seed=7)
    res = auprc_with_bootstrap_se(labels, scores, mg, n_bootstrap=10, rng=0)
    expected = float(average_precision_score(labels, scores))
    assert res["value"] == pytest.approx(expected)


def test_auprc_nan_score_raises():
    labels, scores, mg = _matched_pairs(separable=False)
    scores = scores.copy()
    scores[3] = np.nan
    with pytest.raises(AssertionError, match="NaN"):
        auprc_with_bootstrap_se(labels, scores, mg, n_bootstrap=10, rng=0)


def test_auprc_single_class_raises():
    labels = np.ones(20, dtype=int)
    scores = np.linspace(0, 1, 20)
    mg = np.arange(20)
    with pytest.raises(AssertionError, match="both classes"):
        auprc_with_bootstrap_se(labels, scores, mg, n_bootstrap=10, rng=0)


def test_auprc_length_mismatch_raises():
    labels = np.array([1, 0, 1])
    scores = np.array([0.5, 0.5])
    mg = np.array([0, 1, 2])
    with pytest.raises(AssertionError, match="length mismatch"):
        auprc_with_bootstrap_se(labels, scores, mg, n_bootstrap=10, rng=0)


# ---------------------------------------------------------------------------
# paired_metric_delta_bootstrap
# ---------------------------------------------------------------------------


def test_paired_delta_zero_for_identical_scores():
    """A == B → delta exactly 0, SE 0, CI brackets 0, p≈1 (ties on both sides)."""
    labels, scores, mg = _matched_pairs(separable=False, seed=5)
    r = paired_metric_delta_bootstrap(
        labels, scores, scores, mg, n_bootstrap=200, rng=0
    )
    assert r["delta"] == pytest.approx(0.0, abs=1e-12)
    assert r["se"] == pytest.approx(0.0, abs=1e-12)
    assert r["ci_low"] <= 0 <= r["ci_high"]
    assert r["p_two_sided"] == pytest.approx(1.0)


def test_paired_delta_positive_and_significant_when_a_better():
    """A ranks positives above negatives, B random → delta>0, CI excludes 0, small p."""
    labels, scores_b, mg = _matched_pairs(separable=False, seed=1)
    scores_a = np.where(labels == 1, 0.9, 0.1)
    r = paired_metric_delta_bootstrap(
        labels, scores_a, scores_b, mg, n_bootstrap=300, rng=0
    )
    assert r["delta"] > 0
    assert r["ci_low"] > 0  # CI excludes 0
    assert r["p_two_sided"] < 0.05


def test_paired_se_tighter_than_independence():
    """The whole point: on shared rows the paired SE is below sqrt(SE_a² + SE_b²)."""
    labels, scores_b, mg = _matched_pairs(separable=False, seed=3)
    scores_a = scores_b.copy()
    scores_a[labels == 1] += (
        0.12  # A = B with positives nudged up → correlated, A better
    )
    se_a = auprc_with_bootstrap_se(labels, scores_a, mg, n_bootstrap=400, rng=0)["se"]
    se_b = auprc_with_bootstrap_se(labels, scores_b, mg, n_bootstrap=400, rng=0)["se"]
    indep = float(np.hypot(se_a, se_b))
    r = paired_metric_delta_bootstrap(
        labels, scores_a, scores_b, mg, n_bootstrap=400, rng=0
    )
    assert r["delta"] > 0
    assert 0 < r["se"] < indep  # paired strictly tighter than the independence formula


def test_paired_delta_seed_reproducibility():
    """rng=0 twice → identical delta and SE (bit-stable for diffing claims)."""
    labels, scores_b, mg = _matched_pairs(separable=False, seed=2)
    scores_a = scores_b.copy()
    scores_a[labels == 1] += 0.1
    a = paired_metric_delta_bootstrap(
        labels, scores_a, scores_b, mg, n_bootstrap=100, rng=0
    )
    b = paired_metric_delta_bootstrap(
        labels, scores_a, scores_b, mg, n_bootstrap=100, rng=0
    )
    assert a["se"] == b["se"]
    assert a["delta"] == b["delta"]


# ---------------------------------------------------------------------------
# compute_auprc_metrics
# ---------------------------------------------------------------------------


def _matched_pairs_with_subsets(
    subsets: list[str] = ["A", "B"],
    n_pos_per_subset: int = 40,
    k: int = 9,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthesize a matched-pair dataset with multiple subsets. Each subset
    gets its own block of match_groups; no group spans subsets."""
    rng = np.random.default_rng(seed)
    parts_ds = []
    parts_score = []
    base_group = 0
    for s in subsets:
        labels, scores, mg = _matched_pairs(
            n_pos=n_pos_per_subset, k=k, separable=False, seed=seed + hash(s) % 1000
        )
        mg = mg + base_group
        base_group = int(mg.max()) + 1
        parts_ds.append(pd.DataFrame({"label": labels, "subset": s, "match_group": mg}))
        parts_score.append(
            pd.DataFrame({"score": scores, "score2": rng.uniform(size=len(scores))})
        )
    return pd.concat(parts_ds, ignore_index=True), pd.concat(
        parts_score, ignore_index=True
    )


def test_compute_auprc_metrics_shape():
    """Output: one row per (score_column × subset) plus global + macro per score."""
    dataset, scores = _matched_pairs_with_subsets(
        subsets=["A", "B", "C"], n_pos_per_subset=40
    )
    metrics = compute_auprc_metrics(
        dataset=dataset, scores=scores, n_bootstrap=20, rng=0
    )
    # 3 subsets * 2 score cols + 2 aggregates * 2 score cols = 10 rows
    assert len(metrics) == 3 * 2 + 2 * 2
    assert set(metrics["score_type"]) == {"score", "score2"}
    assert set(metrics["subset"]) == {"A", "B", "C", GLOBAL_SUBSET, MACRO_AVG_SUBSET}
    assert set(metrics.columns) == {
        "score_type",
        "subset",
        "value",
        "se",
        "n_groups",
        "n_rows",
    }


def test_compute_auprc_metrics_global_matches_sklearn():
    """``_global_`` row's value equals sklearn's AUPRC over all rows for that score."""
    dataset, scores = _matched_pairs_with_subsets(subsets=["A", "B"])
    metrics = compute_auprc_metrics(
        dataset=dataset,
        scores=scores,
        score_columns=["score"],
        n_bootstrap=10,
        rng=0,
    )
    global_row = metrics[
        (metrics["score_type"] == "score") & (metrics["subset"] == GLOBAL_SUBSET)
    ].iloc[0]
    expected = float(average_precision_score(dataset["label"], scores["score"]))
    assert global_row["value"] == pytest.approx(expected)
    assert global_row["n_rows"] == len(dataset)


def test_compute_auprc_metrics_macro_avg_matches_mean_of_qualifying():
    """``_macro_avg_`` row's value equals the unweighted mean of per-subset
    values for subsets meeting ``n_min``."""
    dataset, scores = _matched_pairs_with_subsets(
        subsets=["A", "B"], n_pos_per_subset=40
    )
    metrics = compute_auprc_metrics(
        dataset=dataset,
        scores=scores,
        score_columns=["score"],
        n_bootstrap=10,
        rng=0,
        n_min=30,
    )
    per_subset = metrics[
        (metrics["score_type"] == "score")
        & (~metrics["subset"].isin({GLOBAL_SUBSET, MACRO_AVG_SUBSET}))
    ]
    macro_row = metrics[
        (metrics["score_type"] == "score") & (metrics["subset"] == MACRO_AVG_SUBSET)
    ].iloc[0]
    assert macro_row["value"] == pytest.approx(per_subset["value"].mean())
    assert macro_row["n_groups"] == len(per_subset)


def test_compute_auprc_metrics_match_group_straddle_raises():
    """A match_group present in more than one subset → AssertionError."""
    dataset, scores = _matched_pairs_with_subsets(subsets=["A", "B"])
    dataset.loc[0, "subset"] = "B"  # group 0's positive now lives in subset B
    with pytest.raises(AssertionError, match="span multiple subsets"):
        compute_auprc_metrics(dataset=dataset, scores=scores, n_bootstrap=10, rng=0)


def test_auprc_constant_scores_equals_prevalence_baseline():
    """When every score is identical, sklearn's ``average_precision_score``
    returns the positive-class prevalence (the random baseline). Lock in
    this invariant so a future refactor doesn't accidentally treat
    all-tied scores as undefined or as a degenerate split."""
    labels, _, mg = _matched_pairs(n_pos=20, k=9, separable=False)
    scores = np.full_like(labels, 0.5, dtype=float)  # every score identical
    res = auprc_with_bootstrap_se(labels, scores, mg, n_bootstrap=50, rng=0)
    expected = labels.sum() / len(labels)  # n_pos / n_total
    assert res["value"] == pytest.approx(expected)
    # All bootstrap iterations also return the prevalence baseline → SE ≈ 0.
    # (Sub-class-balance fluctuation across draws is the only source of
    # variance; with k=9 + 1000 iters it stays tiny.)
    assert res["se"] < 0.05


def test_compute_auprc_metrics_n_min_excludes_small_subsets():
    """The production scenario: some subsets fall below ``n_min`` and are
    excluded from the macro average, but still contribute to ``_global_``.
    This is the dominant case for complex_traits (per the dataset card,
    only distal and missense clear n_min=30)."""
    # Subset "small" has 10 positives (< n_min=30); "big" has 40.
    dataset, scores = _matched_pairs_with_subsets(
        subsets=["big"], n_pos_per_subset=40, seed=1
    )
    small_ds, small_scores = _matched_pairs_with_subsets(
        subsets=["small"], n_pos_per_subset=10, seed=2
    )
    # Renumber small's match_groups so they don't collide with big's.
    small_ds["match_group"] = small_ds["match_group"] + dataset["match_group"].max() + 1
    dataset = pd.concat([dataset, small_ds], ignore_index=True)
    scores = pd.concat([scores, small_scores], ignore_index=True)

    metrics = compute_auprc_metrics(
        dataset=dataset,
        scores=scores,
        score_columns=["score"],
        n_bootstrap=10,
        rng=0,
        n_min=30,
    )
    macro_row = metrics[
        (metrics["score_type"] == "score") & (metrics["subset"] == MACRO_AVG_SUBSET)
    ].iloc[0]
    big_row = metrics[
        (metrics["score_type"] == "score") & (metrics["subset"] == "big")
    ].iloc[0]
    # Only "big" qualifies → macro reduces to big's value, K=1.
    assert macro_row["value"] == pytest.approx(big_row["value"])
    assert macro_row["n_groups"] == 1
    # The small subset still has a per-subset row (we don't drop it; just
    # exclude it from the macro).
    assert (metrics["subset"] == "small").any()
    # Global covers all 50 groups, not just the 40 qualifying ones.
    global_row = metrics[
        (metrics["score_type"] == "score") & (metrics["subset"] == GLOBAL_SUBSET)
    ].iloc[0]
    assert global_row["n_groups"] == 50


# ---------------------------------------------------------------------------
# compute_qtl_metrics (unmatched DART-Eval QTL datasets: caqtl / dsqtl)
# ---------------------------------------------------------------------------


def _qtl_data(
    n_pos: int = 40,
    n_neg: int = 360,
    *,
    seed: int = 0,
    perfect: bool = False,
    control_effect_size_nan: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthesize an unmatched QTL dataset (no subset/match_group).

    Positives carry a measured ``effect_size``. Controls' ``effect_size`` is
    NaN by default (the dsQTL case — controls have no measured effect). With
    ``perfect=True``, positives' score equals their effect_size (→ pearson /
    spearman = 1 over positives) and controls score low (→ AUPRC = 1)."""
    rng = np.random.default_rng(seed)
    n = n_pos + n_neg
    label = np.concatenate([np.ones(n_pos, dtype=bool), np.zeros(n_neg, dtype=bool)])

    es_pos = rng.uniform(0.1, 2.0, n_pos)
    effect_size = np.full(n, np.nan)
    effect_size[:n_pos] = es_pos
    if not control_effect_size_nan:
        effect_size[n_pos:] = rng.uniform(0.1, 2.0, n_neg)

    score = np.empty(n)
    if perfect:
        score[:n_pos] = es_pos  # positives: score == effect_size
        score[n_pos:] = rng.uniform(0.0, 0.05, n_neg)  # controls clearly lower
    else:
        score[:] = rng.uniform(0.0, 1.0, n)

    dataset = pd.DataFrame({"label": label, "effect_size": effect_size})
    scores = pd.DataFrame({"score": score})
    return dataset, scores


def test_compute_qtl_metrics_shape():
    """Three rows per score column (AUPRC, pearson, spearman); fixed columns."""
    dataset, scores = _qtl_data(seed=3)
    scores["score2"] = np.random.default_rng(4).uniform(size=len(dataset))
    metrics = compute_qtl_metrics(dataset, scores, n_bootstrap=20, rng=0)
    assert len(metrics) == 2 * 3  # 2 score columns × 3 metrics
    assert set(metrics["metric"]) == {"AUPRC", "pearson", "spearman"}
    assert set(metrics["score_type"]) == {"score", "score2"}
    assert set(metrics.columns) == {
        "metric",
        "score_type",
        "value",
        "se",
        "n_rows",
        "n_pos",
    }
    assert metrics["value"].notna().all()
    assert metrics["se"].notna().all()


def test_compute_qtl_metrics_auprc_matches_sklearn():
    """The AUPRC row equals sklearn's average_precision_score over all rows."""
    dataset, scores = _qtl_data(seed=11)
    metrics = compute_qtl_metrics(dataset, scores, n_bootstrap=10, rng=0)
    auprc_row = metrics[metrics["metric"] == "AUPRC"].iloc[0]
    expected = float(
        average_precision_score(dataset["label"].astype(int), scores["score"])
    )
    assert auprc_row["value"] == pytest.approx(expected)
    assert auprc_row["n_rows"] == len(dataset)
    assert auprc_row["n_pos"] == int(dataset["label"].sum())


def test_compute_qtl_metrics_perfect_correlation_and_separation():
    """Positives' score == effect_size → pearson = spearman = 1 (SE≈0); and
    controls scoring lower → AUPRC = 1."""
    dataset, scores = _qtl_data(perfect=True, seed=5)
    metrics = compute_qtl_metrics(dataset, scores, n_bootstrap=50, rng=0)
    by_metric = metrics.set_index("metric")["value"]
    assert by_metric["pearson"] == pytest.approx(1.0, abs=1e-9)
    assert by_metric["spearman"] == pytest.approx(1.0, abs=1e-9)
    assert by_metric["AUPRC"] == pytest.approx(1.0, abs=1e-12)
    # Perfectly monotone positives → every bootstrap resample also corr=1.
    corr_se = metrics[metrics["metric"].isin(["pearson", "spearman"])]["se"]
    assert (corr_se < 1e-9).all()
    # The correlation rows used only the positives.
    n_pos = int(dataset["label"].sum())
    assert metrics.loc[metrics["metric"] == "pearson", "n_rows"].iloc[0] == n_pos


def test_compute_qtl_metrics_correlation_uses_positives_only():
    """Controls are excluded from the correlation: positives are perfectly
    correlated while controls are anti-correlated; result stays ~1."""
    rng = np.random.default_rng(0)
    n_pos, n_neg = 30, 200
    es_pos = rng.uniform(0.1, 2.0, n_pos)
    es_neg = rng.uniform(0.1, 2.0, n_neg)
    label = np.concatenate([np.ones(n_pos, bool), np.zeros(n_neg, bool)])
    effect_size = np.concatenate([es_pos, es_neg])
    # Positives: score == effect_size (corr +1). Controls: score == -effect_size
    # (corr -1). If controls leaked in, the pooled correlation would collapse.
    score = np.concatenate([es_pos, -es_neg])
    dataset = pd.DataFrame({"label": label, "effect_size": effect_size})
    scores = pd.DataFrame({"score": score})
    metrics = compute_qtl_metrics(dataset, scores, n_bootstrap=10, rng=0)
    assert metrics.set_index("metric").loc["pearson", "value"] == pytest.approx(
        1.0, abs=1e-9
    )


def test_compute_qtl_metrics_control_effect_size_nan_ok():
    """Controls with NaN effect_size (the dsQTL case) must not raise — only
    positives are required to carry a measured effect."""
    dataset, scores = _qtl_data(control_effect_size_nan=True, seed=2)
    assert dataset.loc[~dataset["label"], "effect_size"].isna().all()
    metrics = compute_qtl_metrics(dataset, scores, n_bootstrap=10, rng=0)
    assert metrics["value"].notna().all()


def test_compute_qtl_metrics_seed_reproducibility():
    dataset, scores = _qtl_data(seed=9)
    a = compute_qtl_metrics(dataset, scores, n_bootstrap=50, rng=0)
    b = compute_qtl_metrics(dataset, scores, n_bootstrap=50, rng=0)
    pd.testing.assert_frame_equal(a, b)


def test_compute_qtl_metrics_missing_effect_size_raises():
    dataset, scores = _qtl_data(seed=1)
    with pytest.raises(AssertionError, match="effect_size"):
        compute_qtl_metrics(dataset.drop(columns=["effect_size"]), scores)


def test_compute_qtl_metrics_nan_effect_size_positive_raises():
    """A positive missing its effect_size is a data error — fail loud."""
    dataset, scores = _qtl_data(seed=1)
    dataset.loc[0, "effect_size"] = np.nan  # row 0 is a positive
    with pytest.raises(AssertionError, match="effect_size"):
        compute_qtl_metrics(dataset, scores, n_bootstrap=10, rng=0)


def test_compute_qtl_metrics_single_class_raises():
    dataset = pd.DataFrame(
        {"label": [True, True, True], "effect_size": [1.0, 2.0, 3.0]}
    )
    scores = pd.DataFrame({"score": [0.1, 0.2, 0.3]})
    with pytest.raises(AssertionError, match="both classes"):
        compute_qtl_metrics(dataset, scores, n_bootstrap=10, rng=0)


def test_compute_qtl_metrics_nan_score_raises():
    dataset, scores = _qtl_data(seed=1)
    scores.loc[5, "score"] = np.nan
    with pytest.raises(AssertionError, match="NaN"):
        compute_qtl_metrics(dataset, scores, n_bootstrap=10, rng=0)


# ---------------------------------------------------------------------------
# compute_sge_metrics (saturation genome editing: evals_sge)
# ---------------------------------------------------------------------------


def _sge_data(
    *,
    n_per_cell: int = 150,
    accessions: tuple[tuple[str, str], ...] = (("urn:1", "GENEA"), ("urn:2", "GENEB")),
    seed: int = 0,
    perfect: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthesize an SGE v3 frame: accessions × {missense_variant, splicing} cells,
    each with a boolean ``label`` (~40% True = impactful) and a score. With
    ``perfect=True`` the score separates the classes (impactful scores high) →
    AUPRC = 1.
    """
    rng = np.random.default_rng(seed)
    frames = []
    score_blocks = []
    for urn, gene in accessions:
        for subset in ("missense_variant", "splicing"):
            label = rng.random(n_per_cell) < 0.4  # ~40% impactful
            frames.append(
                pd.DataFrame(
                    {
                        "mavedb_urn": urn,
                        "gene": gene,
                        "subset": subset,
                        "label": label,
                    }
                )
            )
            score_blocks.append(
                np.where(label, 1.0, 0.0) + rng.normal(0, 1e-6, n_per_cell)
                if perfect
                else rng.normal(size=n_per_cell)
            )
    dataset = pd.concat(frames, ignore_index=True)
    scores = pd.DataFrame({"score": np.concatenate(score_blocks)})
    return dataset, scores


def _val(df: pd.DataFrame, **filters: object) -> pd.Series:
    """Fetch the single row matching all column==value filters."""
    sub = df
    for col, value in filters.items():
        sub = sub[sub[col] == value]
    assert len(sub) == 1, f"expected 1 row for {filters}, got {len(sub)}"
    return sub.iloc[0]


def test_sge_shape_and_grid():
    """Columns fixed; AUPRC spans all four subset scopes per accession; the
    across-accession macro row exists; gene carried for display."""
    dataset, scores = _sge_data(seed=0)
    out = compute_sge_metrics(dataset, scores, n_bootstrap=20, rng=0)
    assert set(out.columns) == {
        "metric",
        "subset",
        "accession",
        "gene",
        "score_type",
        "value",
        "se",
        "n",
        "n_pos",
    }
    assert set(out["metric"]) == {"AUPRC"}
    assert set(out["subset"]) <= {
        "missense_variant",
        "splicing",
        SGE_POOLED_SUBSET,
        MACRO_AVG_SUBSET,
    }
    assert {"urn:1", "urn:2", MACRO_AVG_SUBSET} <= set(out["accession"])
    for urn in ("urn:1", "urn:2"):
        scopes = set(out[out["accession"] == urn]["subset"])
        assert scopes == {
            "missense_variant",
            "splicing",
            SGE_POOLED_SUBSET,
            MACRO_AVG_SUBSET,
        }, (urn, scopes)
    assert (
        _val(out, metric="AUPRC", subset=SGE_POOLED_SUBSET, accession="urn:1")["gene"]
        == "GENEA"
    )
    assert (
        _val(out, metric="AUPRC", subset=SGE_POOLED_SUBSET, accession=MACRO_AVG_SUBSET)[
            "gene"
        ]
        == MACRO_AVG_SUBSET
    )
    assert out["value"].notna().all()
    assert out["se"].notna().all()


def test_sge_perfect_auprc():
    """A score that separates the label (impactful scores high) → every AUPRC ≈ 1."""
    dataset, scores = _sge_data(seed=0, perfect=True)
    out = compute_sge_metrics(dataset, scores, n_bootstrap=30, rng=0)
    assert (out["value"] > 0.999).all()


def test_sge_auprc_matches_sklearn():
    """A leaf AUPRC equals sklearn's average_precision_score over the cell, with
    n = rows and n_pos = the impactful (label-True) count."""
    rng = np.random.default_rng(0)
    label = np.array([True] * 70 + [False] * 130)
    dataset = pd.DataFrame(
        {
            "mavedb_urn": "urn:1",
            "gene": "G",
            "subset": "missense_variant",
            "label": label,
        }
    )
    score = rng.normal(size=len(label))
    scores = pd.DataFrame({"score": score})
    out = compute_sge_metrics(dataset, scores, n_bootstrap=10, rng=0)
    auprc = _val(out, metric="AUPRC", subset="missense_variant", accession="urn:1")
    expected = average_precision_score(label.astype(int), score)
    assert auprc["value"] == pytest.approx(expected)
    assert auprc["n"] == 200
    assert auprc["n_pos"] == 70


def test_sge_subset_macro_is_mean_of_base_subsets():
    """The per-accession _macro_avg_ subset = unweighted mean of the missense &
    splicing AUPRC, with SE = sqrt(Σ SE²)/2 over the two."""
    dataset, scores = _sge_data(seed=3)
    out = compute_sge_metrics(dataset, scores, n_bootstrap=20, rng=0)
    m = _val(out, metric="AUPRC", subset="missense_variant", accession="urn:1")
    s = _val(out, metric="AUPRC", subset="splicing", accession="urn:1")
    macro = _val(out, metric="AUPRC", subset=MACRO_AVG_SUBSET, accession="urn:1")
    assert macro["value"] == pytest.approx((m["value"] + s["value"]) / 2)
    assert macro["se"] == pytest.approx(math.sqrt(m["se"] ** 2 + s["se"] ** 2) / 2)
    assert macro["n"] == 2


def test_sge_accession_macro_is_mean_over_accessions():
    """The accession _macro_avg_ row = unweighted mean over accessions of the
    same subset scope; n records the count of accessions averaged."""
    dataset, scores = _sge_data(seed=3)
    out = compute_sge_metrics(dataset, scores, n_bootstrap=20, rng=0)
    vals = [
        _val(out, metric="AUPRC", subset=SGE_POOLED_SUBSET, accession=u)["value"]
        for u in ("urn:1", "urn:2")
    ]
    macro = _val(
        out, metric="AUPRC", subset=SGE_POOLED_SUBSET, accession=MACRO_AVG_SUBSET
    )
    assert macro["value"] == pytest.approx(sum(vals) / 2)
    assert macro["n"] == 2


# ---------------------------------------------------------------------------
# compute_sge_probe_metrics (SGE linear-probe vs paired zero-shot baseline, #353)
# ---------------------------------------------------------------------------


def _sge_probe_df(
    *, seed: int = 0, perfect_probe: bool = True, nan_subset: str | None = None
) -> pd.DataFrame:
    """A ``compute_probe`` predictions frame for SGE: the ``_sge_data`` keys +
    ``probe_score`` (separates ``label`` when ``perfect_probe``) + the raw
    ``llr_fwd``/``llr_rc`` atoms the baseline is built from. If ``nan_subset`` is set,
    that subset's ``probe_score`` is all-NaN (a subset the probe skipped)."""
    dataset, _ = _sge_data(seed=seed)
    rng = np.random.default_rng(seed + 100)
    n = len(dataset)
    label = dataset["label"].to_numpy()
    df = dataset.copy()
    df["probe_score"] = (
        np.where(label, 1.0, 0.0) + rng.normal(0, 1e-6, n)
        if perfect_probe
        else rng.normal(size=n)
    )
    # Raw LLR atoms: impactful (label True) gets negative LLR → minus_llr high (the SGE
    # deleteriousness orientation), so the zero-shot baseline is weakly informative.
    signal = np.where(label, -1.0, 1.0)
    df["llr_fwd"] = signal + rng.normal(0, 0.5, n)
    df["llr_rc"] = signal + rng.normal(0, 0.5, n)
    if nan_subset is not None:
        df.loc[df["subset"] == nan_subset, "probe_score"] = np.nan
    return df


def test_sge_probe_metrics_shape_and_paired_baseline():
    """Reuses the per-accession × subset SGE grid, emitting BOTH the probe score and its
    paired zero-shot baseline (minus_llr_avg); a perfect probe → AUPRC ≈ 1."""
    df = _sge_probe_df(seed=0, perfect_probe=True)
    out = compute_sge_probe_metrics(df, "minus_llr", n_bootstrap=20, rng=0)
    # Exactly two score types on identical rows: the probe + its zero-shot baseline.
    assert set(out["score_type"]) == {"probe_score", "minus_llr_avg"}
    # Same per-accession × subset structure as compute_sge_metrics.
    assert set(out["subset"]) <= {
        "missense_variant",
        "splicing",
        SGE_POOLED_SUBSET,
        MACRO_AVG_SUBSET,
    }
    assert {"urn:1", "urn:2", MACRO_AVG_SUBSET} <= set(out["accession"])
    probe_cells = out[(out["score_type"] == "probe_score") & out["value"].notna()]
    assert (probe_cells["value"] > 0.999).all()


def test_sge_probe_metrics_drops_skipped_subset():
    """Rows the probe skipped (all-NaN ``probe_score`` for a subset below the gate) are
    dropped, so that subset is absent while the scored subset is still evaluated for both
    score types."""
    df = _sge_probe_df(seed=1, perfect_probe=True, nan_subset="splicing")
    out = compute_sge_probe_metrics(df, "minus_llr", n_bootstrap=20, rng=0)
    assert "splicing" not in set(out["subset"])
    assert "missense_variant" in set(out["subset"])
    assert set(out["score_type"]) == {"probe_score", "minus_llr_avg"}


def test_sge_n_min_auprc_gates_small_cells():
    """A subset cell with <n_min_auprc of either label class is gated: it's still
    emitted (so a blanked cell reports its class balance) but with value/se = NaN
    and a real n / n_pos, and it's excluded from the subset-macro (which then
    averages only the qualifying base subset, K=1)."""
    rng = np.random.default_rng(0)

    def block(subset: str, n_pos: int, n_neg: int) -> tuple[pd.DataFrame, np.ndarray]:
        label = np.array([True] * n_pos + [False] * n_neg)
        df = pd.DataFrame(
            {"mavedb_urn": "urn:1", "gene": "G", "subset": subset, "label": label}
        )
        return df, rng.normal(size=n_pos + n_neg)

    dm, sm = block("missense_variant", 60, 90)  # ≥30 each → qualifies
    ds, ss = block("splicing", 10, 90)  # only 10 impactful (<30) → gated
    dataset = pd.concat([dm, ds], ignore_index=True)
    scores = pd.DataFrame({"score": np.concatenate([sm, ss])})
    out = compute_sge_metrics(dataset, scores, n_bootstrap=10, rng=0)

    miss = _val(out, metric="AUPRC", subset="missense_variant", accession="urn:1")
    assert np.isfinite(miss["value"])
    # Gated splicing cell: emitted, value NaN, but class counts preserved.
    spl = _val(out, metric="AUPRC", subset="splicing", accession="urn:1")
    assert np.isnan(spl["value"]) and np.isnan(spl["se"])
    assert spl["n_pos"] == 10 and spl["n"] == 100  # n_neg = n - n_pos = 90
    # Subset-macro excludes the gated cell → equals missense alone (K=1).
    macro = _val(out, metric="AUPRC", subset=MACRO_AVG_SUBSET, accession="urn:1")
    assert macro["value"] == pytest.approx(miss["value"])
    assert macro["n"] == 1


def test_sge_nan_scores_dropped():
    """NaN scores (conservation has no value at unaligned loci) are dropped per
    cell rather than raising; AUPRC is computed on the finite remainder."""
    dataset, scores = _sge_data(seed=2)
    scores = scores.copy()
    scores.loc[:20, "score"] = (
        np.nan
    )  # 21 NaN scores in the first cell (urn:1 missense)
    out = compute_sge_metrics(dataset, scores, n_bootstrap=10, rng=0)
    a = out[out["metric"] == "AUPRC"]
    assert len(a) > 0
    # The first cell shrinks 150→129 but still clears the gate (≥30/class), so no
    # cell is gated → every AUPRC value is finite. (A gated cell would carry NaN;
    # this asserts the fixture stays un-gated, not that gating is impossible.)
    assert a["value"].notna().all()
    # The 21 NaN-score rows were dropped, not counted: n reflects the remainder.
    first = a[(a["accession"] == "urn:1") & (a["subset"] == "missense_variant")]
    assert int(first["n"].iloc[0]) == 150 - 21


def test_sge_multiple_score_columns():
    dataset, scores = _sge_data(seed=0)
    scores = scores.copy()
    scores["jsd"] = np.random.default_rng(1).normal(size=len(scores))
    out = compute_sge_metrics(dataset, scores, n_bootstrap=10, rng=0)
    assert set(out["score_type"]) == {"score", "jsd"}


def test_sge_urn_maps_to_multiple_genes_raises():
    dataset = pd.DataFrame(
        {
            "mavedb_urn": ["urn:1"] * 4,
            "gene": ["A", "A", "B", "B"],
            "subset": ["missense_variant"] * 4,
            "label": [True, False, True, False],
        }
    )
    scores = pd.DataFrame({"score": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(AssertionError, match="maps to >1 gene"):
        compute_sge_metrics(dataset, scores, n_bootstrap=2, rng=0)


def test_sge_seed_reproducibility():
    dataset, scores = _sge_data(seed=1)
    a = compute_sge_metrics(dataset, scores, n_bootstrap=50, rng=0)
    b = compute_sge_metrics(dataset, scores, n_bootstrap=50, rng=0)
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------------------
# per_chrom_weighted_ap (TraitGym / #314 linear-probe headline metric)
# ---------------------------------------------------------------------------


def _ref_per_chrom_weighted_ap(
    y: np.ndarray, score: np.ndarray, chrom: np.ndarray
) -> float:
    """Verbatim copy of the iter3 inlined helper (the finite-score-guard variant) —
    the bit-for-bit reference the library function must reproduce. The source lives
    on the #314 branch, not ``main``: ``scripts/issue314/iter3_transfer.py`` at commit
    e963ebd (.../blob/e963ebd/scripts/issue314/iter3_transfer.py#L35-L42)."""
    tot, w = 0.0, 0
    for c in np.unique(chrom):
        m = (chrom == c) & np.isfinite(score)
        if 0 < int(y[m].sum()) < int(m.sum()):
            tot += average_precision_score(y[m], score[m]) * int(m.sum())
            w += int(m.sum())
    return tot / w if w else float("nan")


def test_per_chrom_weighted_ap_perfect_separation():
    """Within every chromosome positives outscore negatives → each chrom AUPRC=1
    → weighted mean 1.0, regardless of the (unequal) chromosome sizes."""
    y = np.array([1, 1, 0, 0, 0, 1, 0, 0])
    score = np.array([0.9, 0.8, 0.2, 0.1, 0.3, 0.95, 0.05, 0.15])
    chrom = np.array(["chr1", "chr1", "chr1", "chr1", "chr1", "chr2", "chr2", "chr2"])
    assert per_chrom_weighted_ap(y, score, chrom) == pytest.approx(1.0)


def test_per_chrom_weighted_ap_single_class_chrom_skipped():
    """A chromosome with only one class is dropped: the result is exactly the
    AUPRC of the lone both-class chromosome (the skipped one adds no weight)."""
    y = np.array([1, 1, 0, 0, 1, 1, 1, 1])
    score = np.array([0.9, 0.3, 0.6, 0.2, 0.7, 0.5, 0.4, 0.8])
    chrom = np.array(["chr1"] * 5 + ["chr2"] * 3)  # chr2 is all-positive → skipped
    m1 = chrom == "chr1"
    expected = average_precision_score(y[m1], score[m1])
    assert per_chrom_weighted_ap(y, score, chrom) == pytest.approx(expected)


def test_per_chrom_weighted_ap_size_weighted_not_equal_weighted():
    """The mean is weighted by chromosome size: two both-class chromosomes of
    different sizes and different AUPRCs → the result is the size-weighted mean
    and differs from the plain (equal-weight) mean."""
    # chr1: 10 variants, imperfect ranking. chr2: 4 variants, perfect ranking.
    y1 = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    s1 = np.array([0.5, 0.9, 0.4, 0.3, 0.2, 0.8, 0.1, 0.05, 0.6, 0.7])
    y2 = np.array([1, 0, 1, 0])
    s2 = np.array([0.9, 0.1, 0.8, 0.2])
    y = np.concatenate([y1, y2])
    score = np.concatenate([s1, s2])
    chrom = np.array(["chr1"] * len(y1) + ["chr2"] * len(y2))
    ap1 = average_precision_score(y1, s1)
    ap2 = average_precision_score(y2, s2)
    n1, n2 = len(y1), len(y2)
    expected = (ap1 * n1 + ap2 * n2) / (n1 + n2)
    got = per_chrom_weighted_ap(y, score, chrom)
    assert got == pytest.approx(expected)
    # Guard the fixture: the chroms must actually differ in AUPRC and size, so
    # the test genuinely separates size-weighting from equal-weighting.
    assert ap1 != pytest.approx(ap2) and n1 != n2
    assert got != pytest.approx((ap1 + ap2) / 2)


def test_per_chrom_weighted_ap_nonfinite_scores_dropped():
    """Non-finite scores (NaN, ±inf) are excluded per chromosome — they neither
    count toward a class nor toward the size weight; a chromosome left
    single-class after the drop is skipped."""
    # chr1: a NaN and a +inf dropped → finite rows [0.9 (pos), 0.2 (neg)] → AP=1, n=2.
    y1 = np.array([1, 1, 0, 0])
    s1 = np.array([0.9, np.nan, 0.2, np.inf])
    # chr2: a -inf dropped → finite labels [1, 0, 0], scores [0.5, 0.6, 0.4].
    y2 = np.array([1, 0, 1, 0])
    s2 = np.array([0.5, 0.6, -np.inf, 0.4])
    # chr3: its only finite-score row is a positive → single class → skipped.
    y3 = np.array([1, 1])
    s3 = np.array([np.nan, 0.5])
    y = np.concatenate([y1, y2, y3])
    score = np.concatenate([s1, s2, s3])
    chrom = np.array(["chr1"] * 4 + ["chr2"] * 4 + ["chr3"] * 2)

    f1, f2 = np.isfinite(s1), np.isfinite(s2)
    ap1, n1 = average_precision_score(y1[f1], s1[f1]), int(f1.sum())
    ap2, n2 = average_precision_score(y2[f2], s2[f2]), int(f2.sum())
    expected = (ap1 * n1 + ap2 * n2) / (n1 + n2)  # chr3 contributes nothing
    assert per_chrom_weighted_ap(y, score, chrom) == pytest.approx(expected)


def test_per_chrom_weighted_ap_all_single_class_returns_nan():
    """No chromosome has both classes → undefined → nan (not 0, not a crash)."""
    y = np.array([1, 1, 0, 0])
    score = np.array([0.9, 0.8, 0.2, 0.1])
    chrom = np.array(["chr1", "chr1", "chr2", "chr2"])  # chr1 all-pos, chr2 all-neg
    assert math.isnan(per_chrom_weighted_ap(y, score, chrom))


def test_per_chrom_weighted_ap_length_mismatch_raises():
    with pytest.raises(AssertionError, match="length mismatch"):
        per_chrom_weighted_ap(
            np.array([1, 0, 1]),
            np.array([0.5, 0.5]),
            np.array(["chr1", "chr1", "chr2"]),
        )


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_per_chrom_weighted_ap_matches_inlined_reference(seed: int):
    """Bit-for-bit agreement with the verbatim iter3 helper on a randomized
    fixture spanning non-finite scores and a forced single-class chromosome
    (the issue's stated verification)."""
    rng = np.random.default_rng(seed)
    n = 200
    y = (rng.random(n) < 0.3).astype(int)
    score = rng.normal(size=n)
    score[rng.choice(n, size=15, replace=False)] = np.nan
    score[rng.choice(n, size=5, replace=False)] = np.inf
    chrom = rng.choice(["chr1", "chr2", "chr3", "chr4"], size=n)
    chrom[:6] = "chrX"  # force a single-class chromosome into the mix
    y[:6] = 1
    got = per_chrom_weighted_ap(y, score, chrom)
    expected = _ref_per_chrom_weighted_ap(y, score, chrom)
    assert got == pytest.approx(expected, nan_ok=True)


def test_per_chrom_weighted_ap_non_binary_labels_raise():
    """Labels outside {0, 1} fail loud rather than silently mis-skipping a
    chromosome — an un-cast object/str column or a {0, 2} encoding would otherwise
    corrupt the size-weighted mean (silently, via a string-concatenated sum)."""
    with pytest.raises(AssertionError, match="labels must be 0/1"):
        per_chrom_weighted_ap(
            np.array([2, 0, 2, 0]),
            np.array([0.9, 0.2, 0.8, 0.1]),
            np.array(["chr1", "chr1", "chr1", "chr1"]),
        )


# ---------------------------------------------------------------------------
# per_chrom_ap_table (per-subset per-chrom-weighted AUPRC table, #341)
# ---------------------------------------------------------------------------


def _probe_like_frame() -> pd.DataFrame:
    """Two subsets × two score columns, each subset spanning two chromosomes so the
    per-chrom weighting is exercised — mirrors the ``compute_probe`` predictions frame
    (``label`` / ``subset`` / ``chrom`` + score columns)."""
    return pd.DataFrame(
        {
            "label": [1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0],
            "subset": ["missense"] * 6 + ["synonymous"] * 6,
            "chrom": (
                ["chr1", "chr1", "chr1", "chr2", "chr2", "chr2"]
                + ["chr1", "chr1", "chr1", "chr3", "chr3", "chr3"]
            ),
            "probe_score": [0.9, 0.7, 0.2, 0.3, 0.8, 0.6, 0.4, 0.9, 0.1, 0.7, 0.2, 0.5],
            "minus_llr_avg": [
                0.2,
                0.4,
                0.8,
                0.7,
                0.1,
                0.3,
                0.6,
                0.2,
                0.9,
                0.4,
                0.8,
                0.5,
            ],
        }
    )


def test_per_chrom_ap_table_shape_and_matches_direct():
    """One row per (subset, score_col); each ``value`` reproduces
    ``per_chrom_weighted_ap`` called directly on that subset's rows."""
    df = _probe_like_frame()
    out = per_chrom_ap_table(df, ["probe_score", "minus_llr_avg"])
    assert list(out.columns) == [
        "score_type",
        "subset",
        "value",
        "n",
        "n_pos",
        "n_chrom",
    ]
    assert len(out) == 4  # 2 subsets × 2 score columns
    assert set(out["subset"]) == {"missense", "synonymous"}
    assert set(out["score_type"]) == {"probe_score", "minus_llr_avg"}
    for (subset, score_type), row in out.set_index(["subset", "score_type"]).iterrows():
        sub = df[df["subset"] == subset]
        expected = per_chrom_weighted_ap(
            sub["label"].to_numpy(),
            sub[score_type].to_numpy(),
            sub["chrom"].to_numpy(),
        )
        assert row["value"] == pytest.approx(expected)
    missense = out[out["subset"] == "missense"].iloc[0]
    assert missense["n"] == 6 and missense["n_pos"] == 3 and missense["n_chrom"] == 2


def test_per_chrom_ap_table_all_nan_score_is_nan_row():
    """A score column that is all-NaN within a subset (a skipped probe) yields a NaN
    value for that (subset, score) while the baseline stays finite; the subset-level
    counts are score-independent."""
    df = _probe_like_frame()
    df.loc[df["subset"] == "missense", "probe_score"] = np.nan
    out = per_chrom_ap_table(df, ["probe_score", "minus_llr_avg"])
    probe_row = out[
        (out["subset"] == "missense") & (out["score_type"] == "probe_score")
    ].iloc[0]
    base_row = out[
        (out["subset"] == "missense") & (out["score_type"] == "minus_llr_avg")
    ].iloc[0]
    assert math.isnan(probe_row["value"])
    assert np.isfinite(base_row["value"])
    assert probe_row["n"] == 6 and probe_row["n_pos"] == 3 and probe_row["n_chrom"] == 2


def test_per_chrom_ap_table_preserves_subset_order():
    """Subsets appear in first-appearance order (``groupby(sort=False)``)."""
    out = per_chrom_ap_table(_probe_like_frame(), ["probe_score"])
    assert list(out["subset"]) == ["missense", "synonymous"]


def test_per_chrom_ap_table_missing_column_raises():
    """A missing key column or a missing score column fails loud."""
    with pytest.raises(AssertionError, match="missing required column"):
        per_chrom_ap_table(_probe_like_frame().drop(columns=["chrom"]), ["probe_score"])
    with pytest.raises(AssertionError, match="missing score columns"):
        per_chrom_ap_table(_probe_like_frame(), ["nonexistent"])


def test_per_chrom_ap_table_null_key_raises():
    """A null subset / chrom / label fails loud rather than being silently dropped by
    groupby(dropna) or mis-weighting the per-chrom AUPRC."""
    for key in ("subset", "chrom", "label"):
        df = _probe_like_frame()
        df.loc[0, key] = None
        with pytest.raises(AssertionError, match="contains nulls"):
            per_chrom_ap_table(df, ["probe_score"])
