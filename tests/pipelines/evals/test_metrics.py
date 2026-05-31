"""Tests for the classical (AUPRC/AUROC/Spearman) metrics API and cross-run
aggregation, plus the cluster-bootstrap AUPRC used by ``evals_v2``.
Per-metric tests for ``pairwise_accuracy`` live in
``test_pairwise_accuracy.py``."""

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
    aggregate_metrics,
    auprc_with_bootstrap_se,
    compute_auprc_metrics,
    compute_metrics,
    compute_qtl_metrics,
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
