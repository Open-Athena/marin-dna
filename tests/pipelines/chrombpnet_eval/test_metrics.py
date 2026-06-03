"""Tests for the supervised QTL metric (signed correlation + |score| binary)."""

import numpy as np
import pandas as pd
import pytest

from marin_dna.pipelines.chrombpnet_eval.metrics import compute_supervised_qtl_metrics


def _val(m: pd.DataFrame, metric: str) -> float:
    return float(m.loc[m.metric == metric, "value"].iloc[0])


def _toy() -> pd.DataFrame:
    """4 positives (signed score == signed effect, perfectly aligned) + 4
    controls with small |score| and no measured effect (NaN, as for dsQTL)."""
    return pd.DataFrame(
        {
            "label": [True, True, True, True, False, False, False, False],
            "score": [1.0, -2.0, 3.0, -0.5, 0.01, -0.02, 0.0, 0.03],
            "effect": [1.0, -2.0, 3.0, -0.5, np.nan, np.nan, np.nan, np.nan],
        }
    )


def test_output_shape_and_metric_names():
    m = compute_supervised_qtl_metrics(_toy(), n_bootstrap=50)
    assert set(m["metric"]) == {"AUROC", "AUPRC", "pearson", "spearman"}
    assert list(m.columns) == ["metric", "value", "se", "n_rows", "n_pos"]
    # binary metrics span all rows; correlations span positives only.
    assert _row(m, "AUROC")["n_rows"] == 8
    assert _row(m, "pearson")["n_rows"] == 4
    assert (m["n_pos"] == 4).all()


def _row(m: pd.DataFrame, metric: str) -> pd.Series:
    return m.loc[m.metric == metric].iloc[0]


def test_perfect_signed_correlation():
    m = compute_supervised_qtl_metrics(_toy(), n_bootstrap=50)
    assert _val(m, "pearson") == pytest.approx(1.0)
    assert _val(m, "spearman") == pytest.approx(1.0)


def test_binary_separates_by_magnitude():
    # positives have large |score|, controls ~0 → perfect separation.
    m = compute_supervised_qtl_metrics(_toy(), n_bootstrap=50)
    assert _val(m, "AUROC") == pytest.approx(1.0)
    assert _val(m, "AUPRC") == pytest.approx(1.0)


def test_sign_flips_correlation_but_not_binary():
    """|score| binary metrics are sign-invariant; signed correlation flips."""
    df = _toy()
    m1 = compute_supervised_qtl_metrics(df, n_bootstrap=50)
    df2 = df.assign(score=-df["score"])
    m2 = compute_supervised_qtl_metrics(df2, n_bootstrap=50)
    for metric in ("AUROC", "AUPRC"):
        assert _val(m1, metric) == pytest.approx(_val(m2, metric))
    assert _val(m1, "pearson") == pytest.approx(-_val(m2, "pearson"))
    assert _val(m1, "spearman") == pytest.approx(-_val(m2, "spearman"))


def test_anticorrelated_model_is_negative():
    # model score is the negation of the true effect on positives.
    df = pd.DataFrame(
        {
            "label": [True, True, True, False, False, False],
            "score": [-1.0, 2.0, -3.0, 0.0, 0.1, -0.1],
            "effect": [1.0, -2.0, 3.0, np.nan, np.nan, np.nan],
        }
    )
    m = compute_supervised_qtl_metrics(df, n_bootstrap=20)
    assert _val(m, "pearson") == pytest.approx(-1.0)


def test_controls_may_lack_effect():
    # NaN effect on controls must not raise (correlation is positives-only).
    compute_supervised_qtl_metrics(_toy(), n_bootstrap=10)


def test_nan_score_raises():
    df = _toy()
    df.loc[0, "score"] = np.nan
    with pytest.raises(AssertionError, match="has NaN"):
        compute_supervised_qtl_metrics(df, n_bootstrap=10)


def test_positive_missing_effect_raises():
    df = _toy()
    df.loc[0, "effect"] = np.nan  # a positive with no measured effect
    with pytest.raises(AssertionError, match="NaN"):
        compute_supervised_qtl_metrics(df, n_bootstrap=10)


def test_single_class_raises():
    df = _toy()
    df["label"] = True
    with pytest.raises(AssertionError, match="both classes"):
        compute_supervised_qtl_metrics(df, n_bootstrap=10)


def test_se_is_nonnegative_and_finite():
    m = compute_supervised_qtl_metrics(_toy(), n_bootstrap=100)
    assert (m["se"] >= 0).all()
    assert np.isfinite(m["se"]).all()
