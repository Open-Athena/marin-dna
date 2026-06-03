"""Supervised variant-effect metrics for the ChromBPNet-on-embeddings eval.

These are the DART-Eval Task-5 / ARSENAL metrics for a **supervised** model that
emits a *signed* allelic score (e.g. ``log2(alt_counts / ref_counts)`` from a
trained ChromBPNet). They are deliberately **separate** from
``marin_dna.pipelines.evals.metrics.compute_qtl_metrics`` (which serves the
zero-shot/LLR leaderboard and correlates against the *unsigned* ``effect_size``):
a supervised model's score is directional, so here we correlate the **signed**
score against the **signed** study effect — exactly as ARSENAL's
``supervised_variant_scoring_*.ipynb`` does.

Two metric families, each over a different variant set (ARSENAL Methods §,
verified against their notebook):

- **AUROC / AUPRC on ``|score|``** over **all** used variants (significant QTL vs
  control). Magnitude is what discriminates: a significant QTL perturbs
  accessibility in *either* direction, so the sign carries no class information.
- **Pearson / Spearman of the signed score vs the signed study ``effect``**, over
  the **positive (significant) variants only** — "correlation metrics are only
  calculated on variants with significant observed effect."

Standard errors are nonparametric row-bootstrap (std of the bootstrap
distribution, ``ddof=1``, NaN-tolerant). They are *marginal, single-scorer* SEs;
comparing two scorers needs a paired-delta bootstrap, not overlapping ``±se``
bars (same caveat as ``compute_qtl_metrics``).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


def _binary_metric_bootstrap_se(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    label: np.ndarray,
    score: np.ndarray,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> float:
    """Row-bootstrap SE of a binary ranking metric (AUROC/AUPRC).

    Resamples paired ``(label, score)`` rows with replacement. Resamples that
    end up single-class leave the metric undefined → NaN, excluded from the std.
    """
    n = len(label)
    boot = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y = label[idx]
        if y.min() == y.max():  # single class → AUROC/AUPRC undefined
            boot[b] = np.nan
            continue
        boot[b] = metric_fn(y, score[idx])
    return float(np.nanstd(boot, ddof=1))


def _correlation_bootstrap_se(
    corr_fn: Callable[[np.ndarray, np.ndarray], tuple[float, float]],
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> float:
    """Row-bootstrap SE of a correlation. Resamples constant in either variable
    leave the correlation undefined → NaN, excluded from the std."""
    n = len(x)
    boot = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        if np.std(xb) == 0 or np.std(yb) == 0:
            boot[b] = np.nan
            continue
        boot[b] = corr_fn(xb, yb)[0]
    return float(np.nanstd(boot, ddof=1))


def compute_supervised_qtl_metrics(
    df: pd.DataFrame,
    *,
    score_col: str = "score",
    label_col: str = "label",
    effect_col: str = "effect",
    n_bootstrap: int = 1000,
    rng: np.random.Generator | int | None = 0,
) -> pd.DataFrame:
    """ARSENAL supervised-VEP metrics for one signed score column.

    Args:
        df: one row per variant, with:
            - ``label_col``: bool / 0-1 — ``True`` = significant QTL (positive),
              ``False`` = control (negative).
            - ``score_col``: float — the model's **signed** allelic score
              (e.g. ``log2(alt/ref)`` counts), oriented to ``alt``. No NaN.
            - ``effect_col``: float — the **signed** study effect, oriented to
              ``alt``. May be NaN for controls, but **must** be present for every
              positive (asserted).
        score_col / label_col / effect_col: column names.
        n_bootstrap: bootstrap iterations per metric.
        rng: ``numpy.random.Generator``, seed int, or ``None``. Default ``0`` for
            bit-stable SEs across re-runs on identical inputs.

    Returns:
        Long-form ``DataFrame[metric, value, se, n_rows, n_pos]`` with ``metric``
        in ``{AUROC, AUPRC, pearson, spearman}``. AUROC/AUPRC use ``|score|`` over
        all rows (``n_rows`` = total); pearson/spearman use the signed score vs
        signed effect over positives (``n_rows`` = ``n_pos``).
    """
    for col in (score_col, label_col, effect_col):
        assert col in df.columns, f"df missing required column {col!r}"

    label = np.asarray(df[label_col]).astype(int)
    score = np.asarray(df[score_col], dtype=float)
    effect = np.asarray(df[effect_col], dtype=float)

    assert not np.isnan(score).any(), (
        f"score column {score_col!r} has NaN — fill/align upstream before scoring"
    )
    n_rows = int(len(label))
    pos_mask = label == 1
    n_pos = int(pos_mask.sum())
    assert 0 < n_pos < n_rows, (
        f"binary metrics need both classes, got n_pos={n_pos} of n={n_rows}"
    )
    assert n_pos >= 2, f"correlation needs >=2 positives, got n_pos={n_pos}"
    # Positives must carry a measured effect (controls legitimately may not —
    # e.g. dsQTL controls have no measured effect). A NaN here means a positive
    # is missing its effect → fail loud rather than silently drop it.
    n_nan_pos = int(np.isnan(effect[pos_mask]).sum())
    assert n_nan_pos == 0, (
        f"{n_nan_pos} positive variants have NaN {effect_col!r} — expected a "
        f"measured effect for every positive"
    )

    rng = np.random.default_rng(rng)
    abs_score = np.abs(score)
    rows: list[dict] = []

    # Binary classification on |score| over all used variants.
    for name, fn in (("AUROC", roc_auc_score), ("AUPRC", average_precision_score)):
        value = float(fn(label, abs_score))
        se = _binary_metric_bootstrap_se(
            fn, label, abs_score, n_bootstrap=n_bootstrap, rng=rng
        )
        rows.append(
            {"metric": name, "value": value, "se": se, "n_rows": n_rows, "n_pos": n_pos}
        )

    # Signed correlation over positives only.
    score_pos, effect_pos = score[pos_mask], effect[pos_mask]
    for name, fn in (("pearson", pearsonr), ("spearman", spearmanr)):
        value = float(fn(score_pos, effect_pos)[0])
        se = _correlation_bootstrap_se(
            fn, score_pos, effect_pos, n_bootstrap=n_bootstrap, rng=rng
        )
        rows.append(
            {"metric": name, "value": value, "se": se, "n_rows": n_pos, "n_pos": n_pos}
        )

    return pd.DataFrame(rows)
