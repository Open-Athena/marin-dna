"""Frozen-embedding linear probe for variant effect prediction (issue #314).

CPU-side feature construction + chromosome-grouped out-of-fold cross-validation
+ AUPRC scoring, operating on per-variant embeddings cached by the GPU
extraction pass (the shared HF runner; see ``run_variant_embeddings``).

The design is deliberately small and explicit (issue #314 is exploratory):

- **Feature construction** combines pooled ref/alt embeddings ``[N, D]`` into a
  per-variant feature (``pair_feature``), or derives features that need the full
  per-token states ``[N, L, D]`` (``innerprod_feature``, ``cov_delta_feature``).
- **Pooling** reduces per-token states ``[N, L, D]`` to ``[N, D]`` over a chosen
  spatial extent (``pool_tokens``). Pooling extent is a primary sweep axis.
- **The probe** is a linear sklearn pipeline (``make_linear_probe``): optional
  ``StandardScaler`` (evaluated on/off) → optional PCA-k → L2 logistic or ridge.
- **CV** is leak-proof chromosome-grouped K-fold (``chrom_grouped_oof``); OOF
  scores are scored with the matched-pair AUPRC + cluster-bootstrap SE from
  ``metrics`` (``probe_auprc``).

``ref``/``alt`` always denote the reference- and alternate-allele embeddings.
Features marked symmetric are invariant under a ref↔alt swap — required for the
swap-invariant datasets (complex_traits, caqtl, dsqtl) and for cross-dataset
transfer.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from marin_dna.pipelines.evals.metrics import auprc_with_bootstrap_se

# ref↔alt combinations of pooled embeddings. The symmetric ones are invariant
# under swapping which allele is "ref" — the only valid features for datasets
# with no biological ref/alt direction, and for cross-dataset transfer.
PAIR_COMBOS: tuple[str, ...] = ("delta", "concat", "abs_delta", "prod", "sum_absdiff")
SYMMETRIC_COMBOS: frozenset[str] = frozenset({"abs_delta", "prod", "sum_absdiff"})

POOLING_EXTENTS: tuple[str, ...] = ("entire_window", "center", "variant_token", "max")


def pair_feature(ref: np.ndarray, alt: np.ndarray, combo: str) -> np.ndarray:
    """Combine pooled ref/alt embeddings ``[N, D]`` into a per-variant feature.

    ``delta``/``concat`` are signed (ref↔alt direction matters); ``abs_delta``/
    ``prod``/``sum_absdiff`` are invariant under a ref↔alt swap (see
    ``SYMMETRIC_COMBOS``). Returns ``[N, D]`` for ``delta``/``abs_delta``/``prod``
    and ``[N, 2D]`` for ``concat``/``sum_absdiff``.
    """
    assert ref.shape == alt.shape and ref.ndim == 2, (ref.shape, alt.shape)
    if combo == "delta":
        return alt - ref
    if combo == "concat":
        return np.concatenate([ref, alt], axis=1)
    if combo == "abs_delta":
        return np.abs(alt - ref)
    if combo == "prod":
        return ref * alt
    if combo == "sum_absdiff":
        return np.concatenate([ref + alt, np.abs(alt - ref)], axis=1)
    raise ValueError(f"unknown combo {combo!r}; expected one of {PAIR_COMBOS}")


def pool_tokens(
    states: np.ndarray,
    extent: str,
    *,
    var_index: int | None = None,
    n_center: int = 100,
) -> np.ndarray:
    """Pool per-token last-layer states ``[N, L, D]`` to ``[N, D]``.

    ``extent``:

    - ``entire_window`` — mean over all ``L`` positions (the natural baseline).
    - ``center`` — mean over the central ``n_center`` positions.
    - ``variant_token`` — the single position ``var_index``.
    - ``max`` — per-dimension max over ``L``.
    """
    assert states.ndim == 3, states.shape
    _, length, _ = states.shape
    if extent == "entire_window":
        return states.mean(axis=1)
    if extent == "max":
        return states.max(axis=1)
    if extent == "variant_token":
        assert var_index is not None and 0 <= var_index < length, var_index
        # .copy() — a bare slice is a view that pins the whole parent array; when
        # pooled features accumulate across many shards that leaks GBs (issue #314).
        return states[:, var_index, :].copy()
    if extent == "center":
        assert 0 < n_center <= length, (n_center, length)
        lo = (length - n_center) // 2
        return states[:, lo : lo + n_center, :].mean(axis=1)
    raise ValueError(f"unknown extent {extent!r}; expected one of {POOLING_EXTENTS}")


def innerprod_feature(ref_tok: np.ndarray, alt_tok: np.ndarray) -> np.ndarray:
    """Per-dimension dot product over spatial positions: ``Σ_t ref[t] ∘ alt[t]`` → ``[N, D]``.

    Symmetric under a ref↔alt swap. Needs per-token states ``[N, L, D]`` — it is
    *not* reconstructable from pooled vectors (pooled-then-multiply ``prod`` ≠
    multiply-per-token-then-sum).
    """
    assert ref_tok.shape == alt_tok.shape and ref_tok.ndim == 3, ref_tok.shape
    return (ref_tok * alt_tok).sum(axis=1)


def random_projection(d: int, r: int, *, seed: int = 0) -> np.ndarray:
    """Fixed Gaussian down-projection matrix ``[d, r]`` (1/√r scaled), seeded."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((d, r)) / np.sqrt(r)).astype(np.float32)


def cov_delta_feature(
    ref_tok: np.ndarray, alt_tok: np.ndarray, proj: np.ndarray
) -> np.ndarray:
    """EVEE-style second-order pooling: down-projected Gram of the per-token delta.

    ``X = alt_tok − ref_tok`` ``[N, L, D]`` → project to ``r`` dims
    (``Y = X @ proj`` ``[N, L, r]``) → per-sample Gram ``G = YᵀY`` ``[N, r, r]`` →
    flatten → ``[N, r²]``. Projecting *before* the Gram keeps it tractable (the
    full ``XᵀX`` is ``D×D``). ``G`` is invariant to the sign of ``X``, so the
    feature is symmetric under a ref↔alt swap. ``proj`` is ``[D, r]`` (see
    ``random_projection``).

    Reference: EVEE (Goodfire, bioRxiv 2026.04.10.717844).
    """
    assert ref_tok.shape == alt_tok.shape and ref_tok.ndim == 3, ref_tok.shape
    assert proj.ndim == 2 and proj.shape[0] == ref_tok.shape[2], proj.shape
    x = alt_tok - ref_tok  # [N, L, D]
    y = x @ proj  # [N, L, r]
    gram = np.matmul(np.swapaxes(y, 1, 2), y)  # [N, r, r] = YᵀY per sample (BLAS)
    return gram.reshape(gram.shape[0], -1)  # [N, r²]


def make_linear_probe(
    *,
    loss: str = "logistic",
    c: float = 1.0,
    n_pca: int | None = None,
    standardize: bool = True,
    class_weight: str | dict | None = None,
) -> Pipeline:
    """Build the linear-probe pipeline: ``[StandardScaler?] → [PCA?] → {logistic, ridge}``.

    L2 penalty only (issue #314 locks the penalty type and sweeps its strength).
    ``c`` is a unified *inverse* regularization strength — larger ``c`` = weaker
    regularization — mapped to logistic's ``C`` and to ridge's ``alpha = 1/c``.
    ``standardize`` and ``n_pca`` toggle the optional preprocessing steps that
    issue #314 evaluates (``StandardScaler`` on/off; PCA-k as a dimensionality
    lens). ``class_weight`` (logistic only) is the minority-reweighting knob — e.g.
    ``"balanced"`` up-weights positives by inverse frequency; it mostly shifts the
    intercept (AUPRC-invisible) but under heavy L2 also nudges the slope/ranking,
    so issue #314 treats it as an ablation axis, not a default. Ignored for ridge
    (regression has no class weights).
    """
    steps: list[tuple[str, object]] = []
    if standardize:
        steps.append(("scaler", StandardScaler()))
    if n_pca is not None:
        steps.append(("pca", PCA(n_components=n_pca, random_state=0)))
    if loss == "logistic":
        # l1_ratio=0.0 is the sklearn 1.8+ idiom for a pure-L2 penalty (the
        # deprecated `penalty="l2"`); issue #314 locks the penalty to L2.
        steps.append((
            "clf",
            LogisticRegression(
                l1_ratio=0.0, C=c, max_iter=2000, class_weight=class_weight
            ),
        ))
    elif loss == "ridge":
        steps.append(("clf", Ridge(alpha=1.0 / c)))
    else:
        raise ValueError(f"unknown loss {loss!r}; expected 'logistic' or 'ridge'")
    return Pipeline(steps)


def chrom_grouped_oof(
    features: np.ndarray,
    label: np.ndarray,
    groups: np.ndarray,
    *,
    loss: str = "logistic",
    c: float = 1.0,
    n_pca: int | None = None,
    standardize: bool = True,
    n_splits: int = 5,
    class_weight: str | dict | None = None,
) -> np.ndarray:
    """Out-of-fold probe scores via ``GroupKFold`` on ``groups`` (e.g. ``chrom``).

    Holds out whole groups each fold — a chromosome (hence all its genes) for the
    leak-proof across-gene split. Fits a fresh probe on the training groups and
    predicts the held-out ones; returns OOF scores aligned to ``features`` rows.
    Logistic → ``predict_proba[:, 1]``; ridge → ``predict``. PCA-k is clipped to
    the per-fold ``min(n_pca, n_features, n_train − 1)``.

    Folds = ``min(n_splits, #groups)``; every row is held out exactly once.
    """
    features = np.asarray(features, dtype=np.float32)
    label = np.asarray(label)
    groups = np.asarray(groups)
    n = len(label)
    assert features.ndim == 2 and features.shape[0] == n == len(groups), (
        features.shape,
        n,
        len(groups),
    )
    n_groups = len(np.unique(groups))
    k = min(n_splits, n_groups)
    assert k >= 2, f"need >=2 groups for CV, got {n_groups}"
    oof = np.full(n, np.nan, dtype=float)
    for tr, te in GroupKFold(n_splits=k).split(features, label, groups):
        n_pca_eff = (
            None if n_pca is None else min(n_pca, features.shape[1], len(tr) - 1)
        )
        probe = make_linear_probe(
            loss=loss, c=c, n_pca=n_pca_eff, standardize=standardize,
            class_weight=class_weight,
        )
        probe.fit(features[tr], label[tr])
        if hasattr(probe, "predict_proba"):
            oof[te] = probe.predict_proba(features[te])[:, 1]
        else:
            oof[te] = probe.predict(features[te])
    assert not np.isnan(oof).any(), "some rows were never held out — check groups"
    return oof


def probe_auprc(
    label: np.ndarray,
    oof_scores: np.ndarray,
    match_group: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    rng: int | None = 0,
) -> dict[str, float | int]:
    """AUPRC + cluster-bootstrap SE of OOF probe scores.

    Thin wrapper over ``metrics.auprc_with_bootstrap_se`` (resamples
    ``match_group`` to respect the matched-pair structure). Returns
    ``{"value", "se", "n_groups", "n_rows"}``.
    """
    return auprc_with_bootstrap_se(
        pd.Series(np.asarray(label)),
        pd.Series(np.asarray(oof_scores, dtype=float)),
        pd.Series(np.asarray(match_group)),
        n_bootstrap=n_bootstrap,
        rng=rng,
    )
