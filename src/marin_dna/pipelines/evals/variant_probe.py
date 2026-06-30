"""Frozen-embedding linear probe for variant effect prediction (issues #314/#320).

Trains a leak-proof linear probe on the **in-bundle pooled embeddings** (the
``emb_ref``/``emb_alt`` columns the #318 score bundle writes per variant) and
emits, per consequence ``subset``:

- the **leave-one-chromosome-out (LOOC) predictions** for every variant
  (``run_subset_probes`` → ``probe_score``), consumed by the downstream metrics
  step; and
- the **fitted classifier** (all-data fit), serialized so it can be reused on
  other datasets.

This is the productionized form of #314's settled protocol — **one** approach,
no sweeps:

- **Feature.** ``pair_feature_from_bundle`` upcasts the f16 allele means to
  float32 (a cancellation guard, see #318) and combines them: ``concat_ref_delta
  = [ref, alt−ref]`` for directional datasets (mendelian, sge) or ``sum_absdiff =
  [ref+alt, |alt−ref|]`` for swap-invariant ones (complex, caqtl, dsqtl).
- **Probe.** A fixed ``StandardScaler → LogisticRegression(L2)`` pipeline; the
  only tuned knob is the L2 strength ``C``.
- **CV.** ``traitgym_nested_oof`` — outer leave-one-chromosome-out, inner
  ``GroupKFold`` ``GridSearchCV`` re-tuning ``C`` per fold (leakage-free, the
  TraitGym protocol). ``fit_full_classifier`` fits the reusable all-data probe
  with the same inner C-search.
- **C-edge diagnostic (verified).** ``summarize_selected_c`` records when the chosen
  ``C`` lands on a grid boundary and, given the all-data inner-CV curve, *verifies*
  the pin is saturated/flat (the interior neighbor gives the same AUPRC) rather than
  a truncation. High edge ⟹ the unregularized limit (the fit saturates; ``1e4`` ≈
  ``C→∞`` in ranking); low edge ⟹ the maximally-shrunk mean-difference direction (a
  *flat* region, not a constant predictor — AUPRC is rank-based, so even ‖w‖→0 keeps
  a ranking). ``truncation_risk`` fires only if a pinned edge is still improving,
  which the anchored grid avoids.

``ref``/``alt`` always denote the reference- and alternate-allele embeddings.
``concat_ref_delta``/``sum_absdiff`` are the production combos; ``delta``/
``abs_delta`` are the documented bare-effect fallbacks.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ref↔alt combinations of the pooled allele embeddings. The symmetric ones are
# invariant under swapping which allele is "ref" — the only valid features for
# datasets with no biological ref/alt direction (complex_traits, caqtl, dsqtl).
PAIR_COMBOS: tuple[str, ...] = (
    "delta",
    "concat",
    "concat_ref_delta",
    "abs_delta",
    "prod",
    "sum_absdiff",
)
SYMMETRIC_COMBOS: frozenset[str] = frozenset({"abs_delta", "prod", "sum_absdiff"})

# The fixed L2-strength grid, anchored at both regularization regimes so it can't
# truncate the optimum (no widening, no np.inf):
#   low  1e-12 — a practical heavy-reg floor. As C→0 the L2 fit shrinks to the
#                scale-free mean-difference direction (NOT a constant predictor —
#                AUPRC is rank-based, so the ranking persists), and the AUPRC-vs-C
#                curve goes flat well above 1e-12 for any realistic model; a low-edge
#                pin = the flat maximally-regularized solution (verified, not assumed).
#   high 1e4   — a saturation cap. Past ~1e2 the logistic fit stops changing, and
#                1e4 reproduces the unregularized (C→∞) *ranking* exactly (all AUPRC
#                sees) for separable and non-separable cells alike — so no np.inf is
#                needed; a high-edge pin means "wants minimal reg (saturated)".
# Both edge-pins are recorded per subset by ``summarize_selected_c`` as diagnostics.
DEFAULT_C_GRID: np.ndarray = np.logspace(-12, 4, 17)
_MAX_ITER: int = 2000


def pair_feature(ref: np.ndarray, alt: np.ndarray, combo: str) -> np.ndarray:
    """Combine pooled ref/alt embeddings ``[N, D]`` into a per-variant feature.

    ``delta``/``concat``/``concat_ref_delta`` are signed (ref↔alt direction
    matters); ``abs_delta``/``prod``/``sum_absdiff`` are invariant under a ref↔alt
    swap (see ``SYMMETRIC_COMBOS``). Returns ``[N, D]`` for
    ``delta``/``abs_delta``/``prod`` and ``[N, 2D]`` for
    ``concat``/``sum_absdiff``/``concat_ref_delta``.
    """
    assert ref.shape == alt.shape and ref.ndim == 2, (ref.shape, alt.shape)
    if combo == "delta":
        return alt - ref
    if combo == "concat":
        return np.concatenate([ref, alt], axis=1)
    if combo == "concat_ref_delta":
        # signed effect (alt−ref) + local context (ref). Spans the same space as
        # `concat` but differs under L2 — the directional analog of `sum_absdiff`.
        return np.concatenate([ref, alt - ref], axis=1)
    if combo == "abs_delta":
        return np.abs(alt - ref)
    if combo == "prod":
        return ref * alt
    if combo == "sum_absdiff":
        return np.concatenate([ref + alt, np.abs(alt - ref)], axis=1)
    raise ValueError(f"unknown combo {combo!r}; expected one of {PAIR_COMBOS}")


def pair_feature_from_bundle(df: pd.DataFrame, combo: str) -> np.ndarray:
    """Build the per-variant pair-feature from the in-bundle pooled embeddings.

    Reads the ``emb_ref``/``emb_alt`` list-columns (length-``D`` float16 vectors
    written by the #318 score bundle) and **upcasts to float32 before
    combining**: the ``delta``/``abs_delta`` subtraction of two near-equal allele
    means is a catastrophic-cancellation risk in f16 (see #318), so the upcast
    must precede ``pair_feature``, not follow it. Returns ``[N, D]`` or ``[N, 2D]``
    per ``combo``.
    """
    assert "emb_ref" in df.columns and "emb_alt" in df.columns, (
        "scores parquet lacks emb_ref/emb_alt columns — re-score with "
        "inference.return_embeddings=true (the #318 overlay)"
    )
    # Guard the empty frame: np.stack([]) raises an opaque "need at least one array
    # to stack" before the shape assert below could give a domain message.
    assert len(df) > 0, "empty frame — no variants to build a probe feature from"
    # f16 store -> f32 BEFORE the difference (cancellation guard).
    ref = np.stack(df["emb_ref"].to_numpy()).astype(np.float32)
    alt = np.stack(df["emb_alt"].to_numpy()).astype(np.float32)
    assert ref.shape == alt.shape and ref.ndim == 2, (ref.shape, alt.shape)
    assert np.isfinite(ref).all() and np.isfinite(alt).all(), "non-finite embeddings"
    return pair_feature(ref, alt, combo)


def _probe_pipeline() -> Pipeline:
    """The locked probe: ``StandardScaler → L2 LogisticRegression``.

    sklearn 1.8 defaults ``LogisticRegression`` to L2 (the ``penalty`` arg is
    mid-deprecation — passing ``penalty='l2'`` warns). ``l1_ratio`` only takes effect
    under elasticnet, so ``l1_ratio=0.0`` is a no-op here; it's passed to pin the
    intent to pure-L2 (and to guard against a future default drifting toward
    elasticnet). ``C`` is set per grid point by the caller's ``GridSearchCV``.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(l1_ratio=0.0, max_iter=_MAX_ITER)),
        ]
    )


def _inner_c_search(c_grid: np.ndarray, k: int, n_jobs: int) -> GridSearchCV:
    """The inner C-tuning search shared by the OOF folds and the all-data fit.

    Kept in one place so the leak-free OOF folds (``traitgym_nested_oof``) and the
    reusable all-data classifier (``fit_full_classifier``) tune ``C`` *identically* —
    same pipeline, same ``GroupKFold`` AUPRC inner CV. Diverging the two would break
    the leakage-equivalence the protocol relies on.
    """
    return GridSearchCV(
        _probe_pipeline(),
        {"clf__C": c_grid},
        cv=GroupKFold(n_splits=k),
        scoring="average_precision",
        n_jobs=n_jobs,
    )


def traitgym_nested_oof(
    features: np.ndarray,
    label: np.ndarray,
    groups: np.ndarray,
    *,
    c_grid: np.ndarray | None = None,
    inner_splits: int = 5,
    n_jobs: int = -1,
) -> tuple[np.ndarray, list[float]]:
    """Nested leave-one-group-out OOF — the TraitGym linear-probing protocol.

    **Outer:** leave-one-group-out over ``groups`` (a whole chromosome held out
    each fold). **Inner:** within each outer fold's training groups, ``GridSearchCV``
    over a ``GroupKFold`` of those groups tunes ``C`` by AUPRC — so ``C`` is re-tuned
    per fold (hence per model and adaptively to its feature dimensionality), with no
    leakage. Fixed pipeline ``StandardScaler → LogisticRegression`` (L2, no PCA).

    Returns ``(oof_scores, selected_Cs)`` — OOF ``predict_proba[:, 1]`` aligned to
    ``features`` rows, and the ``C`` chosen on each outer fold (feed to
    ``summarize_selected_c`` to confirm the grid isn't truncating the optimum).
    """
    if c_grid is None:
        c_grid = DEFAULT_C_GRID
    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(label).astype(int)
    g = np.asarray(groups)
    n = len(y)
    assert X.ndim == 2 and X.shape[0] == n == len(g), (X.shape, n, len(g))
    assert len(np.unique(g)) >= 3, "need >=3 groups for nested LOGO"
    oof = np.full(n, np.nan, dtype=float)
    selected: list[float] = []
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        k = min(inner_splits, len(np.unique(g[tr])))
        gs = _inner_c_search(c_grid, k, n_jobs)
        gs.fit(X[tr], y[tr], groups=g[tr])
        oof[te] = gs.predict_proba(X[te])[:, 1]
        selected.append(float(gs.best_params_["clf__C"]))
    assert not np.isnan(oof).any(), "some rows were never held out — check groups"
    return oof, selected


def fit_full_classifier(
    features: np.ndarray,
    label: np.ndarray,
    groups: np.ndarray,
    *,
    c_grid: np.ndarray | None = None,
    inner_splits: int = 5,
    n_jobs: int = -1,
) -> tuple[Pipeline, float, np.ndarray]:
    """Fit the reusable all-data probe, ``C`` chosen by one inner C-search.

    The same pipeline and inner ``GroupKFold`` ``GridSearchCV`` as the OOF folds,
    but fit on **all** the variants (no held-out chromosome) — this is the
    classifier serialized for cross-dataset reuse. Returns ``(fitted_pipeline,
    selected_C, c_scores)`` (``GridSearchCV`` refits ``best_estimator_`` on the full
    data). ``c_scores`` is the inner-CV mean AUPRC per grid ``C``, **ascending in
    C**, used by ``summarize_selected_c`` to verify a pinned edge is saturated (the
    interior neighbor gives the same result), not a truncation.
    """
    if c_grid is None:
        c_grid = DEFAULT_C_GRID
    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(label).astype(int)
    g = np.asarray(groups)
    n = len(y)
    assert X.ndim == 2 and X.shape[0] == n == len(g), (X.shape, n, len(g))
    k = min(inner_splits, len(np.unique(g)))
    assert k >= 2, f"need >=2 groups for inner C-search, got {len(np.unique(g))}"
    gs = _inner_c_search(c_grid, k, n_jobs)
    gs.fit(X, y, groups=g)
    # Inner-CV mean AUPRC per grid C, ordered ascending in C so c_scores[0] is the
    # heavy-reg floor and c_scores[-1] the weak-reg cap (the edge-saturation check).
    grid_c = np.asarray(gs.cv_results_["param_clf__C"], dtype=float)
    c_scores = np.asarray(gs.cv_results_["mean_test_score"], dtype=float)[
        np.argsort(grid_c)
    ]
    return gs.best_estimator_, float(gs.best_params_["clf__C"]), c_scores


def summarize_selected_c(
    selected_cs: list[float] | np.ndarray,
    full_c: float,
    c_grid: np.ndarray,
    *,
    full_c_scores: np.ndarray | None = None,
    tol: float = 0.01,
) -> dict[str, float | int | bool | None]:
    """Summarize the chosen ``C``s, flag grid-edge pinning, and verify saturation.

    ``GridSearchCV.best_params_`` returns an exact grid value, so edge detection is
    exact equality with the grid's min/max. ``at_edge`` is true when any outer fold
    *or* the shipped ``full_c`` lands on a boundary.

    A pin is benign as long as the edge is **saturated/flat** — i.e. the interior
    neighbor gives the same result, so there is nothing beyond the grid to miss
    (high edge ⟹ the unregularized limit; low edge ⟹ the maximally-shrunk
    mean-difference direction). When ``full_c_scores`` (the all-data inner-CV mean
    AUPRC per grid ``C``, **ascending in C**) is given, this is *verified* rather
    than assumed: ``high_edge_gain = c_scores[-1] − c_scores[-2]`` and
    ``low_edge_gain = c_scores[0] − c_scores[1]`` measure how much better each edge
    is than its interior neighbor, and ``truncation_risk`` is set only when a
    *pinned* edge is still improving by more than ``tol`` AUPRC (the optimum may lie
    outside the grid → widen). With the grid anchored at both regularization limits
    this should not fire; the check makes that auditable per cell.
    """
    sel = np.asarray(selected_cs, dtype=float)
    grid = np.sort(np.asarray(c_grid, dtype=float))
    assert sel.ndim == 1 and sel.size >= 1, sel.shape
    assert grid.size >= 2, f"c_grid needs >=2 points for edge logic, got {grid.size}"
    lo, hi = float(grid[0]), float(grid[-1])
    n_low = int(np.sum(sel == lo))
    n_high = int(np.sum(sel == hi))
    full_at_edge = bool(full_c == lo or full_c == hi)
    out: dict[str, float | int | bool | None] = {
        "c_min": float(sel.min()),
        "c_med": float(np.median(sel)),
        "c_max": float(sel.max()),
        "full_c": float(full_c),
        "n_at_low_edge": n_low,
        "n_at_high_edge": n_high,
        "full_at_edge": full_at_edge,
        "at_edge": bool(n_low > 0 or n_high > 0 or full_at_edge),
        "high_edge_gain": None,
        "low_edge_gain": None,
        "truncation_risk": None,
    }
    if full_c_scores is not None:
        s = np.asarray(full_c_scores, dtype=float)
        assert s.shape == grid.shape, (s.shape, grid.shape)
        high_gain = float(s[-1] - s[-2])  # >tol ⟹ still rising at the weak-reg cap
        low_gain = float(s[0] - s[1])  # >tol ⟹ wants heavier reg than the floor
        high_pinned = n_high > 0 or full_c == hi
        low_pinned = n_low > 0 or full_c == lo
        out["high_edge_gain"] = high_gain
        out["low_edge_gain"] = low_gain
        # A pinned edge is risky if it's still improving past tol — OR if its gain is
        # NaN (a failed/single-class inner-CV fold left mean_test_score NaN): we then
        # could NOT verify saturation, so we must not silently report "benign"
        # (`nan > tol` is False, which would). Flag it instead.
        high_risky = high_pinned and (np.isnan(high_gain) or high_gain > tol)
        low_risky = low_pinned and (np.isnan(low_gain) or low_gain > tol)
        out["truncation_risk"] = bool(high_risky or low_risky)
    return out


def run_subset_probes(
    df: pd.DataFrame,
    *,
    feature_combo: str,
    c_grid: np.ndarray | None = None,
    min_variants: int = 300,
    min_chroms: int = 3,
    inner_splits: int = 5,
    n_jobs: int = -1,
) -> tuple[pd.DataFrame, dict]:
    """Train a per-subset nested-LOOC probe over an in-bundle scores frame.

    For each consequence ``subset`` (or a single synthetic ``"all"`` group when the
    frame has no ``subset`` column, e.g. caqtl/dsqtl) that clears ``min_variants``
    *and* ``min_chroms``: run ``traitgym_nested_oof`` for the LOOC ``probe_score``
    and ``fit_full_classifier`` for the reusable classifier. Subsets below the
    threshold (or that error / are single-class) are skipped — their rows keep a
    ``NaN`` ``probe_score`` and get no classifier.

    Returns ``(predictions, classifiers)``:

    - ``predictions`` — ``df`` minus the bulky ``emb_ref``/``emb_alt`` columns, plus
      a ``probe_score`` column (the LOOC OOF prediction; ``NaN`` for skipped rows).
    - ``classifiers`` — ``{subset: {"pipeline", "C", "feature", "n", "n_pos",
      "c_summary"}}`` (``c_summary`` is ``summarize_selected_c``).
    """
    if c_grid is None:
        c_grid = DEFAULT_C_GRID
    c_grid = np.asarray(c_grid, dtype=float)
    assert "chrom" in df.columns and "label" in df.columns, (
        "frame needs `chrom` and `label` columns"
    )
    feat = pair_feature_from_bundle(df, feature_combo)  # [N, F] float32
    n_rows = len(df)
    assert feat.shape[0] == n_rows, (feat.shape, n_rows)
    chrom = df["chrom"].astype(str).to_numpy()
    label = df["label"].to_numpy().astype(int)
    # No biological consequence subsets on the QTL datasets -> one "all" group.
    subset = (
        df["subset"].astype(str).to_numpy()
        if "subset" in df.columns
        else np.full(n_rows, "all")
    )

    oof = np.full(n_rows, np.nan, dtype=float)
    classifiers: dict = {}
    for s in sorted(set(subset)):
        mask = subset == s
        n = int(mask.sum())
        n_chrom = len(set(chrom[mask]))
        n_pos = int(label[mask].sum())
        if n < min_variants or n_chrom < min_chroms:
            print(
                f"[probe] skip {s!r}: n={n} (min {min_variants}), "
                f"n_chrom={n_chrom} (min {min_chroms})"
            )
            continue
        if not 0 < n_pos < n:
            print(f"[probe] skip {s!r}: single-class (n_pos={n_pos}/{n})")
            continue
        try:
            oof_s, selected = traitgym_nested_oof(
                feat[mask],
                label[mask],
                chrom[mask],
                c_grid=c_grid,
                inner_splits=inner_splits,
                n_jobs=n_jobs,
            )
            clf, full_c, c_scores = fit_full_classifier(
                feat[mask],
                label[mask],
                chrom[mask],
                c_grid=c_grid,
                inner_splits=inner_splits,
                n_jobs=n_jobs,
            )
        except Exception as e:  # keep an unattended multi-subset run alive
            print(f"[probe] SKIP {s!r}: {type(e).__name__}: {e}")
            continue
        oof[mask] = oof_s
        c_summary = summarize_selected_c(
            selected, full_c, c_grid, full_c_scores=c_scores
        )
        classifiers[s] = {
            "pipeline": clf,
            "C": full_c,
            "feature": feature_combo,
            "n": n,
            "n_pos": n_pos,
            "c_summary": c_summary,
        }
        print(
            f"[probe] {s}: n={n} n_pos={n_pos} C med={c_summary['c_med']:.1e} "
            f"[{c_summary['c_min']:.0e},{c_summary['c_max']:.0e}] full={full_c:.1e}"
        )
        if c_summary["at_edge"]:
            # A pinned edge is benign when it's saturated/flat (the interior neighbor
            # gives the same result — verified via the inner-CV gains). truncation_risk
            # fires only if a pinned edge is still improving > tol (optimum may be
            # outside the grid). With both ends at the reg limits it should not fire.
            verdict = (
                "TRUNCATION RISK — edge still improving, widen c_grid"
                if c_summary["truncation_risk"]
                else "saturated/flat (benign)"
            )
            print(
                f"[probe] note {s!r}: C pinned grid edge "
                f"(n_low={c_summary['n_at_low_edge']}, "
                f"n_high={c_summary['n_at_high_edge']}, full_c={c_summary['full_c']:.0e}; "
                f"gain low={c_summary['low_edge_gain']:+.3f} "
                f"high={c_summary['high_edge_gain']:+.3f}) — {verdict}"
            )

    assert classifiers, (
        f"no subset met the threshold (min_variants={min_variants}, "
        f"min_chroms={min_chroms}) — check the dataset / lower the threshold"
    )
    keep = [c for c in df.columns if c not in ("emb_ref", "emb_alt")]
    predictions = df[keep].copy()
    predictions["probe_score"] = oof
    return predictions, classifiers
