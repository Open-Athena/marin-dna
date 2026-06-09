"""Metric utilities for variant-effect evaluations.

Four metric families live here:

- ``auprc_with_bootstrap_se`` / ``compute_auprc_metrics``: AUPRC with a
  cluster bootstrap SE that resamples ``match_group``s (preserving the
  matched-pair clustering). Used by the ``evals_v2``, ``conservation_eval``,
  and ``alphagenome_eval`` pipelines on the matched-pair datasets
  (``mendelian_traits`` / ``complex_traits``), whose 1:k structure (PR #194)
  the Wald-binomial pairwise metric can't represent.
- ``compute_qtl_metrics``: global AUPRC (plain row bootstrap) plus Pearson /
  Spearman of the model score vs ``effect_size`` over the positive variants
  only. For the unmatched DART-Eval QTL datasets (``caqtl`` / ``dsqtl``,
  PR #214) that carry no ``subset`` / ``match_group``. The same three
  pipelines select this path per-dataset via ``eval_protocol: qtl_global``.
- ``pairwise_accuracy`` / ``compute_pairwise_metrics``: matched-pair within-
  ``match_group`` accuracy (ties = 0.5) with Wald-binomial SE. Used on 1:1
  match groups by the ``gpn_star_eval`` pipeline, the ``scripts/evo2_eval/``
  scripts, and the ``lm_eval`` DNA-VEP harness (``dna_vep_llr_eval``).
- ``METRIC_FUNCTIONS`` / ``compute_metrics``: classical AUPRC / AUROC /
  Spearman over (label, score) pairs. Still used by older pipelines
  (``snakemake/analysis/evals_v1/``, ``scripts/evo2_eval/``).
"""

import math
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


METRIC_FUNCTIONS: dict[str, Callable[[pd.Series, pd.Series], float]] = {
    "AUPRC": lambda label, score: average_precision_score(label, score),
    "AUROC": lambda label, score: roc_auc_score(label, score),
    "Spearman": lambda label, score: spearmanr(label, score)[0],
}


def pairwise_accuracy(
    label: pd.Series,
    score: pd.Series,
    match_group: pd.Series,
) -> dict[str, float | int]:
    """Within-``match_group`` accuracy: fraction of pairs where the positive
    scores higher than the negative (ties = 0.5).

    Asserts each ``match_group`` has exactly one positive and one negative.
    Do **not** call this on the new 1:k matched-pair datasets from PR #194
    — the assertion will fire and the Wald SE assumes paired comparisons.
    Use ``auprc_with_bootstrap_se`` on those instead. This function is
    retained for the ``conservation_eval`` pipeline, which still produces
    1:1 datasets.

    Args:
        label: 0/1 (or bool) per row. Cast to int internally.
        score: numeric score per row. **Must not contain NaN** — fill
            upstream with a semantically appropriate value (the
            ``conservation_eval`` pipeline does ``.fillna(0)``). Without
            this rule, a NaN-vs-NaN pair would silently count as a loss
            for the positive (since ``NaN > NaN`` and ``NaN == NaN`` are
            both False) — that's exactly the kind of silent-corruption
            risk we want to fail loud on.
        match_group: integer group id; positives and negatives are paired
            within a group.

    Returns:
        ``{"value", "se", "n_pairs", "n_ties"}``. ``se`` is the Wald binomial
        form ``sqrt(value * (1 - value) / n_pairs)``.
    """
    assert len(label) == len(score) == len(match_group), (
        f"length mismatch: label={len(label)} score={len(score)} "
        f"match_group={len(match_group)}"
    )

    label_int = pd.Series(label).astype(int).reset_index(drop=True)
    score_arr = pd.Series(score).reset_index(drop=True)
    mg = pd.Series(match_group).reset_index(drop=True)
    assert not score_arr.isna().any(), (
        f"score has {int(score_arr.isna().sum())} NaN values; fill upstream "
        f"with a semantically appropriate default before scoring"
    )

    df = pd.DataFrame({"label": label_int, "score": score_arr, "match_group": mg})

    # Each group must have exactly 1 pos + 1 neg.
    counts = df.groupby("match_group")["label"].agg(["sum", "count"])
    bad = counts[(counts["sum"] != 1) | (counts["count"] != 2)]
    assert bad.empty, (
        f"pairwise_accuracy expects exactly 1 positive + 1 negative per "
        f"match_group, got {len(bad)} bad groups; first: {bad.head().to_dict()}"
    )

    pos = df[df["label"] == 1].set_index("match_group")["score"].sort_index()
    neg = df[df["label"] == 0].set_index("match_group")["score"].sort_index()
    assert pos.index.equals(neg.index), "positive/negative match_group sets differ"

    diff = pos.values - neg.values
    n = len(diff)
    wins = int((diff > 0).sum())
    ties = int((diff == 0).sum())
    value = (wins + 0.5 * ties) / n
    se = math.sqrt(value * (1 - value) / n)
    return {"value": float(value), "se": float(se), "n_pairs": int(n), "n_ties": ties}


GLOBAL_SUBSET = "_global_"
MACRO_AVG_SUBSET = "_macro_avg_"

# Derived SGE subset scope. The dataset's real ``subset`` values are
# ``missense_variant`` / ``splicing``; ``SGE_POOLED_SUBSET`` is the two pooled
# within an accession ("both"), while ``MACRO_AVG_SUBSET`` (reused) is the
# equal-weight mean of the base-subset values. Both the ``subset`` and the
# ``accession`` axes use ``MACRO_AVG_SUBSET`` for their macro rows.
SGE_POOLED_SUBSET = "both"

# Per-strand LLR → score-protocol transforms. Keyed by the
# ``score_protocol`` field in `snakemake/analysis/evals_v2/config/config.yaml`
# and consumed by `metrics.smk` to materialize `{protocol}_{fwd,rc,avg}`
# columns from the raw `llr_*` atoms before AUPRC. Single source of truth
# so a typo in config fails loud with `KeyError` instead of silently
# producing wrong scores.
SCORE_PROTOCOLS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "minus_llr": lambda x: -x,
    "abs_llr": np.abs,
}


def auprc_with_bootstrap_se(
    label: pd.Series,
    score: pd.Series,
    match_group: pd.Series,
    *,
    n_bootstrap: int = 1000,
    rng: np.random.Generator | int | None = 0,
) -> dict[str, float | int]:
    """AUPRC + cluster-bootstrap SE over ``match_group``.

    The matched-pair structure (each positive shares a ``match_group``
    with its k matched negatives) means rows are not iid. Naive row-
    bootstrap would resample positives and negatives independently and
    inflate SE. We do a *cluster bootstrap*: each iteration resamples
    the unique ``match_group`` IDs with replacement, gathers all rows
    belonging to the sampled groups (with multiplicity), and computes
    AUPRC on the resampled set. SE = std of the bootstrap distribution.

    The ``rng`` default is ``0`` (not ``None``) so that pipeline outputs
    are bit-stable across re-runs on identical inputs — important for
    snakemake's metadata-aware re-trigger logic and for diffing
    leaderboard rows. Pass ``rng=None`` for fresh randomness in ad-hoc
    analysis. SE is still a sample estimate; the seed pins the sample,
    not the truth.

    Args:
        label: 0/1 (or bool) per row.
        score: numeric score per row. Must not contain NaN — fill
            upstream (same rationale as ``pairwise_accuracy``).
        match_group: integer group id; cluster bootstrap unit.
        n_bootstrap: number of bootstrap iterations. ``<= 0`` skips the
            resample loop and returns ``se=NaN`` (point estimate only) — used on
            the online in-training hot path, where the SE is computed offline.
        rng: ``numpy.random.Generator``, seed int, or ``None``.

    Returns:
        ``{"value", "se", "n_groups", "n_rows"}``. ``value`` is the
        point-estimate AUPRC over all input rows; ``se`` is the std of
        the bootstrap distribution (``ddof=1``, NaN-tolerant for
        degenerate resamples), or ``NaN`` when ``n_bootstrap <= 0``.
    """
    assert len(label) == len(score) == len(match_group), (
        f"length mismatch: label={len(label)} score={len(score)} "
        f"match_group={len(match_group)}"
    )
    score_arr = np.asarray(score, dtype=float)
    label_arr = np.asarray(label).astype(int)
    mg_arr = np.asarray(match_group)
    assert not np.isnan(score_arr).any(), (
        f"score has {int(np.isnan(score_arr).sum())} NaN values; fill "
        f"upstream with a semantically appropriate default before scoring"
    )
    n_pos = int(label_arr.sum())
    assert 0 < n_pos < len(label_arr), (
        f"AUPRC undefined: need both classes, got n_pos={n_pos} of n={len(label_arr)}"
    )

    point = float(average_precision_score(label_arr, score_arr))

    # `groupby(mg_arr).indices` is an O(n) hash-based group → positional-
    # index map. The earlier `[np.where(inv == i)[0] for i in groups]`
    # form was O(n_groups · n_rows) and dominated the non-AP cost when
    # n_groups was ~10³ within a subset. Computed even for the point-only
    # path below — `n_groups` feeds the macro-average n_min gate.
    group_to_rows: list[np.ndarray] = list(
        pd.Series(mg_arr).groupby(mg_arr).indices.values()
    )
    n_groups = len(group_to_rows)

    if n_bootstrap <= 0:
        # Point estimate only — skip the resample loop. Used on the in-training
        # lm_eval hot path (online VEP), where a per-eval-step bootstrap is too
        # slow and the SE is redundant with the offline evals_v2 parquet.
        se = float("nan")
    else:
        rng = np.random.default_rng(rng)
        boot = np.empty(n_bootstrap, dtype=float)
        for b in range(n_bootstrap):
            sampled = rng.integers(0, n_groups, size=n_groups)
            idx = np.concatenate([group_to_rows[i] for i in sampled])
            y = label_arr[idx]
            # Rare degenerate resamples may be single-class — AUPRC undefined.
            s = int(y.sum())
            if s == 0 or s == len(y):
                boot[b] = np.nan
                continue
            boot[b] = average_precision_score(y, score_arr[idx])
        se = float(np.nanstd(boot, ddof=1))
    return {
        "value": point,
        "se": se,
        "n_groups": int(n_groups),
        "n_rows": int(len(label_arr)),
    }


def paired_metric_delta_bootstrap(
    label: pd.Series,
    score_a: pd.Series,
    score_b: pd.Series,
    match_group: pd.Series,
    *,
    n_bootstrap: int = 1000,
    rng: np.random.Generator | int | None = 0,
) -> dict[str, float | int]:
    """Paired cluster-bootstrap of the AUPRC delta ``AP(a) − AP(b)`` on shared rows.

    Both score columns are scored on the SAME rows (shared ``label`` and
    ``match_group``), so each bootstrap iteration resamples ``match_group``s ONCE
    and recomputes AUPRC for *both* scores on that single resample. The resulting
    delta distribution captures the positive cross-score correlation, so its SE is
    the paired form ``sqrt(SE_a² + SE_b² − 2 ρ SE_a SE_b)`` — tighter than the
    independence formula ``sqrt(SE_a² + SE_b²)``, which over-states the delta's
    uncertainty and can bury real effects. Use for any "model A vs model B on the
    same matched-pair eval" comparison.

    Args:
        label: 0/1 per row (shared by both scores).
        score_a: score for arm A (delta is A − B); no NaN.
        score_b: score for arm B (the baseline); no NaN.
        match_group: integer group id; the cluster-bootstrap unit.
        n_bootstrap: bootstrap iterations.
        rng: ``numpy.random.Generator``, seed int, or ``None`` (default ``0`` →
            reproducible across re-runs).

    Returns:
        ``{"delta", "se", "ci_low", "ci_high", "p_two_sided", "n_groups",
        "n_rows"}``. ``delta`` is the point ``AP(a) − AP(b)`` over all rows; ``se``
        is the std of the bootstrap deltas; ``ci_low``/``ci_high`` the 2.5/97.5
        percentiles; ``p_two_sided = 2·min(frac≤0, frac≥0)`` (ties count on both
        sides), clamped to ``[1/n_bootstrap, 1]``.
    """
    assert len(label) == len(score_a) == len(score_b) == len(match_group), (
        f"length mismatch: label={len(label)} a={len(score_a)} "
        f"b={len(score_b)} match_group={len(match_group)}"
    )
    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)
    y = np.asarray(label).astype(int)
    mg = np.asarray(match_group)
    assert not np.isnan(a).any() and not np.isnan(b).any(), (
        "scores contain NaN; fill upstream before scoring"
    )
    n_pos = int(y.sum())
    assert 0 < n_pos < len(y), f"AUPRC undefined: n_pos={n_pos} of n={len(y)}"

    point = float(average_precision_score(y, a) - average_precision_score(y, b))
    group_to_rows: list[np.ndarray] = list(pd.Series(mg).groupby(mg).indices.values())
    n_groups = len(group_to_rows)

    rng = np.random.default_rng(rng)
    boot = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sampled = rng.integers(0, n_groups, size=n_groups)
        idx = np.concatenate([group_to_rows[g] for g in sampled])
        yy = y[idx]
        s = int(yy.sum())
        if s == 0 or s == len(yy):  # single-class resample — delta undefined
            boot[i] = np.nan
            continue
        boot[i] = average_precision_score(yy, a[idx]) - average_precision_score(
            yy, b[idx]
        )
    boot = boot[~np.isnan(boot)]
    if boot.size == 0:
        # Every resample was single-class (only possible on non-matched-pair input,
        # where a group can be all-one-label) — the delta SE/CI/p are undefined.
        nan = float("nan")
        return {
            "delta": point,
            "se": nan,
            "ci_low": nan,
            "ci_high": nan,
            "p_two_sided": nan,
            "n_groups": int(n_groups),
            "n_rows": int(len(y)),
        }
    se = float(np.std(boot, ddof=1))
    lo, hi = (float(v) for v in np.percentile(boot, [2.5, 97.5]))
    # Two-sided bootstrap p. Ties (boot == 0) count on BOTH sides, so identical
    # scores give p≈1 rather than a spurious 0; clamp to [1/n_bootstrap, 1].
    p = min(2.0 * min(float((boot <= 0).mean()), float((boot >= 0).mean())), 1.0)
    p = max(p, 1.0 / n_bootstrap)
    return {
        "delta": point,
        "se": se,
        "ci_low": lo,
        "ci_high": hi,
        "p_two_sided": p,
        "n_groups": int(n_groups),
        "n_rows": int(len(y)),
    }


def compute_auprc_metrics(
    dataset: pd.DataFrame,
    scores: pd.DataFrame,
    score_columns: list[str] | None = None,
    *,
    n_bootstrap: int = 1000,
    rng: np.random.Generator | int | None = 0,
    n_min: int = 30,
) -> pd.DataFrame:
    """AUPRC + cluster-bootstrap SE per ``subset`` for one or more score columns.

    Mirrors ``compute_pairwise_metrics`` structure: per-subset rows
    plus ``_global_`` and ``_macro_avg_`` aggregates per score column.

    Aggregate semantics:

    - **Per-subset**: AUPRC on rows in that subset; cluster bootstrap on
      ``match_group``s within the subset.
    - **``_global_``**: AUPRC over all rows; cluster bootstrap on all
      ``match_group``s.
    - **``_macro_avg_``**: unweighted mean of per-subset values for
      subsets with ``n_groups >= n_min``. SE via the SE-of-mean formula
      ``sqrt(sum(SE_s^2)) / K`` over qualifying subsets — same form the
      pairwise pipeline uses (``compute_pairwise_metrics``); the
      bootstrap unit is the same (groups), so the formula is consistent.
      ``n_groups`` on the macro row is repurposed to record K. The
      independence assumption that justifies SE-of-mean is satisfied
      because each subset is bootstrapped over a disjoint
      ``match_group`` set; the same ``rng`` seed across subsets adds a
      shared PRNG-stream coupling but doesn't break the disjoint-data
      independence, only correlates the per-iteration noise.

    Asserts no ``match_group`` straddles subsets (same as
    ``compute_pairwise_metrics``).

    Args:
        dataset: DataFrame with columns ``[label, subset, match_group]``.
        scores: DataFrame whose columns are model scores; row-aligned
            with ``dataset``.
        score_columns: Score column names to evaluate. Defaults to all
            columns of ``scores``.
        n_bootstrap: bootstrap iterations per (subset, score_column).
        rng: ``numpy.random.Generator``, seed int, or ``None`` —
            forwarded to ``auprc_with_bootstrap_se``. ``0`` (default) →
            reproducible across re-runs; ``None`` → fresh randomness.
        n_min: minimum ``n_groups`` per subset to qualify for the
            macro average. Default 30 (project-wide convention for the
            leaderboard issues).

    Returns:
        DataFrame with columns
        ``[score_type, subset, value, se, n_groups, n_rows]``.
    """
    for col in ("label", "subset", "match_group"):
        assert col in dataset.columns, f"dataset missing required column {col!r}"

    if score_columns is None:
        score_columns = list(scores.columns)

    merged = pd.concat(
        [dataset.reset_index(drop=True), scores.reset_index(drop=True)], axis=1
    )

    # No match_group may straddle subsets.
    subset_per_group = merged.groupby("match_group")["subset"].nunique()
    bad_groups = subset_per_group[subset_per_group > 1]
    assert bad_groups.empty, (
        f"{len(bad_groups)} match_group(s) span multiple subsets; first: "
        f"{bad_groups.head().to_dict()}"
    )

    rows: list[dict] = []
    per_score_rows: dict[str, list[dict]] = {sc: [] for sc in score_columns}
    for subset_name, subset_df in merged.groupby("subset", sort=False):
        for score_col in score_columns:
            res = auprc_with_bootstrap_se(
                label=subset_df["label"],
                score=subset_df[score_col],
                match_group=subset_df["match_group"],
                n_bootstrap=n_bootstrap,
                rng=rng,
            )
            row = {
                "score_type": score_col,
                "subset": str(subset_name),
                "value": res["value"],
                "se": res["se"],
                "n_groups": res["n_groups"],
                "n_rows": res["n_rows"],
            }
            rows.append(row)
            per_score_rows[score_col].append(row)

    for score_col in score_columns:
        global_res = auprc_with_bootstrap_se(
            label=merged["label"],
            score=merged[score_col],
            match_group=merged["match_group"],
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
        rows.append(
            {
                "score_type": score_col,
                "subset": GLOBAL_SUBSET,
                "value": global_res["value"],
                "se": global_res["se"],
                "n_groups": global_res["n_groups"],
                "n_rows": global_res["n_rows"],
            }
        )

        qualifying = [r for r in per_score_rows[score_col] if r["n_groups"] >= n_min]
        assert qualifying, (
            f"no subsets meet n_min={n_min} for score_type={score_col!r}; "
            f"per-subset sizes: "
            f"{ {r['subset']: r['n_groups'] for r in per_score_rows[score_col]} }"
        )
        k = len(qualifying)
        macro_value = sum(r["value"] for r in qualifying) / k
        macro_se = math.sqrt(sum(r["se"] ** 2 for r in qualifying)) / k
        rows.append(
            {
                "score_type": score_col,
                "subset": MACRO_AVG_SUBSET,
                "value": float(macro_value),
                "se": float(macro_se),
                "n_groups": k,
                "n_rows": sum(r["n_rows"] for r in qualifying),
            }
        )

    return pd.DataFrame(rows)


def _correlation_with_bootstrap_se(
    x: np.ndarray,
    y: np.ndarray,
    *,
    method: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Point correlation + row-bootstrap SE between paired ``x`` and ``y``.

    Args:
        x: first variable (e.g. model score over positives).
        y: second variable (e.g. ``effect_size`` over positives).
        method: ``"pearson"`` or ``"spearman"``.
        n_bootstrap: bootstrap iterations (resamples rows with replacement).
        rng: ``numpy.random.Generator`` — threaded in so the caller controls
            reproducibility.

    Returns:
        ``(value, se)``. ``se`` is the std (``ddof=1``, NaN-tolerant) of the
        bootstrap distribution; degenerate resamples with zero variance in
        either variable contribute NaN and are dropped from the std.
    """
    assert method in ("pearson", "spearman"), f"unknown method {method!r}"
    assert len(x) == len(y), f"length mismatch: x={len(x)} y={len(y)}"
    corr_fn = pearsonr if method == "pearson" else spearmanr
    point = float(corr_fn(x, y)[0])
    n = len(x)
    boot = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        # Correlation is undefined when a resample is constant in either var.
        if np.std(xb) == 0 or np.std(yb) == 0:
            boot[b] = np.nan
            continue
        boot[b] = corr_fn(xb, yb)[0]
    se = float(np.nanstd(boot, ddof=1))
    return point, se


def compute_qtl_metrics(
    dataset: pd.DataFrame,
    scores: pd.DataFrame,
    score_columns: list[str] | None = None,
    *,
    n_bootstrap: int = 1000,
    rng: np.random.Generator | int | None = 0,
) -> pd.DataFrame:
    """Global AUPRC + positives-only ``effect_size`` correlations per score column.

    For the unmatched DART-Eval QTL datasets (``caqtl`` / ``dsqtl``), which
    have no ``subset`` / ``match_group`` — so there is no per-subset
    stratification and no cluster structure. Two metric families, mirroring
    DART-Eval Task 5 (and the dataset cards):

    - **AUPRC** over **all** rows (significant QTL vs control via ``label``).
      Computed through ``auprc_with_bootstrap_se`` with singleton
      ``match_group``s (``np.arange(n)``), which degenerates the cluster
      bootstrap to a plain row bootstrap — a consistent point estimate + SE
      with no separate code path.
    - **Pearson** and **Spearman** between each model score and
      ``effect_size`` (the unsigned ``|effect|`` magnitude), over the
      **positive variants only** (``label == True``). Controls are excluded
      — for ``dsqtl`` they carry no measured effect at all. Each gets a
      paired row-bootstrap SE over the positive rows (the same resampled
      indices draw both the score and ``effect_size``).

    Standard errors are **nonparametric bootstrap** — the std of the bootstrap
    distribution (``ddof=1``, NaN-tolerant) over ``n_bootstrap`` resamples —
    *not* closed-form. Rationale: AUPRC has no clean closed form, and the
    bootstrap keeps all three SEs uniform and assumption-free. The closed-form
    alternatives (Pearson Fisher-z = ``.confidence_interval()`` on
    ``scipy.stats.pearsonr``; Spearman has no built-in CI/SE, only the
    Bonett–Wright approximation) lean on a bivariate-normal / large-n
    assumption that ``effect_size`` (a right-skewed ``|effect|`` magnitude) and
    the scores don't meet — though at these sample sizes they land close to the
    bootstrap regardless (e.g. caqtl Pearson SE ≈ 0.017 either way). **Caveat:**
    these are *marginal, single-scorer* SEs. A difference between two scorers
    must **not** be read off overlapping ``±se`` bars — the scorers share the
    same variants (correlated estimates), so a rigorous comparison needs a
    paired-delta bootstrap (resample variants once, recompute both scorers,
    take the difference), as in ``paired_metric_delta_bootstrap``.

    The ``rng`` default ``0`` (a single ``Generator`` threaded through every
    bootstrap below) keeps outputs bit-stable across re-runs on identical
    inputs — same rationale as ``auprc_with_bootstrap_se``.

    Args:
        dataset: DataFrame with columns ``[label, effect_size]`` (row-aligned
            with ``scores``). ``effect_size`` may be NaN for controls but
            **must** be present for every positive.
        scores: DataFrame whose columns are model scores; row-aligned with
            ``dataset``. Must not contain NaN.
        score_columns: score column names to evaluate. Defaults to all
            columns of ``scores``.
        n_bootstrap: bootstrap iterations per (metric, score_column).
        rng: ``numpy.random.Generator``, seed int, or ``None``.

    Returns:
        DataFrame with columns ``[metric, score_type, value, se, n_rows,
        n_pos]``. ``metric`` is one of ``"AUPRC"``, ``"pearson"``,
        ``"spearman"``; three rows per score column. ``n_rows`` is the number
        of rows the metric used (all rows for AUPRC, positives for the
        correlations); ``n_pos`` is the positive count throughout.
    """
    for col in ("label", "effect_size"):
        assert col in dataset.columns, f"dataset missing required column {col!r}"
    assert len(dataset) == len(scores), (
        f"length mismatch: dataset={len(dataset)} scores={len(scores)}"
    )
    if score_columns is None:
        score_columns = list(scores.columns)

    label = np.asarray(dataset["label"]).astype(int)
    effect_size = np.asarray(dataset["effect_size"], dtype=float)
    pos_mask = label == 1
    n_pos = int(pos_mask.sum())
    n_rows = int(len(label))
    assert 0 < n_pos < n_rows, (
        f"AUPRC needs both classes, got n_pos={n_pos} of n={n_rows}"
    )
    assert n_pos >= 2, f"correlation needs >=2 positives, got n_pos={n_pos}"
    # Positives must carry a measured effect (dsQTL controls legitimately
    # don't; a NaN here means a positive is missing its effect — fail loud).
    n_nan_pos = int(np.isnan(effect_size[pos_mask]).sum())
    assert n_nan_pos == 0, (
        f"{n_nan_pos} positive variants have NaN effect_size — expected a "
        f"measured effect for every positive"
    )

    rng = np.random.default_rng(rng)
    es_pos = effect_size[pos_mask]
    rows: list[dict] = []
    for score_col in score_columns:
        score = np.asarray(scores[score_col], dtype=float)
        assert not np.isnan(score).any(), (
            f"score column {score_col!r} has NaN; fill upstream with a "
            f"semantically appropriate default before scoring"
        )

        auprc = auprc_with_bootstrap_se(
            label=label,
            score=score,
            match_group=np.arange(n_rows),
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
        rows.append(
            {
                "metric": "AUPRC",
                "score_type": score_col,
                "value": auprc["value"],
                "se": auprc["se"],
                "n_rows": n_rows,
                "n_pos": n_pos,
            }
        )

        score_pos = score[pos_mask]
        for method in ("pearson", "spearman"):
            value, se = _correlation_with_bootstrap_se(
                score_pos, es_pos, method=method, n_bootstrap=n_bootstrap, rng=rng
            )
            rows.append(
                {
                    "metric": method,
                    "score_type": score_col,
                    "value": value,
                    "se": se,
                    "n_rows": n_pos,
                    "n_pos": n_pos,
                }
            )

    return pd.DataFrame(rows)


def _macro(children: list[dict]) -> dict | None:
    """Equal-weight macro over leaf/child metric dicts.

    ``value`` = unweighted mean; ``se`` = SE-of-mean ``sqrt(Σ SE²)/K`` (same form
    as ``compute_auprc_metrics``); ``n`` = K (children averaged); ``n_pos`` =
    summed abnormal count, or NaN if any child's ``n_pos`` is NaN (Spearman
    children carry no positive count). Returns ``None`` if there are no children.
    """
    if not children:
        return None
    k = len(children)
    value = sum(c["value"] for c in children) / k
    se = math.sqrt(sum(c["se"] ** 2 for c in children)) / k
    n_pos_vals = [c["n_pos"] for c in children]
    n_pos = (
        float("nan") if any(np.isnan(v) for v in n_pos_vals) else int(sum(n_pos_vals))
    )
    return {"value": float(value), "se": float(se), "n": k, "n_pos": n_pos}


def _sge_cell_metrics(
    cell: pd.DataFrame,
    score_col: str,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
    n_min_auprc: int,
) -> tuple[dict[str, dict], int, int]:
    """AUPRC for one (accession, subset-scope) cell and one score column.

    Returns ``(out, n_pos, n_neg)``. ``out`` is ``{"AUPRC": dict(value, se, n,
    n_pos)}`` when both classes of the binary ``label`` (True = impactful) have
    ``>= n_min_auprc`` rows, else ``{}`` (the cell is gated — too few of one
    class for a stable AUPRC). ``n_pos`` / ``n_neg`` are the (finite-score) class
    counts, returned **regardless** of the gate so the caller can still report a
    blanked cell's sample size. Plain row bootstrap via singleton ``match_group``
    (as in ``compute_qtl_metrics``). NaN-score rows are dropped (conservation
    fills unaligned loci with 0 upstream, so this is a defensive guard).
    """
    out: dict[str, dict] = {}
    score = np.asarray(cell[score_col], dtype=float)
    label = np.asarray(cell["label"]).astype(bool)
    keep = np.isfinite(score)
    label, score = label[keep], score[keep]
    n_pos = int(label.sum())
    n_neg = int((~label).sum())
    if n_pos >= n_min_auprc and n_neg >= n_min_auprc:
        res = auprc_with_bootstrap_se(
            label=label.astype(int),
            score=score,
            match_group=np.arange(len(label)),
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
        out["AUPRC"] = {
            "value": res["value"],
            "se": res["se"],
            "n": res["n_rows"],
            "n_pos": n_pos,
        }
    return out, n_pos, n_neg


def compute_sge_metrics(
    dataset: pd.DataFrame,
    scores: pd.DataFrame,
    score_columns: list[str] | None = None,
    *,
    n_bootstrap: int = 1000,
    rng: np.random.Generator | int | None = 0,
    n_min_auprc: int = 30,
) -> pd.DataFrame:
    """Per-accession × per-subset **AUPRC** for the SGE benchmark (#301).

    Saturation-genome-editing (``evals_sge`` v3) is a binary VEP task: each
    variant has a boolean ``label`` (True = impactful = calibrated abnormal). The
    deleteriousness-oriented model score (gLM ``minus_llr`` / ``jsd``,
    conservation ``score``) predicts ``label``; **AUPRC** (rank-based, so it
    compares fairly across model families) is computed **per accession**
    (``mavedb_urn``) — scores are per-study, non-comparable — then macro-averaged.
    (Spearman vs the continuous function score was dropped in #301; the
    continuous columns stay in the dataset for provenance.)

    Per cell: AUPRC via a plain row bootstrap (singleton ``match_group``, as in
    ``compute_qtl_metrics``); requires ``>= n_min_auprc`` per label class.

    Two macro axes, each using the ``_macro_avg_`` sentinel:

    - ``subset`` ∈ {each base subset (e.g. ``missense_variant`` / ``splicing``),
      ``SGE_POOLED_SUBSET`` (``"both"``, pooled within the accession),
      ``MACRO_AVG_SUBSET`` (equal-weight mean of the base-subset values)}.
    - ``accession`` ∈ {each ``mavedb_urn``, ``MACRO_AVG_SUBSET`` (equal-weight
      mean over qualifying accessions)} — taken of **every** subset scope above,
      including the per-accession subset-macro.

    The same single ``rng`` ``Generator`` is threaded through every bootstrap (as
    in ``compute_qtl_metrics``) so outputs are bit-stable across re-runs.

    Args:
        dataset: columns ``[mavedb_urn, gene, subset, label]``, row-aligned with
            ``scores``.
        scores: model-score columns; row-aligned with ``dataset``. gLM passes
            ``minus_llr_*`` + ``jsd_*``; conservation passes ``["score"]``.
        score_columns: which score columns to evaluate (default: all of
            ``scores``).
        n_bootstrap: bootstrap iterations per cell.
        rng: ``Generator`` / seed / ``None`` (default ``0`` → reproducible).
        n_min_auprc: min count **per label class** per cell.

    Returns:
        DataFrame ``[metric, subset, accession, gene, score_type, value, se, n,
        n_pos]``. ``metric`` is always ``"AUPRC"``. For leaf cells ``n`` is the
        rows used and ``n_pos`` the impactful (label-True) count; for macro rows
        ``n`` is K (children averaged) and ``n_pos`` the summed impactful count.
        Leaf cells that fail the ``n_min_auprc`` gate are still emitted, with
        ``value``/``se`` = NaN but a real ``n`` / ``n_pos`` (so a blanked cell's
        class balance is still reported); these gated rows are excluded from
        every macro.
    """
    for col in ("mavedb_urn", "gene", "subset", "label"):
        assert col in dataset.columns, f"dataset missing required column {col!r}"
    assert len(dataset) == len(scores), (
        f"length mismatch: dataset={len(dataset)} scores={len(scores)}"
    )
    if score_columns is None:
        score_columns = list(scores.columns)

    merged = pd.concat(
        [dataset.reset_index(drop=True), scores.reset_index(drop=True)], axis=1
    )

    base_subsets = sorted(merged["subset"].dropna().unique().tolist())
    assert base_subsets, "no subset values in dataset"
    for sentinel in (SGE_POOLED_SUBSET, MACRO_AVG_SUBSET):
        assert sentinel not in base_subsets, (
            f"reserved subset name {sentinel!r} collides with a real subset"
        )

    # Each MaveDB study (mavedb_urn) identifies exactly one gene — fail loud on
    # drift, then carry `gene` for display alongside the accession key.
    urn_gene = merged.groupby("mavedb_urn")["gene"].nunique()
    assert (urn_gene == 1).all(), (
        f"mavedb_urn maps to >1 gene: {urn_gene[urn_gene > 1].to_dict()}"
    )
    urn_to_gene = merged.groupby("mavedb_urn")["gene"].first().to_dict()

    rng = np.random.default_rng(rng)

    rows: list[dict] = []
    for score_col in score_columns:
        # (accession, subset_scope, metric) -> dict(value, se, n, n_pos)
        cells: dict[tuple[str, str, str], dict] = {}
        # Leaf cells that failed the AUPRC gate: (accession, scope, n_pos, n_neg).
        # Emitted as value=NaN rows so the grid still reports the blanked cell's
        # class balance, but kept OUT of `cells` so they never enter a macro.
        gated: list[tuple[str, str, int, int]] = []

        for urn, urn_df in merged.groupby("mavedb_urn", sort=False):
            scopes: dict[str, pd.DataFrame] = {
                s: urn_df[urn_df["subset"] == s] for s in base_subsets
            }
            scopes[SGE_POOLED_SUBSET] = urn_df
            for scope, cell_df in scopes.items():
                got, n_pos, n_neg = _sge_cell_metrics(
                    cell_df,
                    score_col,
                    n_bootstrap=n_bootstrap,
                    rng=rng,
                    n_min_auprc=n_min_auprc,
                )
                for metric, res in got.items():
                    cells[(str(urn), scope, metric)] = res
                if "AUPRC" not in got and (n_pos + n_neg) > 0:
                    gated.append((str(urn), scope, n_pos, n_neg))

            # Per-accession subset-macro: mean over base subsets that qualified.
            for metric in ("AUPRC",):
                kids = [
                    cells[(str(urn), s, metric)]
                    for s in base_subsets
                    if (str(urn), s, metric) in cells
                ]
                macro = _macro(kids)
                if macro is not None:
                    cells[(str(urn), MACRO_AVG_SUBSET, metric)] = macro

        # Per-accession rows.
        for (urn, scope, metric), res in cells.items():
            rows.append(
                {
                    "metric": metric,
                    "subset": scope,
                    "accession": urn,
                    "gene": urn_to_gene[urn],
                    "score_type": score_col,
                    "value": res["value"],
                    "se": res["se"],
                    "n": res["n"],
                    "n_pos": res["n_pos"],
                }
            )

        # Gated leaf cells: value=NaN, but carry the class counts so the grid
        # reports the sample size of each blanked cell (n_pos / n=n_pos+n_neg).
        for urn, scope, n_pos, n_neg in gated:
            rows.append(
                {
                    "metric": "AUPRC",
                    "subset": scope,
                    "accession": urn,
                    "gene": urn_to_gene[urn],
                    "score_type": score_col,
                    "value": float("nan"),
                    "se": float("nan"),
                    "n": n_pos + n_neg,
                    "n_pos": n_pos,
                }
            )

        # Accession-macro: mean over accessions of every subset scope.
        all_scopes = base_subsets + [SGE_POOLED_SUBSET, MACRO_AVG_SUBSET]
        accessions = [str(u) for u in merged["mavedb_urn"].unique()]
        for scope in all_scopes:
            for metric in ("AUPRC",):
                kids = [
                    cells[(urn, scope, metric)]
                    for urn in accessions
                    if (urn, scope, metric) in cells
                ]
                macro = _macro(kids)
                if macro is not None:
                    rows.append(
                        {
                            "metric": metric,
                            "subset": scope,
                            "accession": MACRO_AVG_SUBSET,
                            "gene": MACRO_AVG_SUBSET,
                            "score_type": score_col,
                            "value": macro["value"],
                            "se": macro["se"],
                            "n": macro["n"],
                            "n_pos": macro["n_pos"],
                        }
                    )

    return pd.DataFrame(rows)


def compute_pairwise_metrics(
    dataset: pd.DataFrame,
    scores: pd.DataFrame,
    score_columns: list[str] | None = None,
    n_min: int = 30,
) -> pd.DataFrame:
    """Compute PairwiseAccuracy + SE per ``subset`` for one or more score columns.

    Aligns ``dataset`` with ``scores`` by row index (assumes same order).
    Stratifies by the ``subset`` column — one row per (subset, score_column).
    Asserts no ``match_group`` straddles subsets.

    Additionally emits two aggregate rows per ``score_col`` (sentinel subset
    values; underscore-bracketed to avoid collision with real consequence-group
    names which never start with ``_``):

    - ``subset="_global_"``: PA over **all** ``match_group``s in the input,
      regardless of subset size. Computed by a direct ``pairwise_accuracy``
      call on the full frame — not by recombining per-subset values — so the
      number is provably "score every pair, average". Same Wald SE.
    - ``subset="_macro_avg_"``: unweighted mean of per-subset values across
      subsets with ``n_pairs >= n_min``. SE = ``sqrt(Σ SE_s²) / K`` (SE of
      the unweighted mean of K independent estimates). ``n_pairs`` on this
      row is **repurposed** to record K, the count of qualifying subsets;
      ``n_ties`` is the sum across qualifying subsets.

    Args:
        dataset: DataFrame with columns ``[label, subset, match_group]`` (and
            optionally other passthrough columns).
        scores: DataFrame whose columns are the model scores; row-aligned with
            ``dataset``.
        score_columns: Score column names to evaluate. Defaults to all columns
            of ``scores``.
        n_min: Minimum ``n_pairs`` per subset to include in the macro average.
            Defaults to 30 (project-wide convention for the leaderboard
            issues). Subsets below this threshold still contribute to the
            global row.

    Returns:
        DataFrame with columns ``[score_type, subset, value, se, n_pairs,
        n_ties]``. ``score_type`` is the column name from ``scores``.
    """
    for col in ("label", "subset", "match_group"):
        assert col in dataset.columns, f"dataset missing required column {col!r}"

    if score_columns is None:
        score_columns = list(scores.columns)

    merged = pd.concat(
        [dataset.reset_index(drop=True), scores.reset_index(drop=True)], axis=1
    )

    # No match_group may straddle subsets — would silently double-count or
    # drop pairs depending on filter order.
    subset_per_group = merged.groupby("match_group")["subset"].nunique()
    bad_groups = subset_per_group[subset_per_group > 1]
    assert bad_groups.empty, (
        f"{len(bad_groups)} match_group(s) span multiple subsets; first: "
        f"{bad_groups.head().to_dict()}"
    )

    rows: list[dict] = []
    per_score_rows: dict[str, list[dict]] = {sc: [] for sc in score_columns}
    for subset_name, subset_df in merged.groupby("subset", sort=False):
        for score_col in score_columns:
            res = pairwise_accuracy(
                label=subset_df["label"],
                score=subset_df[score_col],
                match_group=subset_df["match_group"],
            )
            row = {
                "score_type": score_col,
                "subset": str(subset_name),
                "value": res["value"],
                "se": res["se"],
                "n_pairs": res["n_pairs"],
                "n_ties": res["n_ties"],
            }
            rows.append(row)
            per_score_rows[score_col].append(row)

    for score_col in score_columns:
        # Global: PA over every match_group, regardless of subset.
        global_res = pairwise_accuracy(
            label=merged["label"],
            score=merged[score_col],
            match_group=merged["match_group"],
        )
        rows.append(
            {
                "score_type": score_col,
                "subset": GLOBAL_SUBSET,
                "value": global_res["value"],
                "se": global_res["se"],
                "n_pairs": global_res["n_pairs"],
                "n_ties": global_res["n_ties"],
            }
        )

        # Macro avg: unweighted mean of per-subset PAs with n_pairs >= n_min.
        qualifying = [r for r in per_score_rows[score_col] if r["n_pairs"] >= n_min]
        assert qualifying, (
            f"no subsets meet n_min={n_min} for score_type={score_col!r}; "
            f"per-subset sizes: "
            f"{ {r['subset']: r['n_pairs'] for r in per_score_rows[score_col]} }"
        )
        k = len(qualifying)
        macro_value = sum(r["value"] for r in qualifying) / k
        macro_se = math.sqrt(sum(r["se"] ** 2 for r in qualifying)) / k
        rows.append(
            {
                "score_type": score_col,
                "subset": MACRO_AVG_SUBSET,
                "value": float(macro_value),
                "se": float(macro_se),
                "n_pairs": k,
                "n_ties": sum(r["n_ties"] for r in qualifying),
            }
        )

    return pd.DataFrame(rows)


def compute_metrics(
    dataset: pd.DataFrame,
    scores: pd.DataFrame,
    metrics: list[str],
    score_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Compute classical (AUPRC/AUROC/Spearman) metrics for variant predictions.

    Aligns ``dataset`` with ``scores`` by row index. For datasets with a
    ``subset`` column, computes metrics on the full dataset (``subset='global'``)
    and on each individual subset.

    Args:
        dataset: DataFrame with ``[chrom, pos, ref, alt, label]`` and optionally
            ``subset``.
        scores: DataFrame whose columns are the model scores; row-aligned with
            ``dataset``.
        metrics: List of metric names from ``METRIC_FUNCTIONS``.
        score_columns: Score column names to evaluate. Defaults to
            ``['minus_llr', 'abs_llr']``.

    Returns:
        DataFrame with columns ``[metric, score_type, subset, value, n_pos,
        n_neg]``.
    """
    if score_columns is None:
        score_columns = ["minus_llr", "abs_llr"]

    merged = pd.concat(
        [dataset.reset_index(drop=True), scores.reset_index(drop=True)], axis=1
    )

    subsets_to_evaluate = [("global", merged)]
    if "subset" in merged.columns:
        for subset_name in merged["subset"].unique():
            subset_data = merged[merged["subset"] == subset_name]
            subsets_to_evaluate.append((subset_name, subset_data))

    results = []
    for subset_name, subset_data in subsets_to_evaluate:
        n_pos = int((subset_data["label"] == 1).sum())
        n_neg = int((subset_data["label"] == 0).sum())
        for metric_name in metrics:
            metric_func = METRIC_FUNCTIONS[metric_name]
            for score_col in score_columns:
                value = metric_func(subset_data["label"], subset_data[score_col])
                results.append(
                    {
                        "metric": metric_name,
                        "score_type": score_col,
                        "subset": subset_name,
                        "value": value,
                        "n_pos": n_pos,
                        "n_neg": n_neg,
                    }
                )
    return pd.DataFrame(results)


def aggregate_metrics(
    metric_files: list[str], dataset_names: list[str], model_steps: list[str]
) -> pd.DataFrame:
    """Aggregate metrics from multiple evaluation runs into a single DataFrame.

    Args:
        metric_files: List of paths to metric parquet files.
        dataset_names: List of dataset names corresponding to each file.
        model_steps: List of model training steps corresponding to each file.

    Returns:
        DataFrame with all per-file rows plus ``[step, dataset]`` columns,
        sorted by step and dataset.
    """
    all_metrics = []
    for file_path, dataset_name, step in zip(metric_files, dataset_names, model_steps):
        df = pd.read_parquet(file_path)
        df["dataset"] = dataset_name
        df["step"] = int(step)
        all_metrics.append(df)
    result = pd.concat(all_metrics, ignore_index=True)
    return result.sort_values(["step", "dataset"]).reset_index(drop=True)
