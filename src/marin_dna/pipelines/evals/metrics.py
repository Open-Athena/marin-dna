"""Metric utilities for variant-effect evaluations.

Three metric families live here:

- ``auprc_with_bootstrap_se`` / ``compute_auprc_metrics``: AUPRC with a
  cluster bootstrap SE that resamples ``match_group``s (preserving the
  matched-pair clustering). Used by the ``evals_v2`` pipeline after PR
  #194 moved the matched-pair datasets to a 1:k structure that the
  Wald-binomial pairwise metric can't represent.
- ``pairwise_accuracy`` / ``compute_pairwise_metrics``: matched-pair within-
  ``match_group`` accuracy (ties = 0.5) with Wald-binomial SE. Used by the
  ``conservation_eval`` pipeline (1:1 match groups).
- ``METRIC_FUNCTIONS`` / ``compute_metrics``: classical AUPRC / AUROC /
  Spearman over (label, score) pairs. Still used by older pipelines
  (``snakemake/analysis/evals_v1/``, ``scripts/evo2_eval/``).
"""

import math
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
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
        n_bootstrap: number of bootstrap iterations.
        rng: ``numpy.random.Generator``, seed int, or ``None``.

    Returns:
        ``{"value", "se", "n_groups", "n_rows"}``. ``value`` is the
        point-estimate AUPRC over all input rows; ``se`` is the std of
        the bootstrap distribution (``ddof=1``, NaN-tolerant for
        degenerate resamples).
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

    rng = np.random.default_rng(rng)
    # `groupby(mg_arr).indices` is an O(n) hash-based group → positional-
    # index map. The earlier `[np.where(inv == i)[0] for i in groups]`
    # form was O(n_groups · n_rows) and dominated the non-AP cost when
    # n_groups was ~10³ within a subset.
    group_to_rows: list[np.ndarray] = list(
        pd.Series(mg_arr).groupby(mg_arr).indices.values()
    )
    n_groups = len(group_to_rows)

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
