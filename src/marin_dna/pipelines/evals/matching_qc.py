"""Per-subset matching diagnostics: subsampling counts + per-feature AUPRC leak.

Two diagnostics, both broken down by ``consequence_group`` subset:

1. **Subsampling drops** — positives that didn't make it into the matched
   dataset because their (chrom, consequence_final) stratum had fewer than
   ``k`` negatives, so ``_match_single_group`` subsampled positives.
2. **Per-feature AUPRC leak** — for each continuous matching feature, the
   AUPRC of `label ~ feature` (with sign flip — the better of `+feature`
   and `−feature`) within each subset of the matched dataset. The baseline
   for 1:k matching is ``1 / (1 + k)``. A value at-or-near baseline means
   matching controlled for that feature in that subset; a value well above
   baseline indicates residual leak.

Output is a parquet artifact intended for human inspection, not a gating
check.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score

# Subset column name in pre-matching `dataset_all.parquet`.
PRE_SUBSET_COL = "consequence_group"
# Subset column name in post-matching `dataset_unsplit/{task}.parquet`
# (renamed from `consequence_group` by each `*_dataset` rule).
POST_SUBSET_COL = "subset"


def compute_matching_qc(
    pre_df: pl.DataFrame,
    post_df: pl.DataFrame,
    continuous_features: list[str],
) -> pl.DataFrame:
    """Per-subset matching diagnostics.

    Args:
        pre_df: pre-matching dataframe (``dataset_all.parquet``); must contain
            ``label`` and :data:`PRE_SUBSET_COL`.
        post_df: post-matching dataframe (``dataset_unsplit/{task}.parquet``);
            must contain ``label``, :data:`POST_SUBSET_COL`, and every column
            in ``continuous_features``.
        continuous_features: features over which to compute per-subset AUPRC.

    Returns:
        One row per subset value (union of subsets seen in pre and post).
        Columns:

        - ``subset``
        - ``n_positives_input`` — positives in ``pre_df`` for this subset.
        - ``n_positives_kept`` — positives that survived matching.
        - ``n_dropped`` = input − kept.
        - ``frac_dropped`` = ``n_dropped / n_positives_input`` (null when input is 0).
        - ``baseline_auprc`` — positive prevalence in this subset of
          ``post_df`` (≈ ``1 / (1 + k)`` if every positive kept all matches).
        - For each ``f`` in ``continuous_features``:
            - ``{f}_auprc`` — ``max(AP(label, +f), AP(label, −f))``.
            - ``{f}_auprc_sign`` — ``+1`` if ``+f`` gave the max, ``-1`` if ``−f``.
          Both columns are null for subsets with <1 positive or <1 negative
          in ``post_df`` (AUPRC undefined).
    """
    assert "label" in pre_df.columns, "pre_df must have a `label` column"
    assert PRE_SUBSET_COL in pre_df.columns, (
        f"pre_df must have a `{PRE_SUBSET_COL}` column"
    )
    assert "label" in post_df.columns, "post_df must have a `label` column"
    assert POST_SUBSET_COL in post_df.columns, (
        f"post_df must have a `{POST_SUBSET_COL}` column"
    )
    for f in continuous_features:
        assert f in post_df.columns, f"post_df missing continuous feature `{f}`"

    summary = _per_subset_counts(pre_df, post_df)
    feature_df = _per_subset_feature_auprc(
        post_df, continuous_features, subsets=summary["subset"].to_list()
    )
    return summary.join(feature_df, on="subset", how="left")


def _per_subset_counts(pre_df: pl.DataFrame, post_df: pl.DataFrame) -> pl.DataFrame:
    n_input = (
        pre_df.filter(pl.col("label"))
        .group_by(PRE_SUBSET_COL)
        .agg(pl.len().alias("n_positives_input"))
        .rename({PRE_SUBSET_COL: "subset"})
    )
    counts = (
        post_df.group_by(POST_SUBSET_COL)
        .agg(
            pl.col("label").sum().cast(pl.Int64).alias("n_positives_kept"),
            pl.len().alias("n_total"),
        )
        .rename({POST_SUBSET_COL: "subset"})
    )
    return (
        n_input.join(counts, on="subset", how="full", coalesce=True)
        .with_columns(
            pl.col("n_positives_input").fill_null(0),
            pl.col("n_positives_kept").fill_null(0),
            pl.col("n_total").fill_null(0),
        )
        .with_columns(
            n_dropped=pl.col("n_positives_input") - pl.col("n_positives_kept"),
            frac_dropped=pl.when(pl.col("n_positives_input") > 0)
            .then(
                (pl.col("n_positives_input") - pl.col("n_positives_kept"))
                / pl.col("n_positives_input")
            )
            .otherwise(None),
            baseline_auprc=pl.when(pl.col("n_total") > 0)
            .then(pl.col("n_positives_kept") / pl.col("n_total"))
            .otherwise(None),
        )
        .drop("n_total")
        .sort("subset")
    )


def _per_subset_feature_auprc(
    post_df: pl.DataFrame,
    continuous_features: list[str],
    subsets: list[str],
) -> pl.DataFrame:
    # Cache the per-subset filtered frames so each feature reuses the same
    # filter result instead of refiltering post_df F times per subset.
    subset_frames = {s: post_df.filter(pl.col(POST_SUBSET_COL) == s) for s in subsets}
    cols: dict[str, list] = {"subset": list(subsets)}
    for feat in continuous_features:
        auprc: list[float | None] = []
        sign: list[int | None] = []
        for subset in subsets:
            sub = subset_frames[subset]
            labels = sub["label"].cast(pl.Int64).to_numpy()
            scores = sub[feat].to_numpy()
            mask = ~np.isnan(scores)
            labels, scores = labels[mask], scores[mask]
            n_pos = int(labels.sum())
            n_neg = int(len(labels) - n_pos)
            if n_pos == 0 or n_neg == 0:
                auprc.append(None)
                sign.append(None)
                continue
            up = float(average_precision_score(labels, scores))
            down = float(average_precision_score(labels, -scores))
            if up >= down:
                auprc.append(up)
                sign.append(1)
            else:
                auprc.append(down)
                sign.append(-1)
        cols[f"{feat}_auprc"] = auprc
        cols[f"{feat}_auprc_sign"] = sign
    return pl.DataFrame(cols)
