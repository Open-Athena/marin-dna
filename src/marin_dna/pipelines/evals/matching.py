"""Feature-based sample matching utilities.

Ported from TraitGym (commit e59d612e9; src/traitgym/matching.py).
"""

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
import polars as pl
from scipy.spatial.distance import cdist
from sklearn.preprocessing import RobustScaler

from marin_dna.pipelines.evals.variants import COORDINATES

MATCH_GROUP_COL = "match_group"

# Sentinels for bin labels.
BIN_OOR = "OOR"  # value outside any bin (or null) — emitted by `bin_feature`
BIN_NA = "NA"  # subset-conditional bin not applicable for this row

# Bin schemes from the iter-33 design (issue #156). No longer used by the
# pipeline rules (matching is now continuous-only + chrom/consequence_final
# categorical), but the helpers + constants are kept until we're confident
# the no-binning recipe holds across rebuilds.
TSS_DIST_BIN_EDGES = [0, 50, 100, 200, 500, 1000]
EXON_DIST_BIN_EDGES = [0, 5, 20, 30]
# Three MAF bin granularities used by the per-subset tiered scheme.
MAF_BIN_EDGES_20 = [
    0.0,
    0.0005,
    0.001,
    0.0015,
    0.002,
    0.0025,
    0.003,
    0.0035,
    0.004,
    0.005,
    0.007,
    0.01,
    0.015,
    0.02,
    0.03,
    0.05,
    0.07,
    0.1,
    0.15,
    0.2,
    0.5,
]
MAF_BIN_EDGES_10 = [0.0, 0.0005, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.5]
MAF_BIN_EDGES_5 = [0.0, 0.001, 0.01, 0.05, 0.2, 0.5]
# Back-compat alias: existing tests / scratch scripts still import MAF_BIN_EDGES.
MAF_BIN_EDGES = MAF_BIN_EDGES_20

# Per-subset MAF bin scheme (iter 33 production).
MAF_TIERED_V1: dict[str, list[float]] = {
    # Big subsets with strong leak in the no-bin baseline → fine 20bin.
    "distal": MAF_BIN_EDGES_20,
    "tss_proximal": MAF_BIN_EDGES_20,
    "non_coding_transcript_exon_variant": MAF_BIN_EDGES_20,
    # Medium subsets → 10bin.
    "3_prime_UTR_variant": MAF_BIN_EDGES_10,
    "5_prime_UTR_variant": MAF_BIN_EDGES_10,
    "missense_variant": MAF_BIN_EDGES_10,
    # Small subsets → 5bin.
    "synonymous_variant": MAF_BIN_EDGES_5,
    "splicing": MAF_BIN_EDGES_5,
    "mature_miRNA_variant": MAF_BIN_EDGES_5,
    "stop_retained_variant": MAF_BIN_EDGES_5,
    "coding_sequence_variant": MAF_BIN_EDGES_5,
}


def bin_feature(
    feature: str,
    edges: Sequence[float],
    *,
    right_closed: bool = False,
) -> pl.Expr:
    """Polars expression binning ``feature`` into ``len(edges) - 1`` string buckets.

    Returns ``"b0".."b{n-1}"`` for in-range values, ``"OOR"`` for out-of-range
    or null values.

    With ``right_closed=False`` (default; used for ``tss_dist`` / ``exon_dist``):
        bins are ``[lo, hi)`` for ``i < n-1``, ``[lo, hi]`` for the last bin.
    With ``right_closed=True`` (used for ``MAF``):
        bins are ``(lo, hi]`` for ``i > 0``, ``[lo, hi]`` for the first bin.
    """
    assert len(edges) >= 2, f"need at least 2 edges, got {edges!r}"
    n = len(edges) - 1
    expr = pl.lit(BIN_OOR)
    col = pl.col(feature)
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        if right_closed:
            cond = ((col >= lo) & (col <= hi)) if i == 0 else ((col > lo) & (col <= hi))
        else:
            cond = (
                ((col >= lo) & (col <= hi))
                if i == n - 1
                else ((col >= lo) & (col < hi))
            )
        expr = pl.when(cond).then(pl.lit(f"b{i}")).otherwise(expr)
    return expr


# Categorical match-key shared by all eval datasets. Gene-ID columns were
# dropped from this key (still carried as passthrough metadata on the
# output) because exact gene matching at scale drops too many positives.
CAT_BASE: list[str] = [
    "chrom",
    "consequence_final",
]


def add_subset_distance_bins(df: pl.DataFrame) -> pl.DataFrame:
    """Iter-33 per-biotype distance bins as exact-match categoricals.

    - ``distance_tss_pc_bin`` and ``distance_tss_nc_bin``: meaningful only for
      ``tss_proximal`` (else ``BIN_NA``); edges = ``TSS_DIST_BIN_EDGES``.
    - ``distance_exon_pc_bin``: meaningful only for ``splicing`` (else
      ``BIN_NA``); edges = ``EXON_DIST_BIN_EDGES``.

    No longer used by the pipeline rules; kept for ad-hoc analysis.
    """
    return df.with_columns(
        pl.when(pl.col("consequence_group") == "tss_proximal")
        .then(bin_feature("distance_tss_pc", TSS_DIST_BIN_EDGES))
        .otherwise(pl.lit(BIN_NA))
        .alias("distance_tss_pc_bin"),
        pl.when(pl.col("consequence_group") == "tss_proximal")
        .then(bin_feature("distance_tss_nc", TSS_DIST_BIN_EDGES))
        .otherwise(pl.lit(BIN_NA))
        .alias("distance_tss_nc_bin"),
        pl.when(pl.col("consequence_group") == "splicing")
        .then(bin_feature("distance_exon_pc", EXON_DIST_BIN_EDGES))
        .otherwise(pl.lit(BIN_NA))
        .alias("distance_exon_pc_bin"),
    )


def add_subset_distance_bins_v2(
    df: pl.DataFrame,
    scheme: dict[tuple[str, str], list[float]],
) -> pl.DataFrame:
    """Per-(subset, feature) distance bins for use as exact-match categoricals.

    For each ``(subset, feature) -> edges`` entry in ``scheme``, populate a
    ``{feature}_bin`` column where the value is ``"{subset[:8]}:b{i}"`` when
    ``consequence_group == subset``, else ``BIN_NA`` (or the value carried
    over from another subset that targets the same feature). The subset
    prefix keeps labels disjoint across subsets so a single ``{feature}_bin``
    column can carry different bin schemes for different subsets without
    collisions; ``consequence_final`` is in the categorical match key so
    cross-subset matching is impossible anyway.

    Args:
        df: dataframe with ``consequence_group`` and the features in ``scheme``.
        scheme: ``{(subset, feature): edges}``. Multiple subsets can share a
            feature.

    Returns:
        ``df`` with one new column per unique feature in ``scheme``.
    """
    # Group by feature so we build one column per feature with subset-conditional
    # bin assignment.
    by_feature: dict[str, list[tuple[str, list[float]]]] = {}
    for (subset, feat), edges in scheme.items():
        by_feature.setdefault(feat, []).append((subset, edges))

    new_cols = []
    for feat, entries in by_feature.items():
        expr: pl.Expr = pl.lit(BIN_NA)
        for subset, edges in entries:
            bin_expr = pl.format(
                "{}:{}",
                pl.lit(subset[:8]),
                bin_feature(feat, edges),
            )
            expr = (
                pl.when(pl.col("consequence_group") == subset)
                .then(bin_expr)
                .otherwise(expr)
            )
        new_cols.append(expr.alias(f"{feat}_bin"))
    return df.with_columns(*new_cols)


def add_tiered_maf_bin(
    df: pl.DataFrame,
    scheme: dict[str, list[float]],
) -> pl.DataFrame:
    """Add a ``MAF_bin`` column whose edges depend on each row's
    ``consequence_group``.

    Bin labels are prefixed with the subset name (``"missense:b3"``,
    ``"distal:OOR"``) so labels never collide across subsets. Rows whose
    consequence_group is not in ``scheme`` get ``"UNKNOWN"``.

    No longer used by the pipeline rules; kept for ad-hoc analysis.
    """
    # Defensive: subset names get truncated to 8 chars in the bin label below
    # to keep the column compact. Fail loudly on first-8-char collisions.
    prefixes = {s[:8] for s in scheme}
    assert len(prefixes) == len(scheme), (
        f"first-8-char collision among scheme keys: {sorted(scheme)}"
    )

    expr = pl.lit("UNKNOWN")
    for subset, edges in scheme.items():
        bin_expr = pl.format(
            "{}:{}",
            pl.lit(subset[:8]),
            bin_feature("MAF", edges, right_closed=True),
        )
        expr = (
            pl.when(pl.col("consequence_group") == subset)
            .then(bin_expr)
            .otherwise(expr)
        )

    return df.with_columns(expr.alias("MAF_bin"))


def match_features(
    pos: pl.DataFrame,
    neg: pl.DataFrame,
    continuous_features: list[str],
    categorical_features: list[str],
    k: int,
    scale: bool = True,
    seed: int | None = 42,
) -> pl.DataFrame:
    """Match positive samples to k negative samples based on features.

    For each unique combination of categorical features, finds the k closest
    negative samples for each positive sample based on continuous features.

    Args:
        pos: Positive samples DataFrame with COORDINATES columns.
        neg: Negative samples DataFrame with COORDINATES columns.
        continuous_features: Columns to use for distance-based matching.
        categorical_features: Columns for exact matching (grouping).
        k: Number of negative samples to match per positive sample.
        scale: Whether to scale continuous features before matching.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with matched samples, match_group column, sorted by COORDINATES.

    Raises:
        ValueError: If required columns are missing from pos or neg, or if
            null values are present in match features.
    """
    required_cols = COORDINATES + continuous_features + categorical_features
    _validate_columns(pos, required_cols, "pos")
    _validate_columns(neg, required_cols, "neg")
    all_features = continuous_features + categorical_features
    _validate_no_nulls(pos, all_features, "pos")
    _validate_no_nulls(neg, all_features, "neg")

    # Pre-filter: drop neg rows whose categorical combo doesn't match any
    # positive's. Behaviorally identical to the current loop (those negs would
    # never enter `neg_groups[group_key]` for any positive's group_key), but
    # cuts the next two steps' cost by ~50-100× on real eval datasets where
    # most negatives sit far from any positive in (gene_id × MAF_bin × …)
    # space. complex_traits goes from ~12M neg rows to ~100k after this join,
    # which makes the to_pandas/partition_by stack run in seconds instead of
    # minutes (and on a 16 GB box instead of needing 256 GB).
    # `partition_by` (used below) and the semi-join both require at least one
    # categorical key. Pre-iter-33 the function had the same limitation
    # (partition_by raises on an empty key list); this assertion just makes
    # the failure mode obvious instead of buried polars-internal.
    assert categorical_features, "categorical_features must be non-empty"
    pos_keys = pos.select(categorical_features).unique()
    neg = neg.join(pos_keys, on=categorical_features, how="semi")

    pos_pd = pos.to_pandas()
    neg_pd = neg.to_pandas()

    if scale and len(continuous_features) > 0:
        pos_pd, neg_pd, match_cols = _scale_features(
            pos_pd, neg_pd, continuous_features
        )
    else:
        match_cols = continuous_features

    # Use polars partition_by for fast group splitting. Pandas multi-index
    # .loc on millions of negatives is the per-call hot spot otherwise.
    pos_groups = pl.from_pandas(pos_pd).partition_by(categorical_features, as_dict=True)
    neg_groups = pl.from_pandas(neg_pd).partition_by(categorical_features, as_dict=True)

    pos_list: list[pd.DataFrame] = []
    neg_list: list[pd.DataFrame] = []

    for group_key, pos_group_pl in pos_groups.items():
        # partition_by returns single-element tuples even for one-column keys;
        # strip when match_single_group expects the unwrapped scalar in messages.
        display_key = group_key[0] if len(group_key) == 1 else group_key
        neg_group_pl = neg_groups.get(group_key)
        if neg_group_pl is None:
            warnings.warn(f"No negatives found for category: {display_key}")
            continue
        result = _match_single_group(
            pos_group_pl.to_pandas(),
            neg_group_pl.to_pandas(),
            display_key,
            match_cols,
            k,
            seed,
        )
        if result is not None:
            pos_group, neg_group = result
            pos_list.append(pos_group)
            neg_list.append(neg_group)

    result_df = _combine_results(pos_list, neg_list, k)

    if scale and len(continuous_features) > 0:
        scaled_cols = [f"{c}_scaled" for c in continuous_features]
        result_df = result_df.drop(columns=scaled_cols, errors="ignore")

    return _sort_by_coordinates(pl.from_pandas(result_df))


def _validate_columns(
    df: pl.DataFrame,
    required: list[str],
    name: str,
) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def _validate_no_nulls(
    df: pl.DataFrame,
    columns: list[str],
    name: str,
) -> None:
    for col in columns:
        null_count = df[col].null_count()
        if null_count > 0:
            raise ValueError(f"{name} has {null_count} null values in column '{col}'")


def _scale_features(
    pos: pd.DataFrame,
    neg: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    scaler = RobustScaler()
    all_data = pd.concat([pos[features], neg[features]])
    scaler.fit(all_data)

    scaled_cols = [f"{c}_scaled" for c in features]
    pos = pos.copy()
    neg = neg.copy()
    pos[scaled_cols] = scaler.transform(pos[features])
    neg[scaled_cols] = scaler.transform(neg[features])

    return pos, neg, scaled_cols


def _match_single_group(
    pos_group: pd.DataFrame,
    neg_group: pd.DataFrame,
    group_key: tuple | str,
    match_cols: list[str],
    k: int,
    seed: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    n_pos_needed = len(pos_group)
    n_neg_available = len(neg_group)
    n_neg_needed = n_pos_needed * k

    if n_neg_needed > n_neg_available:
        warnings.warn(
            f"Insufficient negatives for category {group_key}: "
            f"need {n_neg_needed}, have {n_neg_available}. Subsampling positives."
        )
        n_pos_possible = n_neg_available // k
        pos_group = pos_group.sample(n_pos_possible, random_state=seed)

    if len(match_cols) == 0:
        n_neg_to_sample = len(pos_group) * k
        neg_matched = neg_group.sample(n_neg_to_sample, random_state=seed)
    else:
        neg_matched = _find_closest(pos_group, neg_group, match_cols, k)

    return pos_group, neg_matched


def _find_closest(
    pos: pd.DataFrame,
    neg: pd.DataFrame,
    cols: list[str],
    k: int,
) -> pd.DataFrame:
    """Find k closest negatives for each positive (Euclidean), without replacement."""
    assert k >= 1, f"_find_closest needs k >= 1, got {k}"
    if len(pos) == 0:
        # `_match_single_group` subsamples positives down to `n_neg // k`
        # when the stratum has < k negatives — possibly to zero. Return an
        # empty selection rather than running cdist on an empty pos and
        # tripping the n_neg < k assertion below.
        return neg.iloc[[]]
    n_neg = len(neg)
    assert k <= n_neg, (
        f"_find_closest needs k ({k}) <= len(neg) ({n_neg}); "
        "_match_single_group is supposed to subsample positives first"
    )
    dist_matrix = cdist(pos[cols], neg[cols])
    closest_indices: list[int] = []

    # argpartition is O(n) vs argsort's O(n log n) — matters at k=9 with
    # the larger strata that arise once we drop gene-id from the categorical
    # match key. After partitioning, sort just those k indices by distance
    # for a deterministic, distance-ordered output.
    for i in range(len(pos)):
        partitioned = np.argpartition(dist_matrix[i], k - 1)[:k]
        sorted_indices = partitioned[np.argsort(dist_matrix[i][partitioned])].tolist()
        closest_indices.extend(sorted_indices)
        dist_matrix[:, sorted_indices] = np.inf

    return neg.iloc[closest_indices]


def _combine_results(
    pos_list: list[pd.DataFrame],
    neg_list: list[pd.DataFrame],
    k: int,
) -> pd.DataFrame:
    if not pos_list:
        return pd.DataFrame()

    all_pos = pd.concat(pos_list, ignore_index=True)
    all_pos[MATCH_GROUP_COL] = np.arange(len(all_pos))

    all_neg = pd.concat(neg_list, ignore_index=True)
    all_neg[MATCH_GROUP_COL] = np.repeat(all_pos[MATCH_GROUP_COL].values, k)

    return pd.concat([all_pos, all_neg], ignore_index=True)


def _sort_by_coordinates(df: pl.DataFrame) -> pl.DataFrame:
    return df.sort(COORDINATES)
