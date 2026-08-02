"""Repeat-aware sensitivity analysis for Mendelian SAE-label associations.

This is Stage 9 of issue 435.  It joins the exact official Mendelian label panel
to the outcome-blind RepeatMasker annotation by row and variant identity, then
reruns the issue-436 all-feature association family independently within focal-
repeat, near-repeat, and repeat-free windows.  The all-variant stratum must
numerically reproduce the archived issue-436 result before any output is kept.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import time
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import scipy
from scipy import stats

from association_common import bh_adjust
from common import ISSUE, assert_commit, sha256_file, write_json
from extract_common import D_SAE
from variant_analysis_common import (
    ARMS,
    BLOCK_BY_ARM,
    ORIENTATIONS,
    PAIRED_ACTIVATION_MANIFEST_SHA256,
    RESPONSES,
    VARIANT_PANEL_ARCHIVE_SHA256,
)
from variant_common import (
    EXPECTED_MATCH_GROUPS,
    EXPECTED_VARIANTS,
    POSITION_STATUSES,
    SOURCE_DATASET_ID,
    SOURCE_DATASET_REVISION,
    SOURCE_PANEL_SHA256,
)

RUN_ID = "dna-exp435-repeat-label-sensitivity-r1"
ACTIVATION_RUN_ID = "dna-exp436-mendelian-focal-seed288-r1"
GLOBAL_ASSOCIATION_RUN_ID = "dna-exp436-mendelian-focal-associations-seed288-r2"
GLOBAL_ASSOCIATION_MANIFEST_SHA256 = (
    "0c7ed2242422be660a9f18e56ff07baa3a22fc1539932e2a788773a21c8f9c6f"
)
REFERENCE_ASSOCIATION_RUN_ID = "dna-exp435-repeat-reference-associations-r1"
REFERENCE_ASSOCIATION_ARCHIVE_SHA256 = (
    "cc72fbb0033290af54d2c6dcb0a7521e9b23f5f84b6906bfd9fee1f69206ece0"
)
PAIRED_REPEAT_RUN_ID = "dna-exp435-repeat-variant-deltas-r1"
PAIRED_REPEAT_ARCHIVE_SHA256 = (
    "2460efef6dbd28e4c71c95d081276d15140db41d1919e50bcf99025f786cf9d6"
)

STRATA = ("all", "repeat_free_window", "focal_repeat", "near_repeat")
MIN_NONZERO_SUPPORT = 10
MIN_CLASS_SIZE = 40
FDR_THRESHOLD = 0.05
FEATURE_OF_INTEREST = 9086
PANEL_KEYS = (
    "chrom",
    "pos",
    "ref",
    "alt",
    "subset",
    "match_group",
    "split",
)


@dataclass(frozen=True)
class Target:
    stratum: str
    kind: str
    name: str
    indices: np.ndarray
    labels: np.ndarray


def verify_outer_archive(
    root: Path, expected_sha256: str, expected_status: str
) -> dict[str, Any]:
    manifest_path = root / "archive_manifest.json"
    assert manifest_path.is_file() and sha256_file(manifest_path) == expected_sha256
    manifest = json.loads(manifest_path.read_text())
    assert manifest["analysis_status"] == expected_status
    for relative, metadata in manifest["artifacts"].items():
        path = root / relative
        assert path.is_file(), path
        assert path.stat().st_size == metadata["bytes"], path
        assert sha256_file(path) == metadata["sha256"], path
    return manifest


def verify_inner_archive(
    root: Path,
    expected_sha256: str,
    expected_run_id: str,
) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    assert manifest_path.is_file() and sha256_file(manifest_path) == expected_sha256
    manifest = json.loads(manifest_path.read_text())
    assert manifest["run_id"] == expected_run_id
    for relative, metadata in manifest["artifacts"].items():
        path = root / relative
        assert path.is_file(), path
        assert path.stat().st_size == metadata["bytes"], path
        assert sha256_file(path) == metadata["sha256"], path
    return manifest


def verify_activation_archive(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    assert manifest_path.is_file()
    assert sha256_file(manifest_path) == PAIRED_ACTIVATION_MANIFEST_SHA256
    manifest = json.loads(manifest_path.read_text())
    assert manifest["run_id"] == ACTIVATION_RUN_ID
    assert manifest["panel"]["sha256"] == SOURCE_PANEL_SHA256
    for arm in ARMS:
        for orientation in ORIENTATIONS:
            relative = f"{arm}/sae_focal_{orientation}.parquet"
            metadata = manifest["artifacts"][relative]
            path = root / relative
            assert path.is_file() and path.stat().st_size == metadata["bytes"]
            assert sha256_file(path) == metadata["sha256"]
    return manifest


def load_aligned_panel(label_panel_path: Path, repeat_root: Path) -> pl.DataFrame:
    """Join labels to repeat annotations only after exact identity validation."""

    assert label_panel_path.is_file()
    assert sha256_file(label_panel_path) == SOURCE_PANEL_SHA256
    source = pl.read_parquet(
        label_panel_path,
        columns=[*PANEL_KEYS, "label"],
    ).with_row_index("panel_row")
    repeat = pl.read_parquet(repeat_root / "panel" / "variant_panel.parquet").sort(
        "panel_row"
    )
    assert source.height == repeat.height == EXPECTED_VARIANTS
    assert source["panel_row"].to_list() == list(range(EXPECTED_VARIANTS))
    assert repeat["panel_row"].to_list() == list(range(EXPECTED_VARIANTS))
    assert source["match_group"].n_unique() == EXPECTED_MATCH_GROUPS
    assert source.select(pl.col("label").unique().sort()).to_series().to_list() == [
        0,
        1,
    ]
    mismatches = (
        source.select("panel_row", *PANEL_KEYS)
        .join(
            repeat.select("panel_row", *PANEL_KEYS),
            on="panel_row",
            how="inner",
            suffix="_repeat",
        )
        .filter(
            pl.any_horizontal(
                [pl.col(column) != pl.col(f"{column}_repeat") for column in PANEL_KEYS]
            )
        )
    )
    assert mismatches.is_empty(), mismatches.head(10)
    result = repeat.with_columns(source["label"].cast(pl.UInt8))
    assert result.filter(~pl.col("position_status").is_in(POSITION_STATUSES)).is_empty()
    return result


def target_definitions(
    panel: pl.DataFrame,
    *,
    minimum_class_size: int = MIN_CLASS_SIZE,
) -> tuple[dict[str, list[Target]], pl.DataFrame]:
    labels = panel["label"].to_numpy().astype(np.uint8)
    status = panel["position_status"].to_numpy()
    subsets = panel["subset"].cast(pl.String).to_numpy()
    targets: dict[str, list[Target]] = {}
    counts: list[dict[str, Any]] = []
    for stratum in STRATA:
        stratum_mask = (
            np.ones(panel.height, dtype=bool) if stratum == "all" else status == stratum
        )
        candidates = [("overall", "overall", stratum_mask)]
        candidates.extend(
            ("subset", str(subset), stratum_mask & (subsets == subset))
            for subset in sorted(np.unique(subsets).tolist())
        )
        retained: list[Target] = []
        for kind, name, mask in candidates:
            indices = np.flatnonzero(mask)
            current_labels = labels[indices]
            positive = int(current_labels.sum())
            negative = int(current_labels.size - positive)
            eligible = positive >= minimum_class_size and negative >= minimum_class_size
            counts.append(
                {
                    "stratum": stratum,
                    "target_kind": kind,
                    "target": name,
                    "n": int(indices.size),
                    "n_positive": positive,
                    "n_negative": negative,
                    "prevalence": positive / indices.size if indices.size else None,
                    "inferential": eligible,
                }
            )
            if eligible:
                retained.append(
                    Target(
                        stratum=stratum,
                        kind=kind,
                        name=name,
                        indices=indices,
                        labels=current_labels,
                    )
                )
        assert retained and retained[0].kind == "overall"
        targets[stratum] = retained
    return targets, pl.DataFrame(counts).sort("stratum", "target_kind", "target")


def _average_precision_from_sorted(
    sorted_scores: np.ndarray,
    sorted_labels: np.ndarray,
    positives: int,
) -> np.ndarray:
    cumulative_positives = np.cumsum(sorted_labels, axis=0, dtype=np.int64)
    group_end = np.empty(sorted_scores.shape, dtype=bool)
    group_end[-1, :] = True
    group_end[:-1, :] = sorted_scores[:-1, :] != sorted_scores[1:, :]
    result = np.empty(sorted_scores.shape[1], dtype=np.float64)
    for column in range(sorted_scores.shape[1]):
        ends = np.flatnonzero(group_end[:, column])
        true_positives = cumulative_positives[ends, column]
        positives_in_group = np.diff(np.concatenate(([0], true_positives)))
        precision = true_positives / (ends + 1)
        result[column] = np.sum((positives_in_group / positives) * precision)
    return result


def average_precision_both_directions(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.uint8)
    scores = np.asarray(scores)
    assert labels.ndim == 1 and scores.ndim == 2
    assert scores.shape[0] == labels.size and chunk_size > 0
    assert np.isfinite(scores).all()
    positives = int(labels.sum())
    assert 0 < positives < labels.size
    raw = np.empty(scores.shape[1], dtype=np.float64)
    negated = np.empty(scores.shape[1], dtype=np.float64)
    for start in range(0, scores.shape[1], chunk_size):
        stop = min(start + chunk_size, scores.shape[1])
        block = scores[:, start:stop]
        order = np.argsort(block, axis=0, kind="stable")
        ascending_scores = np.take_along_axis(block, order, axis=0)
        ascending_labels = labels[order]
        negated[start:stop] = _average_precision_from_sorted(
            ascending_scores, ascending_labels, positives
        )
        raw[start:stop] = _average_precision_from_sorted(
            ascending_scores[::-1, :], ascending_labels[::-1, :], positives
        )
    return raw, negated


def load_dense_pair(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    table = pq.read_table(
        path,
        columns=["panel_row", "feature_id", "ref_activation", "alt_activation"],
        memory_map=True,
    )
    panel_rows = table["panel_row"].to_numpy(zero_copy_only=False)
    feature_ids = table["feature_id"].to_numpy(zero_copy_only=False)
    ref_values = table["ref_activation"].to_numpy(zero_copy_only=False)
    alt_values = table["alt_activation"].to_numpy(zero_copy_only=False)
    assert panel_rows.dtype == np.uint32 and feature_ids.dtype == np.uint32
    assert ref_values.dtype == np.float32 and alt_values.dtype == np.float32
    assert panel_rows.size > 0
    assert int(panel_rows.max()) < EXPECTED_VARIANTS
    assert int(feature_ids.max()) < D_SAE
    ordered = (panel_rows[1:] > panel_rows[:-1]) | (
        (panel_rows[1:] == panel_rows[:-1]) & (feature_ids[1:] > feature_ids[:-1])
    )
    assert ordered.all()
    assert np.isfinite(ref_values).all() and np.isfinite(alt_values).all()
    assert np.all((ref_values != 0) | (alt_values != 0))
    ref = np.zeros((EXPECTED_VARIANTS, D_SAE), dtype=np.float32)
    alt = np.zeros((EXPECTED_VARIANTS, D_SAE), dtype=np.float32)
    ref[panel_rows, feature_ids] = ref_values
    alt[panel_rows, feature_ids] = alt_values
    sparse_rows = int(panel_rows.size)
    del table, panel_rows, feature_ids, ref_values, alt_values, ordered
    gc.collect()
    return ref, alt, sparse_rows


def analyze_target(
    response: np.ndarray,
    target: Target,
    *,
    arm: str,
    orientation: str,
    response_name: str,
    ap_chunk_size: int,
) -> pl.DataFrame:
    matrix = response[target.indices, :]
    labels = target.labels
    support = np.count_nonzero(matrix, axis=0)
    minimum = matrix.min(axis=0)
    maximum = matrix.max(axis=0)
    eligible = np.flatnonzero((support >= MIN_NONZERO_SUPPORT) & (minimum != maximum))
    assert eligible.size > 0
    values = matrix[:, eligible]
    positive = values[labels == 1, :]
    negative = values[labels == 0, :]
    n_positive = positive.shape[0]
    n_negative = negative.shape[0]
    assert n_positive >= MIN_CLASS_SIZE and n_negative >= MIN_CLASS_SIZE

    mean_positive = positive.mean(axis=0, dtype=np.float64)
    mean_negative = negative.mean(axis=0, dtype=np.float64)
    variance_positive = positive.var(axis=0, ddof=1, dtype=np.float64)
    variance_negative = negative.var(axis=0, ddof=1, dtype=np.float64)
    mean_difference = mean_positive - mean_negative
    pooled_sd = np.sqrt((variance_positive + variance_negative) / 2)
    standardized = np.divide(
        mean_difference,
        pooled_sd,
        out=np.full(mean_difference.shape, np.nan, dtype=np.float64),
        where=pooled_sd > 0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        welch = stats.ttest_ind(
            positive, negative, axis=0, equal_var=False, nan_policy="raise"
        )
        mann = stats.mannwhitneyu(
            positive,
            negative,
            axis=0,
            alternative="two-sided",
            method="asymptotic",
        )
    welch_statistic = np.asarray(welch.statistic, dtype=np.float64)
    welch_p = np.asarray(welch.pvalue, dtype=np.float64)
    u_statistic = np.asarray(mann.statistic, dtype=np.float64)
    mann_p = np.asarray(mann.pvalue, dtype=np.float64)
    rank_biserial = 2 * u_statistic / (n_positive * n_negative) - 1
    auprc, auprc_negated = average_precision_both_directions(
        labels, values, chunk_size=ap_chunk_size
    )
    best_is_raw = auprc >= auprc_negated
    rows = eligible.size
    return pl.DataFrame(
        {
            "arm": [arm] * rows,
            "block": np.full(rows, BLOCK_BY_ARM[arm], dtype=np.uint8),
            "budget": np.full(rows, 25_000_200, dtype=np.uint32),
            "orientation": [orientation] * rows,
            "pooling": ["focal"] * rows,
            "response": [response_name] * rows,
            "response_role": ["primary"] * rows,
            "target_kind": [target.kind] * rows,
            "target": [target.name] * rows,
            "feature_id": eligible.astype(np.uint32),
            "n": np.full(rows, labels.size, dtype=np.uint32),
            "n_positive": np.full(rows, n_positive, dtype=np.uint32),
            "n_negative": np.full(rows, n_negative, dtype=np.uint32),
            "prevalence": np.full(rows, n_positive / labels.size),
            "nonzero_support": support[eligible].astype(np.uint32),
            "mean_positive": mean_positive,
            "mean_negative": mean_negative,
            "mean_difference": mean_difference,
            "standardized_mean_difference": standardized,
            "welch_statistic": welch_statistic,
            "welch_p": welch_p,
            "u_statistic": u_statistic,
            "rank_biserial": rank_biserial,
            "mann_whitney_p": mann_p,
            "auprc": auprc,
            "auprc_negated": auprc_negated,
            "best_auprc": np.maximum(auprc, auprc_negated),
            "best_auprc_direction": np.where(best_is_raw, "higher", "lower"),
        }
    )


def correct_family(frames: list[pl.DataFrame]) -> pl.DataFrame:
    family = pl.concat(frames, how="vertical")
    welch_q = bh_adjust(family["welch_p"].to_numpy())
    mann_q = bh_adjust(family["mann_whitney_p"].to_numpy())
    return family.with_columns(
        pl.Series("welch_q", welch_q),
        pl.Series("mann_whitney_q", mann_q),
        pl.Series("minimum_q", np.fmin(welch_q, mann_q)),
    )


def add_stratum_calls(family: pl.DataFrame, stratum: str) -> pl.DataFrame:
    same_direction = (
        (pl.col("mean_difference") > 0) & (pl.col("rank_biserial") > 0)
    ) | ((pl.col("mean_difference") < 0) & (pl.col("rank_biserial") < 0))
    both = (pl.col("welch_q") <= FDR_THRESHOLD) & (
        pl.col("mann_whitney_q") <= FDR_THRESHOLD
    )
    return family.with_columns(
        pl.lit(stratum).alias("repeat_stratum"),
        pl.max_horizontal("welch_q", "mann_whitney_q").alias("maximum_q"),
        (both & same_direction).alias("concordant_discovery"),
    ).select(
        "arm",
        "block",
        "budget",
        "orientation",
        "pooling",
        "response",
        "response_role",
        "repeat_stratum",
        pl.exclude(
            "arm",
            "block",
            "budget",
            "orientation",
            "pooling",
            "response",
            "response_role",
            "repeat_stratum",
        ),
    )


def assert_global_reproduction(
    observed: pl.DataFrame, expected_path: Path
) -> dict[str, float | int]:
    expected = pl.read_parquet(expected_path)
    keys = ["target_kind", "target", "feature_id"]
    observed = observed.sort(keys)
    expected = expected.sort(keys)
    assert observed.columns == expected.columns
    assert observed.schema == expected.schema and observed.height == expected.height
    max_error = 0.0
    for column, dtype in observed.schema.items():
        left = observed[column].to_numpy()
        right = expected[column].to_numpy()
        if dtype.is_float():
            finite = np.isfinite(left) & np.isfinite(right)
            if finite.any():
                max_error = max(
                    max_error, float(np.max(np.abs(left[finite] - right[finite])))
                )
            np.testing.assert_allclose(
                left, right, rtol=1e-12, atol=1e-12, equal_nan=True
            )
        else:
            assert np.array_equal(left, right), column
    return {"rows": observed.height, "maximum_absolute_error": max_error}


def target_summary(frame: pl.DataFrame) -> pl.DataFrame:
    keys = [
        "arm",
        "block",
        "orientation",
        "response",
        "repeat_stratum",
        "target_kind",
        "target",
    ]
    return (
        frame.group_by(keys)
        .agg(
            pl.len().alias("eligible_features"),
            pl.col("concordant_discovery").sum().alias("discoveries"),
            pl.col("best_auprc").max().alias("best_auprc"),
            pl.col("maximum_q").min().alias("minimum_maximum_q"),
            pl.col("prevalence").first().alias("prevalence"),
        )
        .sort(keys)
    )


def _spearman_or_nan(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.all(left == left[0]) or np.all(right == right[0]):
        return math.nan
    return float(stats.spearmanr(left, right).statistic)


def strand_overlap(frame: pl.DataFrame) -> pl.DataFrame:
    keys = ["arm", "block", "response", "repeat_stratum", "target_kind", "target"]
    rows: list[dict[str, Any]] = []
    for key, group in frame.group_by(keys, maintain_order=True):
        forward = group.filter(pl.col("orientation") == "forward")
        reverse = group.filter(pl.col("orientation") == "reverse_complement")
        if forward.is_empty() or reverse.is_empty():
            continue
        forward_set = set(
            forward.filter(pl.col("concordant_discovery"))["feature_id"].to_list()
        )
        reverse_set = set(
            reverse.filter(pl.col("concordant_discovery"))["feature_id"].to_list()
        )
        joined = forward.select(
            "feature_id", pl.col("rank_biserial").alias("forward_effect")
        ).join(
            reverse.select(
                "feature_id", pl.col("rank_biserial").alias("reverse_effect")
            ),
            on="feature_id",
        )
        shared = forward_set & reverse_set
        union = forward_set | reverse_set
        shared_frame = joined.filter(pl.col("feature_id").is_in(sorted(shared)))
        sign_concordance = (
            float(
                np.mean(
                    np.sign(shared_frame["forward_effect"].to_numpy())
                    == np.sign(shared_frame["reverse_effect"].to_numpy())
                )
            )
            if shared_frame.height
            else math.nan
        )
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "forward_discoveries": len(forward_set),
                "reverse_discoveries": len(reverse_set),
                "shared_discoveries": len(shared),
                "union_discoveries": len(union),
                "discovery_jaccard": len(shared) / len(union) if union else math.nan,
                "shared_eligible_features": joined.height,
                "eligible_effect_spearman": _spearman_or_nan(
                    joined["forward_effect"].to_numpy(),
                    joined["reverse_effect"].to_numpy(),
                ),
                "shared_discovery_sign_concordance": sign_concordance,
            }
        )
    return pl.DataFrame(rows).sort(keys)


def repeat_free_retention(frame: pl.DataFrame) -> pl.DataFrame:
    keys = ["arm", "block", "orientation", "response", "target_kind", "target"]
    rows: list[dict[str, Any]] = []
    for key, group in frame.group_by(keys, maintain_order=True):
        all_frame = group.filter(pl.col("repeat_stratum") == "all")
        free = group.filter(pl.col("repeat_stratum") == "repeat_free_window")
        if all_frame.is_empty() or free.is_empty():
            continue
        global_set = set(
            all_frame.filter(pl.col("concordant_discovery"))["feature_id"].to_list()
        )
        free_set = set(
            free.filter(pl.col("concordant_discovery"))["feature_id"].to_list()
        )
        joined = all_frame.select(
            "feature_id", pl.col("rank_biserial").alias("global_effect")
        ).join(
            free.select("feature_id", pl.col("rank_biserial").alias("free_effect")),
            on="feature_id",
        )
        global_joined = joined.filter(pl.col("feature_id").is_in(sorted(global_set)))
        retained = global_set & free_set
        sign_concordance = (
            float(
                np.mean(
                    np.sign(global_joined["global_effect"].to_numpy())
                    == np.sign(global_joined["free_effect"].to_numpy())
                )
            )
            if global_joined.height
            else math.nan
        )
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "global_discoveries": len(global_set),
                "repeat_free_discoveries": len(free_set),
                "retained_global_discoveries": len(retained),
                "retention_fraction": (
                    len(retained) / len(global_set) if global_set else math.nan
                ),
                "global_discoveries_eligible_repeat_free": global_joined.height,
                "global_discovery_effect_spearman": _spearman_or_nan(
                    global_joined["global_effect"].to_numpy(),
                    global_joined["free_effect"].to_numpy(),
                ),
                "global_discovery_sign_concordance": sign_concordance,
            }
        )
    return pl.DataFrame(rows).sort(keys)


def repeat_inventory_sets(
    reference_root: Path, paired_root: Path
) -> tuple[
    dict[tuple[str, str], set[int]],
    dict[tuple[str, str, str], set[int]],
]:
    reference: dict[tuple[str, str], set[int]] = {}
    paired: dict[tuple[str, str, str], set[int]] = {}
    for arm in ARMS:
        for orientation in ORIENTATIONS:
            reference_frame = pl.read_parquet(
                reference_root
                / "associations"
                / "families"
                / arm
                / orientation
                / "repeat.parquet"
            )
            reference[(arm, orientation)] = set(
                reference_frame.filter(pl.col("concordant_positive_association"))[
                    "feature_id"
                ].to_list()
            )
            for response in RESPONSES:
                paired_frame = pl.read_parquet(
                    paired_root
                    / "paired"
                    / "families"
                    / arm
                    / "broad"
                    / "all"
                    / f"{response}.parquet"
                ).filter(pl.col("orientation") == orientation)
                call = (
                    "positive_mutation_association"
                    if response == "abs_delta"
                    else "concordant_association"
                )
                paired[(arm, orientation, response)] = set(
                    paired_frame.filter(pl.col(call))["feature_id"].to_list()
                )
    return reference, paired


def inventory_overlap(
    frame: pl.DataFrame,
    reference: dict[tuple[str, str], set[int]],
    paired: dict[tuple[str, str, str], set[int]],
) -> pl.DataFrame:
    keys = [
        "arm",
        "block",
        "orientation",
        "response",
        "repeat_stratum",
        "target_kind",
        "target",
    ]
    rows: list[dict[str, Any]] = []
    for key, group in frame.group_by(keys, maintain_order=True):
        values = dict(zip(keys, key, strict=True))
        discoveries = set(
            group.filter(pl.col("concordant_discovery"))["feature_id"].to_list()
        )
        reference_set = reference[(values["arm"], values["orientation"])]
        paired_set = paired[(values["arm"], values["orientation"], values["response"])]
        rows.append(
            {
                **values,
                "label_discoveries": len(discoveries),
                "reference_repeat_features": len(reference_set),
                "paired_repeat_response_features": len(paired_set),
                "label_reference_repeat_overlap": len(discoveries & reference_set),
                "label_paired_repeat_overlap": len(discoveries & paired_set),
                "label_both_repeat_inventories_overlap": len(
                    discoveries & reference_set & paired_set
                ),
            }
        )
    return pl.DataFrame(rows).sort(keys)


def contingency_tables(panel: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    labels = panel["label"].to_numpy()
    status = panel["position_status"].to_numpy()
    subsets = panel["subset"].cast(pl.String).to_numpy()
    targets = [("overall", np.ones(panel.height, dtype=bool))]
    targets.extend(
        (str(subset), subsets == subset) for subset in sorted(np.unique(subsets))
    )
    omnibus: list[dict[str, Any]] = []
    pairwise: list[dict[str, Any]] = []
    for target, mask in targets:
        table = np.asarray(
            [
                [
                    int(np.count_nonzero(mask & (status == current) & (labels == 0))),
                    int(np.count_nonzero(mask & (status == current) & (labels == 1))),
                ]
                for current in POSITION_STATUSES
            ],
            dtype=np.int64,
        )
        assert table.sum() == int(mask.sum())
        active = table.sum(axis=1) > 0
        active_table = table[active]
        if active_table.shape[0] >= 2 and np.all(active_table.sum(axis=0) > 0):
            chi2 = stats.chi2_contingency(active_table, correction=False)
            chi2_statistic = float(chi2.statistic)
            chi2_dof = int(chi2.dof)
            chi2_p = float(chi2.pvalue)
        else:
            chi2_statistic = math.nan
            chi2_dof = 0
            chi2_p = math.nan
        omnibus.append(
            {
                "target": target,
                "n": int(table.sum()),
                "chi2": chi2_statistic,
                "degrees_freedom": chi2_dof,
                "p": chi2_p,
            }
        )
        for left in ("focal_repeat", "near_repeat"):
            left_row = table[POSITION_STATUSES.index(left)]
            free_row = table[POSITION_STATUSES.index("repeat_free_window")]
            fisher = stats.fisher_exact(np.stack((left_row, free_row)))
            pairwise.append(
                {
                    "target": target,
                    "contrast": f"{left}_vs_repeat_free_window",
                    "left_negative": int(left_row[0]),
                    "left_positive": int(left_row[1]),
                    "repeat_free_negative": int(free_row[0]),
                    "repeat_free_positive": int(free_row[1]),
                    "odds_ratio": float(fisher.statistic),
                    "p": float(fisher.pvalue),
                }
            )
    omnibus_frame = pl.DataFrame(omnibus).with_columns(
        pl.Series("q", bh_adjust(np.asarray([row["p"] for row in omnibus])))
    )
    pairwise_frame = pl.DataFrame(pairwise).with_columns(
        pl.Series("q", bh_adjust(np.asarray([row["p"] for row in pairwise])))
    )
    return omnibus_frame.sort("target"), pairwise_frame.sort("target", "contrast")


def artifact_record(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def analyze(
    *,
    label_panel_path: Path,
    repeat_panel_root: Path,
    activation_root: Path,
    global_association_root: Path,
    reference_association_root: Path,
    paired_repeat_root: Path,
    output_dir: Path,
    ap_chunk_size: int,
) -> dict[str, Any]:
    assert not output_dir.exists() and ap_chunk_size > 0
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert_commit(experiment_commit)
    assert os.environ.get("RUN_ID") == RUN_ID
    started = time.perf_counter()

    repeat_manifest = verify_outer_archive(
        repeat_panel_root,
        VARIANT_PANEL_ARCHIVE_SHA256,
        "outcome_blind_paired_repeat_variant_panel",
    )
    activation_manifest = verify_activation_archive(activation_root)
    global_manifest = verify_inner_archive(
        global_association_root,
        GLOBAL_ASSOCIATION_MANIFEST_SHA256,
        GLOBAL_ASSOCIATION_RUN_ID,
    )
    reference_manifest = verify_outer_archive(
        reference_association_root,
        REFERENCE_ASSOCIATION_ARCHIVE_SHA256,
        "frozen_reference_repeat_capacity_associations",
    )
    paired_manifest = verify_outer_archive(
        paired_repeat_root,
        PAIRED_REPEAT_ARCHIVE_SHA256,
        "frozen_paired_repeat_variant_delta_associations",
    )
    panel = load_aligned_panel(label_panel_path, repeat_panel_root)
    targets, cell_counts = target_definitions(panel)
    omnibus, pairwise = contingency_tables(panel)
    reference_sets, paired_sets = repeat_inventory_sets(
        reference_association_root, paired_repeat_root
    )

    output_dir.mkdir(parents=True)
    artifacts: dict[str, dict[str, Any]] = {}
    all_families: list[pl.DataFrame] = []
    reproduction: dict[str, Any] = {}
    family_summaries: dict[str, Any] = {}
    for arm in ARMS:
        for orientation in ORIENTATIONS:
            sparse_path = activation_root / arm / f"sae_focal_{orientation}.parquet"
            ref, alt, sparse_rows = load_dense_pair(sparse_path)
            assert (
                sparse_rows == activation_manifest["outputs"][arm][orientation]["rows"]
            )
            delta = alt - ref
            assert delta.dtype == np.float32 and np.isfinite(delta).all()
            responses = {"abs_delta": np.abs(delta), "delta": delta}
            for response_name, response in responses.items():
                for stratum in STRATA:
                    frames = [
                        analyze_target(
                            response,
                            target,
                            arm=arm,
                            orientation=orientation,
                            response_name=response_name,
                            ap_chunk_size=ap_chunk_size,
                        )
                        for target in targets[stratum]
                    ]
                    family = correct_family(frames)
                    relative = (
                        Path("families")
                        / arm
                        / orientation
                        / stratum
                        / f"{response_name}.parquet"
                    )
                    if stratum == "all":
                        reproduction[str(relative)] = assert_global_reproduction(
                            family,
                            global_association_root
                            / "families"
                            / arm
                            / orientation
                            / f"{response_name}.parquet",
                        )
                    family = add_stratum_calls(family, stratum)
                    path = output_dir / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    family.write_parquet(path, compression="zstd")
                    artifacts[str(relative)] = artifact_record(path)
                    family_summaries[str(relative)] = {
                        "rows": family.height,
                        "targets": family["target"].n_unique(),
                        "features": family["feature_id"].n_unique(),
                        "discoveries": int(family["concordant_discovery"].sum()),
                        "best_auprc": float(family["best_auprc"].max()),
                        "minimum_maximum_q": float(family["maximum_q"].min()),
                    }
                    all_families.append(family)
                    del frames, family
                    gc.collect()
            del ref, alt, delta, responses
            gc.collect()

    combined = pl.concat(all_families, how="vertical")
    summaries = {
        "target_summary.parquet": target_summary(combined),
        "strand_overlap.parquet": strand_overlap(combined),
        "repeat_free_retention.parquet": repeat_free_retention(combined),
        "inventory_overlap.parquet": inventory_overlap(
            combined, reference_sets, paired_sets
        ),
        "feature9086.parquet": combined.filter(
            (pl.col("block") == 19) & (pl.col("feature_id") == FEATURE_OF_INTEREST)
        ).sort("orientation", "response", "repeat_stratum", "target_kind", "target"),
        "stratum_target_counts.parquet": cell_counts,
        "label_repeat_status_omnibus.parquet": omnibus,
        "label_repeat_status_pairwise.parquet": pairwise,
    }
    for name, frame in summaries.items():
        path = output_dir / name
        frame.write_parquet(path, compression="zstd")
        artifacts[name] = artifact_record(path)

    result = {
        "issue": ISSUE,
        "run_id": RUN_ID,
        "analysis_status": "frozen_repeat_aware_mendelian_label_sensitivity",
        "created_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "experiment_commit": experiment_commit,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "polars": pl.__version__,
        "scipy": scipy.__version__,
        "inputs": {
            "label_panel": {
                "dataset": SOURCE_DATASET_ID,
                "revision": SOURCE_DATASET_REVISION,
                "sha256": SOURCE_PANEL_SHA256,
                "rows": EXPECTED_VARIANTS,
            },
            "repeat_panel": {
                "run_id": repeat_manifest["run_id"],
                "archive_manifest_sha256": VARIANT_PANEL_ARCHIVE_SHA256,
            },
            "paired_activations": {
                "run_id": activation_manifest["run_id"],
                "manifest_sha256": PAIRED_ACTIVATION_MANIFEST_SHA256,
            },
            "global_associations": {
                "run_id": global_manifest["run_id"],
                "manifest_sha256": GLOBAL_ASSOCIATION_MANIFEST_SHA256,
            },
            "reference_repeat_associations": {
                "run_id": reference_manifest["run_id"],
                "archive_manifest_sha256": REFERENCE_ASSOCIATION_ARCHIVE_SHA256,
            },
            "paired_repeat_associations": {
                "run_id": paired_manifest["run_id"],
                "archive_manifest_sha256": PAIRED_REPEAT_ARCHIVE_SHA256,
            },
        },
        "protocol": {
            "layers": [1, 10, 19],
            "checkpoint": "25M",
            "orientations": list(ORIENTATIONS),
            "responses": list(RESPONSES),
            "repeat_strata": list(STRATA),
            "minimum_nonzero_support": MIN_NONZERO_SUPPORT,
            "minimum_class_size": MIN_CLASS_SIZE,
            "tests": ["Welch t", "Mann-Whitney U"],
            "fdr": (
                "BH within layer/orientation/response/repeat-stratum/statistic, "
                "joint across eligible overall/subset feature-target pairs"
            ),
            "discovery": "both tests q<=0.05 with concordant effect direction",
            "descriptive_metric": "AUPRC in raw and sign-reversed direction",
            "uses_chromosome_split": False,
            "inventory_overlap_is_descriptive": True,
            "ap_chunk_size": ap_chunk_size,
        },
        "global_reproduction": reproduction,
        "family_summaries": family_summaries,
        "artifacts": artifacts,
    }
    write_json(output_dir / "results.json", result)
    artifacts["results.json"] = artifact_record(output_dir / "results.json")
    write_json(output_dir / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-panel", type=Path, required=True)
    parser.add_argument("--repeat-panel-root", type=Path, required=True)
    parser.add_argument("--activation-root", type=Path, required=True)
    parser.add_argument("--global-association-root", type=Path, required=True)
    parser.add_argument("--reference-association-root", type=Path, required=True)
    parser.add_argument("--paired-repeat-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ap-chunk-size", type=int, default=128)
    args = parser.parse_args()
    result = analyze(
        label_panel_path=args.label_panel,
        repeat_panel_root=args.repeat_panel_root,
        activation_root=args.activation_root,
        global_association_root=args.global_association_root,
        reference_association_root=args.reference_association_root,
        paired_repeat_root=args.paired_repeat_root,
        output_dir=args.output_dir,
        ap_chunk_size=args.ap_chunk_size,
    )
    print(json.dumps(result["family_summaries"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
