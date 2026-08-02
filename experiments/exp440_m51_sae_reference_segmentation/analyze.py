"""Run the preregistered issue-440 reference-state association scan."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import norm
from scipy.stats import t as student_t

from build_panel import REFERENCE_CLASSES
from extract_focal import (
    BLOCK_INDICES,
    D_SAE,
    EXTRACTION_RUN_ID,
    ISSUE,
    ORIENTATIONS,
    PANEL_MANIFEST_SHA256,
    PANEL_RUN_ID,
    PANEL_SHA256,
    arm_label,
    assert_commit,
    load_panel,
    sha256_file,
    write_json,
)

ASSOCIATION_RUN_ID = "dna-exp440-reference-state-associations-seed288-r1"
EXTRACTION_COMMIT = "c3e662c868e4c55af64f1f7ed9fa87afcd94c175"
MINIMUM_SUPPORT = 32
PRIMARY_Q = 0.05
PRIMARY_ABS_RANK_BISERIAL = 0.1
VIEWS = (*ORIENTATIONS, "same_id_mean", "same_id_max")


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    assert values.ndim == 1 and np.isfinite(values).all()
    assert np.all((values >= 0) & (values <= 1))
    count = len(values)
    if count == 0:
        return values.copy()
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    adjusted = ranked * count / np.arange(1, count + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty(count, dtype=np.float64)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def _group_positive_values(
    values: np.ndarray, class_codes: np.ndarray, class_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert values.ndim == class_codes.ndim == 1 and len(values) == len(class_codes)
    assert len(values) > 0 and np.all(values > 0) and np.isfinite(values).all()
    order = np.argsort(-values, kind="stable")
    sorted_values = values[order]
    sorted_codes = class_codes[order]
    starts = np.empty(len(values), dtype=bool)
    starts[0] = True
    starts[1:] = sorted_values[1:] != sorted_values[:-1]
    group_ids = np.cumsum(starts) - 1
    group_count = int(group_ids[-1] + 1)
    by_group_class = np.zeros((group_count, class_count), dtype=np.float64)
    np.add.at(by_group_class, (group_ids, sorted_codes), 1.0)
    group_sizes = by_group_class.sum(axis=1)
    assert int(group_sizes.sum()) == len(values)
    return by_group_class, group_sizes, np.cumsum(group_sizes)


def feature_statistics(
    values: np.ndarray,
    class_codes: np.ndarray,
    class_sizes: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute seven one-vs-rest tests without materializing feature zeros."""
    values = np.asarray(values, dtype=np.float64)
    class_codes = np.asarray(class_codes, dtype=np.int64)
    class_sizes = np.asarray(class_sizes, dtype=np.int64)
    class_count = len(class_sizes)
    total_rows = int(class_sizes.sum())
    assert values.ndim == class_codes.ndim == 1 and len(values) == len(class_codes)
    assert len(values) > 0 and np.all(values > 0) and np.isfinite(values).all()
    assert np.all((class_codes >= 0) & (class_codes < class_count))
    assert np.all(class_sizes > 1) and len(values) <= total_rows

    support_class = np.bincount(class_codes, minlength=class_count).astype(np.int64)
    assert np.all(support_class <= class_sizes)
    sums_class = np.bincount(class_codes, weights=values, minlength=class_count).astype(
        np.float64
    )
    squares_class = np.bincount(
        class_codes, weights=values * values, minlength=class_count
    ).astype(np.float64)
    rest_sizes = total_rows - class_sizes
    sums_rest = values.sum() - sums_class
    squares_rest = np.dot(values, values) - squares_class
    means_class = sums_class / class_sizes
    means_rest = sums_rest / rest_sizes
    variances_class = np.maximum(
        (squares_class - sums_class * sums_class / class_sizes) / (class_sizes - 1),
        0.0,
    )
    variances_rest = np.maximum(
        (squares_rest - sums_rest * sums_rest / rest_sizes) / (rest_sizes - 1),
        0.0,
    )
    mean_difference = means_class - means_rest
    standard_error_squared = variances_class / class_sizes + variances_rest / rest_sizes
    welch_t = np.zeros(class_count, dtype=np.float64)
    nonzero_error = standard_error_squared > 0
    welch_t[nonzero_error] = mean_difference[nonzero_error] / np.sqrt(
        standard_error_squared[nonzero_error]
    )
    welch_t[~nonzero_error & (mean_difference > 0)] = np.inf
    welch_t[~nonzero_error & (mean_difference < 0)] = -np.inf
    denominator = (variances_class / class_sizes) ** 2 / (class_sizes - 1) + (
        variances_rest / rest_sizes
    ) ** 2 / (rest_sizes - 1)
    welch_df = np.full(class_count, np.inf, dtype=np.float64)
    valid_df = denominator > 0
    welch_df[valid_df] = standard_error_squared[valid_df] ** 2 / denominator[valid_df]
    welch_p = 2 * student_t.sf(np.abs(welch_t), welch_df)
    welch_p[~nonzero_error & (mean_difference == 0)] = 1.0
    welch_p[~nonzero_error & (mean_difference != 0)] = 0.0

    pooled_variance = (
        (class_sizes - 1) * variances_class + (rest_sizes - 1) * variances_rest
    ) / (total_rows - 2)
    cohen_d = np.zeros(class_count, dtype=np.float64)
    nonzero_pooled = pooled_variance > 0
    cohen_d[nonzero_pooled] = mean_difference[nonzero_pooled] / np.sqrt(
        pooled_variance[nonzero_pooled]
    )
    cohen_d[~nonzero_pooled & (mean_difference > 0)] = np.inf
    cohen_d[~nonzero_pooled & (mean_difference < 0)] = -np.inf

    by_group_class, group_sizes, cumulative_sizes = _group_positive_values(
        values, class_codes, class_count
    )
    cumulative_class = np.cumsum(by_group_class, axis=0)
    precision = cumulative_class / cumulative_sizes[:, None]
    zero_class = class_sizes - support_class
    prevalence = class_sizes / total_rows
    auprc = (by_group_class * precision).sum(axis=0) / class_sizes
    auprc += zero_class / class_sizes * prevalence

    positive_count = len(values)
    zero_count = total_rows - positive_count
    group_starts = cumulative_sizes - group_sizes
    positive_average_rank = (
        positive_count - cumulative_sizes + 1 + positive_count - group_starts
    ) / 2
    full_average_rank = zero_count + positive_average_rank
    zero_average_rank = (zero_count + 1) / 2
    rank_sum_class = zero_class * zero_average_rank + (
        by_group_class * full_average_rank[:, None]
    ).sum(axis=0)
    mwu_u = rank_sum_class - class_sizes * (class_sizes + 1) / 2
    pair_count = class_sizes * rest_sizes
    rank_biserial = 2 * mwu_u / pair_count - 1
    tie_sum = float(zero_count**3 - zero_count) + float(
        np.sum(group_sizes**3 - group_sizes)
    )
    mwu_variance = (
        pair_count / 12 * ((total_rows + 1) - tie_sum / (total_rows * (total_rows - 1)))
    )
    mwu_p = np.ones(class_count, dtype=np.float64)
    nonzero_mwu_variance = mwu_variance > 0
    assert np.allclose(
        mwu_u[~nonzero_mwu_variance], pair_count[~nonzero_mwu_variance] / 2
    )
    mwu_z = (
        mwu_u[nonzero_mwu_variance] - pair_count[nonzero_mwu_variance] / 2
    ) / np.sqrt(mwu_variance[nonzero_mwu_variance])
    mwu_p[nonzero_mwu_variance] = 2 * norm.sf(np.abs(mwu_z))

    for array in (
        means_class,
        means_rest,
        mean_difference,
        welch_p,
        mwu_u,
        mwu_p,
        rank_biserial,
        auprc,
    ):
        assert np.isfinite(array).all()
    assert np.all((welch_p >= 0) & (welch_p <= 1))
    assert np.all((mwu_p >= 0) & (mwu_p <= 1))
    assert np.all((auprc >= 0) & (auprc <= 1))
    assert np.all((rank_biserial >= -1) & (rank_biserial <= 1))
    return {
        "nonzero_support_class": support_class,
        "mean_class": means_class,
        "mean_rest": means_rest,
        "mean_difference": mean_difference,
        "welch_t": welch_t,
        "welch_df": welch_df,
        "welch_p": welch_p,
        "cohen_d": cohen_d,
        "mwu_u": mwu_u,
        "mwu_p": mwu_p,
        "rank_biserial": rank_biserial,
        "auprc": auprc,
    }


def sparse_view(
    forward: pl.DataFrame, reverse: pl.DataFrame, view: str
) -> pl.DataFrame:
    required = {"panel_row", "feature_id", "activation"}
    assert set(forward.columns) == required and set(reverse.columns) == required
    if view == "forward":
        result = forward
    elif view == "reverse_complement":
        result = reverse
    else:
        assert view in {"same_id_mean", "same_id_max"}
        result = (
            forward.rename({"activation": "forward_activation"})
            .join(
                reverse.rename({"activation": "reverse_activation"}),
                on=["panel_row", "feature_id"],
                how="full",
                coalesce=True,
            )
            .with_columns(
                pl.col("forward_activation").fill_null(0.0),
                pl.col("reverse_activation").fill_null(0.0),
            )
        )
        if view == "same_id_mean":
            activation = (
                pl.col("forward_activation") + pl.col("reverse_activation")
            ) / 2
        else:
            activation = pl.max_horizontal("forward_activation", "reverse_activation")
        result = result.select(
            "panel_row", "feature_id", activation.alias("activation")
        )
    assert result.filter(pl.col("activation") <= 0).is_empty()
    assert result.filter(~pl.col("activation").is_finite()).is_empty()
    assert result.select(pl.struct("panel_row", "feature_id").n_unique()).item() == len(
        result
    )
    return result


def analyze_sparse_table(
    sparse: pl.DataFrame,
    *,
    panel_classes: np.ndarray,
    class_names: tuple[str, ...] = REFERENCE_CLASSES,
    d_sae: int = D_SAE,
    minimum_support: int = MINIMUM_SUPPORT,
) -> pl.DataFrame:
    panel_classes = np.asarray(panel_classes, dtype=np.int64)
    assert panel_classes.ndim == 1
    class_sizes = np.bincount(panel_classes, minlength=len(class_names))
    assert len(class_sizes) == len(class_names) and np.all(class_sizes > 1)
    assert sparse.filter(pl.col("panel_row") >= len(panel_classes)).is_empty()
    assert sparse.filter(pl.col("feature_id") >= d_sae).is_empty()
    ordered = sparse.sort("feature_id", "panel_row")
    feature_ids = ordered["feature_id"].to_numpy().astype(np.int64, copy=False)
    panel_rows = ordered["panel_row"].to_numpy().astype(np.int64, copy=False)
    activations = ordered["activation"].to_numpy().astype(np.float64, copy=False)
    counts = np.bincount(feature_ids, minlength=d_sae)
    eligible = np.flatnonzero(counts >= minimum_support)
    offsets = np.concatenate(([0], np.cumsum(counts)))

    columns: dict[str, list[Any]] = {
        "feature_id": [],
        "reference_class": [],
        "n_class": [],
        "n_rest": [],
        "nonzero_support_total": [],
        "nonzero_support_class": [],
        "mean_class": [],
        "mean_rest": [],
        "mean_difference": [],
        "welch_t": [],
        "welch_df": [],
        "welch_p": [],
        "cohen_d": [],
        "mwu_u": [],
        "mwu_p": [],
        "rank_biserial": [],
        "auprc": [],
    }
    for feature_id in eligible:
        start, end = offsets[feature_id : feature_id + 2]
        rows = panel_rows[start:end]
        values = activations[start:end]
        assert len(rows) == counts[feature_id]
        assert len(np.unique(rows)) == len(rows)
        statistics = feature_statistics(values, panel_classes[rows], class_sizes)
        for class_index, reference_class in enumerate(class_names):
            columns["feature_id"].append(int(feature_id))
            columns["reference_class"].append(reference_class)
            columns["n_class"].append(int(class_sizes[class_index]))
            columns["n_rest"].append(int(len(panel_classes) - class_sizes[class_index]))
            columns["nonzero_support_total"].append(int(counts[feature_id]))
            for name, values_by_class in statistics.items():
                columns[name].append(values_by_class[class_index].item())

    result = pl.DataFrame(columns)
    assert result.height == len(eligible) * len(class_names)
    welch_q = np.empty(result.height, dtype=np.float64)
    mwu_q = np.empty(result.height, dtype=np.float64)
    for reference_class in class_names:
        indices = np.flatnonzero(
            result["reference_class"].to_numpy() == reference_class
        )
        assert len(indices) == len(eligible)
        welch_q[indices] = bh_adjust(result["welch_p"].to_numpy()[indices])
        mwu_q[indices] = bh_adjust(result["mwu_p"].to_numpy()[indices])
    return result.with_columns(
        pl.Series("welch_q", welch_q),
        pl.Series("mwu_q", mwu_q),
    ).with_columns(
        (
            (pl.col("welch_q") < PRIMARY_Q)
            & (pl.col("mwu_q") < PRIMARY_Q)
            & (pl.col("rank_biserial").abs() >= PRIMARY_ABS_RANK_BISERIAL)
        ).alias("primary_association")
    )


def validate_extraction(extraction_root: Path) -> dict[str, Any]:
    manifest_path = extraction_root / "manifest.json"
    assert manifest_path.is_file()
    expected_manifest_sha256 = os.environ.get("EXTRACTION_MANIFEST_SHA256", "")
    assert len(expected_manifest_sha256) == 64
    assert sha256_file(manifest_path) == expected_manifest_sha256
    manifest = json.loads(manifest_path.read_text())
    assert manifest["issue"] == ISSUE and manifest["run_id"] == EXTRACTION_RUN_ID
    assert manifest["experiment_commit"] == EXTRACTION_COMMIT
    assert manifest["analysis_status"] == "frozen_reference_state_sae_extraction"
    assert manifest["panel"]["run_id"] == PANEL_RUN_ID
    assert manifest["panel"]["panel_sha256"] == PANEL_SHA256
    assert manifest["panel"]["manifest_sha256"] == PANEL_MANIFEST_SHA256
    for relative, expected in manifest["artifacts"].items():
        path = extraction_root / relative
        assert path.is_file() and path.stat().st_size == expected["bytes"]
        assert sha256_file(path) == expected["sha256"]
    return manifest


def analyze(
    *,
    panel_path: Path,
    panel_manifest_path: Path,
    extraction_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    assert not output_dir.exists()
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert_commit(experiment_commit)
    assert os.environ.get("RUN_ID") == ASSOCIATION_RUN_ID
    started = time.monotonic()
    panel = load_panel(panel_path, panel_manifest_path)
    extraction = validate_extraction(extraction_root)
    class_lookup = {name: index for index, name in enumerate(REFERENCE_CLASSES)}
    panel_classes = np.array(
        [class_lookup[value] for value in panel["reference_class"]], dtype=np.int64
    )
    assert np.bincount(panel_classes).tolist() == [2_048] * len(REFERENCE_CLASSES)
    output_dir.mkdir(parents=True)

    frames: list[pl.DataFrame] = []
    family_records: list[dict[str, Any]] = []
    for block_index in BLOCK_INDICES:
        arm = arm_label(block_index)
        forward = pl.read_parquet(extraction_root / arm / "sae_focal_forward.parquet")
        reverse = pl.read_parquet(
            extraction_root / arm / "sae_focal_reverse_complement.parquet"
        )
        for view in VIEWS:
            view_started = time.monotonic()
            associations = analyze_sparse_table(
                sparse_view(forward, reverse, view),
                panel_classes=panel_classes,
            ).with_columns(
                pl.lit(block_index + 1).cast(pl.UInt8).alias("block"),
                pl.lit(arm).alias("arm"),
                pl.lit(view).alias("view"),
                pl.lit(view in ORIENTATIONS).alias("primary_view"),
            )
            frames.append(associations)
            for reference_class in REFERENCE_CLASSES:
                family = associations.filter(
                    pl.col("reference_class") == reference_class
                )
                family_records.append(
                    {
                        "block": block_index + 1,
                        "arm": arm,
                        "view": view,
                        "primary_view": view in ORIENTATIONS,
                        "reference_class": reference_class,
                        "eligible_features": family.height,
                        "primary_associations": int(
                            family["primary_association"].sum()
                        ),
                        "maximum_abs_rank_biserial": float(
                            family["rank_biserial"].abs().max()
                        ),
                        "maximum_auprc": float(family["auprc"].max()),
                    }
                )
            print(
                json.dumps(
                    {
                        "stage": "associate_reference_state",
                        "arm": arm,
                        "view": view,
                        "eligible_features": associations["feature_id"].n_unique(),
                        "primary_associations": int(
                            associations["primary_association"].sum()
                        ),
                        "elapsed_seconds": time.monotonic() - view_started,
                    }
                ),
                flush=True,
            )

    all_associations = pl.concat(frames).select(
        "block",
        "arm",
        "view",
        "primary_view",
        "reference_class",
        "feature_id",
        "n_class",
        "n_rest",
        "nonzero_support_total",
        "nonzero_support_class",
        "mean_class",
        "mean_rest",
        "mean_difference",
        "cohen_d",
        "welch_t",
        "welch_df",
        "welch_p",
        "welch_q",
        "mwu_u",
        "mwu_p",
        "mwu_q",
        "rank_biserial",
        "auprc",
        "primary_association",
    )
    family_summary = pl.DataFrame(family_records).sort(
        "block", "view", "reference_class"
    )
    primary_hits = all_associations.filter(pl.col("primary_association")).sort(
        "block",
        "view",
        "reference_class",
        pl.col("rank_biserial").abs(),
        descending=[False, False, False, True],
    )
    paths = {
        "associations.parquet": all_associations,
        "family_summary.parquet": family_summary,
        "primary_hits.parquet": primary_hits,
    }
    artifacts: dict[str, Any] = {}
    for relative, frame in paths.items():
        path = output_dir / relative
        frame.write_parquet(path, compression="zstd")
        artifacts[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "rows": frame.height,
        }

    elapsed = time.monotonic() - started
    result: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "issue": ISSUE,
        "run_id": ASSOCIATION_RUN_ID,
        "analysis_status": "preregistered_reference_state_association_scan",
        "experiment_commit": experiment_commit,
        "elapsed_seconds": elapsed,
        "platform": platform.platform(),
        "inputs": {
            "panel_run_id": PANEL_RUN_ID,
            "panel_sha256": PANEL_SHA256,
            "extraction_run_id": EXTRACTION_RUN_ID,
            "extraction_experiment_commit": extraction["experiment_commit"],
            "extraction_manifest_sha256": sha256_file(
                extraction_root / "manifest.json"
            ),
        },
        "protocol": {
            "classes": list(REFERENCE_CLASSES),
            "class_prevalence": 1 / len(REFERENCE_CLASSES),
            "views": list(VIEWS),
            "primary_views": list(ORIENTATIONS),
            "minimum_nonzero_support": MINIMUM_SUPPORT,
            "welch_effect": "Cohen d using pooled sample variance",
            "mwu": "two-sided asymptotic normal approximation without continuity correction; all zero ties included",
            "rank_biserial": "2*U/(n_class*n_rest)-1; positive means higher in class",
            "auprc": "descriptive exact average precision including the tied zero threshold",
            "bh_family": "within block x view x class, separately for Welch and Mann-Whitney",
            "primary_call": (
                "Welch q<0.05 and Mann-Whitney q<0.05 and |rank-biserial|>=0.1"
            ),
        },
        "association_rows": all_associations.height,
        "primary_associations": primary_hits.height,
        "primary_view_associations": primary_hits.filter(pl.col("primary_view")).height,
        "family_summary": family_records,
        "artifacts": artifacts,
    }
    results_path = output_dir / "results.json"
    write_json(results_path, result)
    result["artifacts"]["results.json"] = {
        "bytes": results_path.stat().st_size,
        "sha256": sha256_file(results_path),
    }
    write_json(output_dir / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--extraction-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        panel_path=args.panel,
        panel_manifest_path=args.panel_manifest,
        extraction_root=args.extraction_root,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "association_rows": result["association_rows"],
                "primary_associations": result["primary_associations"],
                "primary_view_associations": result["primary_view_associations"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
