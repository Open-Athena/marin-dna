"""Run composition, overlap, boundary, and decoder sensitivities for issue 435."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import scipy
from safetensors.numpy import load_file as load_safetensors
from scipy import stats

from analyze_reference_capacity import load_sparse
from association_common import bh_adjust, comparison_metrics
from common import ISSUE, assert_commit, write_json
from extract_common import (
    BLOCK_INDICES,
    D_SAE,
    EXPECTED_SAE_ARTIFACTS,
    EXTRACTION_RUN_ID,
    ORIENTATIONS,
    TRAINING_TOKENS,
    arm_label,
    read_model_provenance,
    sha256_file,
)
from sensitivity_common import (
    CategorySubset,
    PairSubset,
    category_sensitivity_subsets,
    decoder_set_geometry,
    nearest_dictionary_neighbors,
    normalize_decoders,
    support_matched_controls,
    uniform_sensitivity_subsets,
)

RUN_ID = "dna-exp435-repeat-reference-sensitivities-r1"
ASSOCIATION_RUN_ID = "dna-exp435-repeat-reference-associations-r1"
EXTRACTION_ARCHIVE_SHA256 = (
    "7a02652172eb42efb5228a0e45a3a495a59fe58f20fdd60b787d01ac000649f6"
)
ASSOCIATION_ARCHIVE_SHA256 = (
    "cc72fbb0033290af54d2c6dcb0a7521e9b23f5f84b6906bfd9fee1f69206ece0"
)
HIERARCHIES = ("class", "family", "subfamily")
FDR_THRESHOLD = 0.05


def verify_archive(
    root: Path,
    *,
    expected_sha256: str,
    expected_run_id: str,
    expected_status: str,
) -> dict[str, Any]:
    manifest_path = root / "archive_manifest.json"
    assert manifest_path.is_file() and sha256_file(manifest_path) == expected_sha256
    manifest = json.loads(manifest_path.read_text())
    assert manifest["issue"] == ISSUE and manifest["run_id"] == expected_run_id
    assert manifest["analysis_status"] == expected_status
    for relative, expected in manifest["artifacts"].items():
        path = root / relative
        assert path.is_file() and path.stat().st_size == expected["bytes"]
        assert sha256_file(path) == expected["sha256"]
    return manifest


def with_fdr(frame: pl.DataFrame) -> pl.DataFrame:
    assert frame.height > 0
    return frame.with_columns(
        pl.Series("welch_q", bh_adjust(frame["welch_p"].to_numpy())),
        pl.Series("mann_whitney_q", bh_adjust(frame["mann_whitney_p"].to_numpy())),
    ).with_columns(
        pl.max_horizontal("welch_q", "mann_whitney_q").alias("maximum_q"),
        (
            (pl.col("mean_difference") > 0)
            & (pl.col("welch_q") <= FDR_THRESHOLD)
            & (pl.col("mann_whitney_q") <= FDR_THRESHOLD)
        ).alias("concordant_positive_association"),
    )


def broad_family(
    matrix: Any,
    subset: PairSubset,
    *,
    arm: str,
    block: int,
    orientation: str,
) -> pl.DataFrame:
    return with_fdr(
        comparison_metrics(
            matrix,
            subset.positive_ids,
            subset.negative_ids,
            minimum_nonzero_support=subset.minimum_nonzero_support,
        )
    ).with_columns(
        pl.lit(arm).alias("arm"),
        pl.lit(block, dtype=pl.UInt8).alias("block"),
        pl.lit(TRAINING_TOKENS, dtype=pl.UInt32).alias("budget"),
        pl.lit(orientation).alias("orientation"),
        pl.lit("repeat").alias("hierarchy"),
        pl.lit("repeat_vs_repeat_free").alias("target"),
        pl.lit(subset.name).alias("sensitivity"),
        pl.lit(subset.positive_ids.size, dtype=pl.UInt32).alias("retained_pairs"),
        pl.lit(subset.minimum_nonzero_support, dtype=pl.UInt32).alias(
            "minimum_nonzero_support"
        ),
    )


def category_family(
    matrix: Any,
    groups: list[CategorySubset],
    *,
    arm: str,
    block: int,
    orientation: str,
    sensitivity: str,
    hierarchy: str,
) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for group in groups:
        assert group.sensitivity == sensitivity and group.hierarchy == hierarchy
        ids = np.concatenate((group.positive_ids, group.negative_ids))
        support = np.asarray(matrix[ids, :].getnnz(axis=0)).ravel()
        if not np.any(support >= group.minimum_nonzero_support):
            continue
        frames.append(
            comparison_metrics(
                matrix,
                group.positive_ids,
                group.negative_ids,
                minimum_nonzero_support=group.minimum_nonzero_support,
            ).with_columns(
                pl.lit(group.target).alias("target"),
                pl.lit(group.positive_ids.size, dtype=pl.UInt16).alias(
                    "retained_pairs"
                ),
                pl.lit(group.minimum_nonzero_support, dtype=pl.UInt16).alias(
                    "minimum_nonzero_support"
                ),
            )
        )
    assert frames
    return with_fdr(pl.concat(frames, how="vertical")).with_columns(
        pl.lit(arm).alias("arm"),
        pl.lit(block, dtype=pl.UInt8).alias("block"),
        pl.lit(TRAINING_TOKENS, dtype=pl.UInt32).alias("budget"),
        pl.lit(orientation).alias("orientation"),
        pl.lit(hierarchy).alias("hierarchy"),
        pl.lit(sensitivity).alias("sensitivity"),
    )


def finite_correlation(x: np.ndarray, y: np.ndarray, *, method: str) -> float | None:
    assert x.shape == y.shape and x.ndim == 1
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    if method == "spearman":
        value = stats.spearmanr(x, y).statistic
    else:
        assert method == "pearson"
        value = stats.pearsonr(x, y).statistic
    return float(value) if np.isfinite(value) else None


def overlap_summary(
    current: pl.DataFrame,
    baseline: pl.DataFrame,
    *,
    key: list[str],
    sensitivity: str,
    hierarchy: str,
) -> dict[str, Any]:
    assert set(key) <= set(current.columns) and set(key) <= set(baseline.columns)
    current_keys = current.select(*key).unique()
    eligible_targets = set(current["target"].unique()) if "target" in key else set()
    relevant_baseline = (
        baseline.filter(pl.col("target").is_in(eligible_targets))
        if eligible_targets
        else baseline
    )
    baseline_positive = relevant_baseline.filter(
        pl.col("concordant_positive_association")
    )
    current_positive = current.filter(pl.col("concordant_positive_association"))
    baseline_testable = baseline_positive.join(current_keys, on=key, how="inner")
    retained = baseline_testable.join(
        current_positive.select(*key), on=key, how="inner"
    )
    joined = current.select(
        *key, pl.col("mean_difference").alias("current_effect")
    ).join(
        relevant_baseline.select(
            *key, pl.col("mean_difference").alias("baseline_effect")
        ),
        on=key,
        how="inner",
    )
    sign_retained = baseline_testable.join(
        current.select(*key, "mean_difference"), on=key, how="inner"
    ).filter(pl.col("mean_difference") > 0)
    baseline_set = set(map(tuple, baseline_positive.select(*key).iter_rows()))
    current_set = set(map(tuple, current_positive.select(*key).iter_rows()))
    union = baseline_set | current_set
    return {
        "sensitivity": sensitivity,
        "hierarchy": hierarchy,
        "supported_feature_target_pairs": current.height,
        "supported_features": current["feature_id"].n_unique(),
        "eligible_targets": current["target"].n_unique(),
        "current_positive_pairs": current_positive.height,
        "current_positive_features": current_positive["feature_id"].n_unique(),
        "current_positive_targets": current_positive["target"].n_unique(),
        "baseline_positive_pairs_relevant": baseline_positive.height,
        "baseline_positive_pairs_testable": baseline_testable.height,
        "baseline_positive_pairs_sign_retained": sign_retained.height,
        "baseline_positive_pairs_q_retained": retained.height,
        "sign_retention_fraction_testable": (
            sign_retained.height / baseline_testable.height
            if baseline_testable.height
            else None
        ),
        "q_retention_fraction_testable": (
            retained.height / baseline_testable.height
            if baseline_testable.height
            else None
        ),
        "positive_set_jaccard": len(baseline_set & current_set) / len(union)
        if union
        else None,
        "shared_effect_pearson": finite_correlation(
            joined["baseline_effect"].to_numpy(),
            joined["current_effect"].to_numpy(),
            method="pearson",
        ),
        "shared_effect_spearman": finite_correlation(
            joined["baseline_effect"].to_numpy(),
            joined["current_effect"].to_numpy(),
            method="spearman",
        ),
        "best_positive_auprc": (
            float(current_positive["auprc"].max()) if current_positive.height else None
        ),
    }


def baseline_family(
    association_root: Path, arm: str, orientation: str, hierarchy: str
) -> pl.DataFrame:
    path = (
        association_root
        / "associations"
        / "families"
        / arm
        / orientation
        / f"{hierarchy}.parquet"
    )
    frame = pl.read_parquet(path)
    assert frame["arm"].unique().to_list() == [arm]
    assert frame["orientation"].unique().to_list() == [orientation]
    assert frame["hierarchy"].unique().to_list() == [hierarchy]
    return frame


def feature_set_and_support(
    association_root: Path,
    arm: str,
    *,
    set_name: str,
) -> tuple[np.ndarray, dict[int, str]]:
    frames: list[pl.DataFrame] = []
    hierarchies = ("repeat",) if set_name == "broad_repeat" else HIERARCHIES
    for orientation in ORIENTATIONS:
        for hierarchy in hierarchies:
            frames.append(
                baseline_family(association_root, arm, orientation, hierarchy).filter(
                    pl.col("concordant_positive_association")
                )
            )
    combined = pl.concat(frames, how="diagonal_relaxed")
    assert combined.height > 0
    membership: dict[int, str] = {}
    for feature_id in combined["feature_id"].unique():
        orientations = set(
            combined.filter(pl.col("feature_id") == feature_id)["orientation"]
        )
        membership[int(feature_id)] = (
            "shared" if orientations == set(ORIENTATIONS) else next(iter(orientations))
        )
    ids = np.array(sorted(membership), dtype=np.int64)
    return ids, membership


def decoder_analysis(
    *,
    association_root: Path,
    models_root: Path,
    arm: str,
    block_index: int,
    panel_support: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    model_root = models_root / arm
    provenance = read_model_provenance(
        model_root,
        block_index=block_index,
        expected_artifacts=EXPECTED_SAE_ARTIFACTS[arm],
    )
    tensors = load_safetensors(model_root / "sae_weights.safetensors")
    assert set(tensors) == {"W_dec", "W_enc", "b_dec", "b_enc", "threshold"}
    decoder = tensors["W_dec"]
    assert decoder.shape == (D_SAE, 1_920) and decoder.dtype == np.float32
    normalized, decoder_norms = normalize_decoders(decoder)
    summary_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for set_name in ("broad_repeat", "category_union"):
        associated_ids, membership = feature_set_and_support(
            association_root, arm, set_name=set_name
        )
        assert np.all(panel_support[associated_ids] > 0)
        candidate_ids = np.flatnonzero(panel_support > 0)
        candidate_ids = candidate_ids[~np.isin(candidate_ids, associated_ids)]
        controls = support_matched_controls(
            associated_ids,
            panel_support[associated_ids],
            candidate_ids,
            panel_support[candidate_ids],
            namespace=f"{RUN_ID}|{arm}|{set_name}",
        )
        associated_geometry = decoder_set_geometry(normalized, associated_ids)
        control_geometry = decoder_set_geometry(normalized, controls)
        neighbors, similarities = nearest_dictionary_neighbors(
            normalized, associated_ids
        )
        associated_set = set(associated_ids)
        for feature_id, neighbor_id, similarity in zip(
            associated_ids, neighbors, similarities, strict=True
        ):
            feature_rows.append(
                {
                    "arm": arm,
                    "block": block_index + 1,
                    "set_name": set_name,
                    "feature_id": int(feature_id),
                    "orientation_membership": membership[int(feature_id)],
                    "panel_nonzero_support": int(panel_support[feature_id]),
                    "decoder_norm": float(decoder_norms[feature_id]),
                    "nearest_dictionary_feature_id": int(neighbor_id),
                    "nearest_dictionary_cosine": float(similarity),
                    "nearest_dictionary_feature_in_set": int(neighbor_id)
                    in associated_set,
                }
            )
        row: dict[str, Any] = {
            "arm": arm,
            "block": block_index + 1,
            "set_name": set_name,
            "matched_control_features": int(controls.size),
            "associated_median_panel_support": float(
                np.median(panel_support[associated_ids])
            ),
            "control_median_panel_support": float(np.median(panel_support[controls])),
            "nearest_dictionary_neighbor_in_set_fraction": float(
                np.mean(np.isin(neighbors, associated_ids))
            ),
        }
        row.update(
            {f"associated_{key}": value for key, value in associated_geometry.items()}
        )
        row.update({f"control_{key}": value for key, value in control_geometry.items()})
        summary_rows.append(row)
    del tensors, decoder, normalized
    return summary_rows, feature_rows, provenance


def artifact_record(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def analyze(
    *,
    extraction_archive: Path,
    association_archive: Path,
    models_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    assert not output_dir.exists() and models_root.is_dir()
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert_commit(experiment_commit)
    assert os.environ.get("RUN_ID") == RUN_ID
    started = time.monotonic()
    extraction_manifest = verify_archive(
        extraction_archive,
        expected_sha256=EXTRACTION_ARCHIVE_SHA256,
        expected_run_id=EXTRACTION_RUN_ID,
        expected_status="frozen_reference_sae_extraction",
    )
    association_manifest = verify_archive(
        association_archive,
        expected_sha256=ASSOCIATION_ARCHIVE_SHA256,
        expected_run_id=ASSOCIATION_RUN_ID,
        expected_status="frozen_reference_repeat_capacity_associations",
    )
    panel_root = extraction_archive / "inputs" / "panel" / "panel"
    contexts = pl.read_parquet(panel_root / "contexts.parquet")
    uniform_pairs = pl.read_parquet(panel_root / "uniform_pairs.parquet").sort(
        "pair_id"
    )
    comparisons = pl.read_parquet(panel_root / "category_comparisons.parquet").sort(
        "level", "label", "pair_id"
    )
    uniform_subsets = uniform_sensitivity_subsets(contexts, uniform_pairs)
    category_subsets = category_sensitivity_subsets(contexts, comparisons)

    output_dir.mkdir(parents=True)
    artifacts: dict[str, Any] = {}
    sensitivity_rows: list[dict[str, Any]] = []
    decoder_summary_rows: list[dict[str, Any]] = []
    decoder_feature_rows: list[dict[str, Any]] = []
    sae_provenance: dict[str, Any] = {}
    extraction_root = extraction_archive / "extraction"
    for block_index in BLOCK_INDICES:
        block = block_index + 1
        arm = arm_label(block_index)
        orientation_support: list[np.ndarray] = []
        for orientation in ORIENTATIONS:
            print(
                json.dumps(
                    {"stage": "load_sparse", "arm": arm, "orientation": orientation}
                ),
                flush=True,
            )
            matrix = load_sparse(
                extraction_root / arm / f"sae_focal_{orientation}.parquet"
            )
            orientation_support.append(
                np.bincount(matrix.indices, minlength=D_SAE).astype(np.int64)
            )
            broad_baseline = baseline_family(
                association_archive, arm, orientation, "repeat"
            )
            for sensitivity, subset in uniform_subsets.items():
                family = broad_family(
                    matrix,
                    subset,
                    arm=arm,
                    block=block,
                    orientation=orientation,
                )
                relative = (
                    Path("families")
                    / arm
                    / orientation
                    / "repeat"
                    / f"{sensitivity}.parquet"
                )
                path = output_dir / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                family.write_parquet(path, compression="zstd")
                artifacts[str(relative)] = artifact_record(path)
                summary = overlap_summary(
                    family,
                    broad_baseline,
                    key=["feature_id"],
                    sensitivity=sensitivity,
                    hierarchy="repeat",
                )
                summary.update({"arm": arm, "block": block, "orientation": orientation})
                sensitivity_rows.append(summary)
                print(json.dumps({"stage": "broad_complete", **summary}), flush=True)

            for sensitivity, hierarchy_groups in category_subsets.items():
                for hierarchy in HIERARCHIES:
                    family = category_family(
                        matrix,
                        hierarchy_groups[hierarchy],
                        arm=arm,
                        block=block,
                        orientation=orientation,
                        sensitivity=sensitivity,
                        hierarchy=hierarchy,
                    )
                    relative = (
                        Path("families")
                        / arm
                        / orientation
                        / hierarchy
                        / f"{sensitivity}.parquet"
                    )
                    path = output_dir / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    family.write_parquet(path, compression="zstd")
                    artifacts[str(relative)] = artifact_record(path)
                    summary = overlap_summary(
                        family,
                        baseline_family(
                            association_archive, arm, orientation, hierarchy
                        ),
                        key=["target", "feature_id"],
                        sensitivity=sensitivity,
                        hierarchy=hierarchy,
                    )
                    summary.update(
                        {"arm": arm, "block": block, "orientation": orientation}
                    )
                    sensitivity_rows.append(summary)
                    print(
                        json.dumps({"stage": "category_complete", **summary}),
                        flush=True,
                    )
            del matrix

        panel_support = np.maximum.reduce(orientation_support)
        decoder_summary, decoder_features, provenance = decoder_analysis(
            association_root=association_archive,
            models_root=models_root,
            arm=arm,
            block_index=block_index,
            panel_support=panel_support,
        )
        decoder_summary_rows.extend(decoder_summary)
        decoder_feature_rows.extend(decoder_features)
        sae_provenance[arm] = provenance

    sensitivity_summary = pl.DataFrame(sensitivity_rows).sort(
        "block", "orientation", "hierarchy", "sensitivity"
    )
    sensitivity_path = output_dir / "sensitivity_summary.parquet"
    sensitivity_summary.write_parquet(sensitivity_path, compression="zstd")
    artifacts[sensitivity_path.name] = artifact_record(sensitivity_path)
    decoder_summary = pl.DataFrame(decoder_summary_rows).sort("block", "set_name")
    decoder_summary_path = output_dir / "decoder_summary.parquet"
    decoder_summary.write_parquet(decoder_summary_path, compression="zstd")
    artifacts[decoder_summary_path.name] = artifact_record(decoder_summary_path)
    decoder_features = pl.DataFrame(decoder_feature_rows).sort(
        "block", "set_name", "feature_id"
    )
    decoder_features_path = output_dir / "decoder_features.parquet"
    decoder_features.write_parquet(decoder_features_path, compression="zstd")
    artifacts[decoder_features_path.name] = artifact_record(decoder_features_path)

    result: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "issue": ISSUE,
        "run_id": RUN_ID,
        "analysis_status": "frozen_reference_repeat_sensitivities",
        "experiment_commit": experiment_commit,
        "elapsed_seconds": time.monotonic() - started,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "polars": pl.__version__,
        "scipy": scipy.__version__,
        "input": {
            "extraction_run_id": EXTRACTION_RUN_ID,
            "extraction_archive_manifest_sha256": EXTRACTION_ARCHIVE_SHA256,
            "extraction_objects_excluding_manifest": extraction_manifest[
                "object_count_excluding_this_manifest"
            ],
            "association_run_id": ASSOCIATION_RUN_ID,
            "association_archive_manifest_sha256": ASSOCIATION_ARCHIVE_SHA256,
            "association_objects_excluding_manifest": association_manifest[
                "object_count_excluding_this_manifest"
            ],
        },
        "protocol": {
            "uniform_subsets": {
                name: {
                    "pairs": subset.positive_ids.size,
                    "minimum_nonzero_support": subset.minimum_nonzero_support,
                }
                for name, subset in uniform_subsets.items()
            },
            "category_subsets": {
                sensitivity: {
                    hierarchy: len(groups)
                    for hierarchy, groups in hierarchy_groups.items()
                }
                for sensitivity, hierarchy_groups in category_subsets.items()
            },
            "category_minimum_pairs": 32,
            "tests": ["Welch t", "Mann-Whitney U"],
            "bh_family": (
                "within layer x orientation x hierarchy x sensitivity x statistic"
            ),
            "association_call": "positive mean and both BH q <= 0.05",
            "decoder_similarity": "signed cosine of row-normalized W_dec",
            "decoder_sets": ["broad_repeat", "category_union"],
            "decoder_controls": (
                "deterministic nonassociated features matched greedily on log1p panel support"
            ),
            "post_result_sensitivity_not_independent_confirmation": True,
        },
        "saes": sae_provenance,
        "sensitivity_summary": sensitivity_summary.to_dicts(),
        "decoder_summary": decoder_summary.to_dicts(),
        "artifacts": artifacts,
    }
    results_path = output_dir / "results.json"
    write_json(results_path, result)
    result["artifacts"]["results.json"] = artifact_record(results_path)
    write_json(output_dir / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction-archive", type=Path, required=True)
    parser.add_argument("--association-archive", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        extraction_archive=args.extraction_archive,
        association_archive=args.association_archive,
        models_root=args.models_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(result["decoder_summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
