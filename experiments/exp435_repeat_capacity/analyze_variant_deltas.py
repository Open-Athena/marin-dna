"""Test whether reference-repeat SAE features respond to variants in repeats."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from common import ISSUE, assert_commit, sha256_file, write_json
from extract_common import D_SAE
from variant_analysis_common import (
    ARMS,
    BLOCK_BY_ARM,
    HIERARCHIES,
    MINIMUM_GLOBAL_SUPPORT,
    MINIMUM_STRATIFIED_SUPPORT,
    ORIENTATIONS,
    PAIRED_ACTIVATION_MANIFEST_SHA256,
    PAIRED_RUN_ID,
    REFERENCE_ASSOCIATION_ARCHIVE_SHA256,
    RESPONSES,
    SUBSET_TARGETS,
    VARIANT_PANEL_ARCHIVE_SHA256,
    binary_feature_metrics,
    with_fdr,
)
from variant_common import (
    EXPECTED_VARIANTS,
    MIN_CATEGORY_VARIANTS,
    VARIANT_PANEL_RUN_ID,
)

ASSOCIATION_RUN_ID = "dna-exp435-repeat-reference-associations-r1"
ACTIVATION_RUN_ID = "dna-exp436-mendelian-focal-seed288-r1"
REFERENCE_CALL_COLUMN = "concordant_positive_association"
VARIANT_SENSITIVITIES = ("all", "unique_overlap", "interior_32")


def verify_archive(
    root: Path, expected_sha256: str, expected_status: str
) -> dict[str, Any]:
    manifest_path = root / "archive_manifest.json"
    assert manifest_path.is_file() and sha256_file(manifest_path) == expected_sha256
    manifest = json.loads(manifest_path.read_text())
    assert manifest["analysis_status"] == expected_status
    for relative, metadata in manifest["artifacts"].items():
        path = root / relative
        assert path.is_file() and path.stat().st_size == metadata["bytes"]
        assert sha256_file(path) == metadata["sha256"]
    return manifest


def verify_activations(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    assert manifest_path.is_file()
    assert sha256_file(manifest_path) == PAIRED_ACTIVATION_MANIFEST_SHA256
    manifest = json.loads(manifest_path.read_text())
    assert manifest["run_id"] == ACTIVATION_RUN_ID
    for arm in ARMS:
        for orientation in ORIENTATIONS:
            relative = f"{arm}/sae_focal_{orientation}.parquet"
            metadata = manifest["artifacts"][relative]
            path = root / relative
            assert path.is_file() and path.stat().st_size == metadata["bytes"]
            assert sha256_file(path) == metadata["sha256"]
    return manifest


def load_reference_features(
    root: Path,
) -> tuple[
    dict[tuple[str, str], np.ndarray],
    dict[tuple[str, str, str, str], np.ndarray],
]:
    broad: dict[tuple[str, str], np.ndarray] = {}
    categories: dict[tuple[str, str, str, str], np.ndarray] = {}
    for arm in ARMS:
        for orientation in ORIENTATIONS:
            family_root = root / "associations" / "families" / arm / orientation
            repeat = pl.read_parquet(family_root / "repeat.parquet")
            selected = (
                repeat.filter(pl.col(REFERENCE_CALL_COLUMN))["feature_id"]
                .unique()
                .sort()
                .to_numpy()
                .astype(np.int64)
            )
            assert selected.size > 0
            broad[(arm, orientation)] = selected
            for hierarchy in HIERARCHIES:
                family = pl.read_parquet(family_root / f"{hierarchy}.parquet").filter(
                    pl.col(REFERENCE_CALL_COLUMN)
                )
                for target in family["target"].unique(maintain_order=True):
                    features = (
                        family.filter(pl.col("target") == target)["feature_id"]
                        .unique()
                        .sort()
                        .to_numpy()
                        .astype(np.int64)
                    )
                    assert features.size > 0
                    categories[(arm, orientation, hierarchy, str(target))] = features
    return broad, categories


def load_activation_table(path: Path) -> pl.DataFrame:
    table = pl.read_parquet(
        path,
        columns=[
            "panel_row",
            "feature_id",
            "ref_activation",
            "alt_activation",
            "delta",
        ],
    )
    assert table.height > 0
    assert (
        table.select(pl.struct("panel_row", "feature_id").n_unique()).item()
        == table.height
    )
    assert table.filter(
        (pl.col("panel_row") >= EXPECTED_VARIANTS) | (pl.col("feature_id") >= D_SAE)
    ).is_empty()
    assert table.filter(
        ~pl.col("ref_activation").is_finite()
        | ~pl.col("alt_activation").is_finite()
        | ~pl.col("delta").is_finite()
        | (pl.col("ref_activation") < 0)
        | (pl.col("alt_activation") < 0)
    ).is_empty()
    assert table.filter(
        (pl.col("delta") - (pl.col("alt_activation") - pl.col("ref_activation"))).abs()
        > 1e-5
    ).is_empty()
    return table


def targeted_delta_matrix(
    table: pl.DataFrame, feature_ids: np.ndarray
) -> tuple[np.ndarray, dict[int, int]]:
    features = np.asarray(sorted(set(feature_ids.tolist())), dtype=np.int64)
    assert features.size > 0 and features.min() >= 0 and features.max() < D_SAE
    feature_to_column = {int(feature): index for index, feature in enumerate(features)}
    selected = table.filter(pl.col("feature_id").is_in(features))
    matrix = np.zeros((EXPECTED_VARIANTS, features.size), dtype=np.float32)
    if selected.height:
        rows = selected["panel_row"].to_numpy().astype(np.int64)
        columns = np.fromiter(
            (feature_to_column[int(feature)] for feature in selected["feature_id"]),
            dtype=np.int64,
            count=selected.height,
        )
        matrix[rows, columns] = selected["delta"].to_numpy().astype(np.float32)
    assert np.isfinite(matrix).all()
    return matrix, feature_to_column


def selected_matrix(
    matrix: np.ndarray,
    feature_to_column: dict[int, int],
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    retained = np.asarray(
        sorted(
            feature
            for feature in features.tolist()
            if int(feature) in feature_to_column
        ),
        dtype=np.int64,
    )
    assert retained.size == len(set(features.tolist()))
    columns = np.asarray(
        [feature_to_column[int(feature)] for feature in retained], dtype=np.int64
    )
    return matrix[:, columns], retained


def add_metadata(
    frame: pl.DataFrame,
    *,
    arm: str,
    orientation: str,
    response: str,
    scope: str,
    hierarchy: str,
    target: str,
    variant_sensitivity: str,
    minimum_support: int,
) -> pl.DataFrame:
    assert frame.height > 0
    return frame.with_columns(
        pl.lit(arm).alias("arm"),
        pl.lit(BLOCK_BY_ARM[arm]).cast(pl.UInt8).alias("block"),
        pl.lit(orientation).alias("orientation"),
        pl.lit(response).alias("response"),
        pl.lit(scope).alias("scope"),
        pl.lit(hierarchy).alias("hierarchy"),
        pl.lit(target).alias("target"),
        pl.lit(variant_sensitivity).alias("variant_sensitivity"),
        pl.lit(minimum_support).cast(pl.UInt32).alias("minimum_nonzero_support"),
    ).select(
        "arm",
        "block",
        "orientation",
        "response",
        "scope",
        "hierarchy",
        "target",
        "variant_sensitivity",
        "minimum_nonzero_support",
        pl.exclude(
            "arm",
            "block",
            "orientation",
            "response",
            "scope",
            "hierarchy",
            "target",
            "variant_sensitivity",
            "minimum_nonzero_support",
        ),
    )


def run_contrast(
    matrix: np.ndarray,
    feature_to_column: dict[int, int],
    features: np.ndarray,
    positive_rows: np.ndarray,
    negative_rows: np.ndarray,
    *,
    response: str,
    minimum_support: int,
    metadata: dict[str, str],
) -> pl.DataFrame | None:
    selected, retained = selected_matrix(matrix, feature_to_column, features)
    response_matrix = np.abs(selected) if response == "abs_delta" else selected
    frame = binary_feature_metrics(
        response_matrix,
        retained,
        positive_rows,
        negative_rows,
        minimum_nonzero_support=minimum_support,
    )
    if frame.is_empty():
        return None
    return add_metadata(
        frame,
        response=response,
        minimum_support=minimum_support,
        **metadata,
    )


def sensitivity_mask(panel: pl.DataFrame, name: str) -> np.ndarray:
    focal = panel["position_status"].to_numpy() == "focal_repeat"
    if name == "all":
        return focal
    if name == "unique_overlap":
        return focal & panel["unique_repeat_overlap"].to_numpy()
    assert name == "interior_32"
    return focal & panel["repeat_interior_32"].to_numpy()


def capacity_rows(
    table: pl.DataFrame,
    panel: pl.DataFrame,
    broad_features: np.ndarray,
    *,
    arm: str,
    orientation: str,
) -> list[dict[str, Any]]:
    panel_rows = table["panel_row"].to_numpy().astype(np.int64)
    feature_ids = table["feature_id"].to_numpy().astype(np.int64)
    ref = table["ref_activation"].to_numpy().astype(np.float64)
    alt = table["alt_activation"].to_numpy().astype(np.float64)
    delta = table["delta"].to_numpy().astype(np.float64)
    absolute = np.abs(delta)
    nonzero = delta != 0
    status_by_row = panel["position_status"].to_numpy()
    associated = np.isin(feature_ids, broad_features)
    rows: list[dict[str, Any]] = []
    for status in ("focal_repeat", "near_repeat", "repeat_free_window"):
        in_group = status_by_row[panel_rows] == status
        nonzero_group = in_group & nonzero
        associated_nonzero = nonzero_group & associated
        total_slots = int(nonzero_group.sum())
        total_mass = float(absolute[in_group].sum())
        associated_slots = int(associated_nonzero.sum())
        associated_mass = float(absolute[in_group & associated].sum())
        variants = int((status_by_row == status).sum())
        rows.append(
            {
                "arm": arm,
                "block": BLOCK_BY_ARM[arm],
                "orientation": orientation,
                "position_status": status,
                "variants": variants,
                "reference_positive_repeat_features": int(broad_features.size),
                "total_nonzero_delta_slots": total_slots,
                "repeat_feature_nonzero_delta_slots": associated_slots,
                "repeat_feature_fraction_nonzero_delta_slots": (
                    associated_slots / total_slots if total_slots else None
                ),
                "total_abs_delta_mass": total_mass,
                "repeat_feature_abs_delta_mass": associated_mass,
                "repeat_feature_fraction_abs_delta_mass": (
                    associated_mass / total_mass if total_mass else None
                ),
                "mean_abs_delta_mass_per_variant": total_mass / variants,
                "repeat_feature_mean_abs_delta_mass_per_variant": associated_mass
                / variants,
                "inactive_to_active_slots": int(
                    (in_group & (ref == 0) & (alt > 0)).sum()
                ),
                "active_to_inactive_slots": int(
                    (in_group & (ref > 0) & (alt == 0)).sum()
                ),
                "active_to_active_changed_slots": int(
                    (in_group & (ref > 0) & (alt > 0) & nonzero).sum()
                ),
            }
        )
    return rows


def artifact_record(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def analyze(
    panel_archive: Path,
    association_archive: Path,
    activation_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    assert not output_dir.exists()
    started = time.perf_counter()
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert_commit(experiment_commit)
    assert os.environ.get("RUN_ID") == PAIRED_RUN_ID
    panel_manifest = verify_archive(
        panel_archive,
        VARIANT_PANEL_ARCHIVE_SHA256,
        "outcome_blind_paired_repeat_variant_panel",
    )
    association_manifest = verify_archive(
        association_archive,
        REFERENCE_ASSOCIATION_ARCHIVE_SHA256,
        "frozen_reference_repeat_capacity_associations",
    )
    activation_manifest = verify_activations(activation_root)

    panel = pl.read_parquet(panel_archive / "panel" / "variant_panel.parquet").sort(
        "panel_row"
    )
    categories = pl.read_parquet(panel_archive / "panel" / "category_counts.parquet")
    assert panel.height == EXPECTED_VARIANTS and "label" not in panel.columns
    assert panel["panel_row"].to_list() == list(range(EXPECTED_VARIANTS))
    broad_features, category_features = load_reference_features(association_archive)

    output_dir.mkdir(parents=True)
    family_frames: dict[tuple[str, str, str, str], list[pl.DataFrame]] = defaultdict(
        list
    )
    capacity: list[dict[str, Any]] = []
    status = panel["position_status"].to_numpy()
    repeat_free_rows = np.flatnonzero(status == "repeat_free_window")
    subset_values = panel["subset"].to_numpy()
    category_columns = {
        "class": "repeat_class",
        "family": "family_label",
        "subfamily": "subfamily_label",
    }

    for arm in ARMS:
        for orientation in ORIENTATIONS:
            broad = broad_features[(arm, orientation)]
            eligible_category_keys = [
                key
                for key in category_features
                if key[0] == arm
                and key[1] == orientation
                and categories.filter(
                    (pl.col("level") == key[2])
                    & (pl.col("category") == key[3])
                    & (pl.col("variants") >= MIN_CATEGORY_VARIANTS)
                ).height
                == 1
            ]
            union = np.asarray(
                sorted(
                    set(broad.tolist()).union(
                        *(
                            set(category_features[key].tolist())
                            for key in eligible_category_keys
                        )
                    )
                ),
                dtype=np.int64,
            )
            table = load_activation_table(
                activation_root / arm / f"sae_focal_{orientation}.parquet"
            )
            matrix, feature_to_column = targeted_delta_matrix(table, union)
            capacity.extend(
                capacity_rows(
                    table,
                    panel,
                    broad,
                    arm=arm,
                    orientation=orientation,
                )
            )
            for response in RESPONSES:
                for variant_sensitivity in VARIANT_SENSITIVITIES:
                    positive_rows = np.flatnonzero(
                        sensitivity_mask(panel, variant_sensitivity)
                    )
                    frame = run_contrast(
                        matrix,
                        feature_to_column,
                        broad,
                        positive_rows,
                        repeat_free_rows,
                        response=response,
                        minimum_support=MINIMUM_GLOBAL_SUPPORT,
                        metadata={
                            "arm": arm,
                            "orientation": orientation,
                            "scope": "broad",
                            "hierarchy": "repeat",
                            "target": "repeat_vs_repeat_free",
                            "variant_sensitivity": variant_sensitivity,
                        },
                    )
                    if frame is not None:
                        family_frames[
                            (arm, "broad", variant_sensitivity, response)
                        ].append(frame)

                focal = sensitivity_mask(panel, "all")
                for target in SUBSET_TARGETS:
                    in_subset = subset_values == target
                    positive_rows = np.flatnonzero(focal & in_subset)
                    negative_rows = np.flatnonzero(
                        (status == "repeat_free_window") & in_subset
                    )
                    assert positive_rows.size >= MIN_CATEGORY_VARIANTS
                    assert negative_rows.size >= MIN_CATEGORY_VARIANTS
                    frame = run_contrast(
                        matrix,
                        feature_to_column,
                        broad,
                        positive_rows,
                        negative_rows,
                        response=response,
                        minimum_support=MINIMUM_STRATIFIED_SUPPORT,
                        metadata={
                            "arm": arm,
                            "orientation": orientation,
                            "scope": "subset",
                            "hierarchy": "repeat",
                            "target": target,
                            "variant_sensitivity": "all",
                        },
                    )
                    if frame is not None:
                        family_frames[(arm, "subset", "all", response)].append(frame)

                for hierarchy in HIERARCHIES:
                    category_values = panel[category_columns[hierarchy]].to_numpy()
                    targets = categories.filter(
                        (pl.col("level") == hierarchy)
                        & (pl.col("variants") >= MIN_CATEGORY_VARIANTS)
                    )["category"].to_list()
                    for variant_sensitivity in VARIANT_SENSITIVITIES:
                        universe = sensitivity_mask(panel, variant_sensitivity)
                        for target in targets:
                            key = (arm, orientation, hierarchy, str(target))
                            features = category_features.get(key)
                            if features is None:
                                continue
                            positive_rows = np.flatnonzero(
                                universe & (category_values == target)
                            )
                            negative_rows = np.flatnonzero(
                                universe & (category_values != target)
                            )
                            if positive_rows.size < 2 or negative_rows.size < 2:
                                continue
                            frame = run_contrast(
                                matrix,
                                feature_to_column,
                                features,
                                positive_rows,
                                negative_rows,
                                response=response,
                                minimum_support=MINIMUM_STRATIFIED_SUPPORT,
                                metadata={
                                    "arm": arm,
                                    "orientation": orientation,
                                    "scope": "category",
                                    "hierarchy": hierarchy,
                                    "target": str(target),
                                    "variant_sensitivity": variant_sensitivity,
                                },
                            )
                            if frame is not None:
                                family_frames[
                                    (
                                        arm,
                                        f"category_{hierarchy}",
                                        variant_sensitivity,
                                        response,
                                    )
                                ].append(frame)
            del table, matrix

    artifacts: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    top_frames: list[pl.DataFrame] = []
    family_templates: list[pl.DataFrame] = []
    for (arm, family_name, variant_sensitivity, response), frames in sorted(
        family_frames.items()
    ):
        family = with_fdr(pl.concat(frames), response=response).sort(
            "maximum_q",
            "best_auprc",
            "orientation",
            "target",
            "feature_id",
            descending=[False, True, False, False, False],
        )
        family_templates.append(family.head(0))
        path = (
            output_dir
            / "families"
            / arm
            / family_name
            / variant_sensitivity
            / f"{response}.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        family.write_parquet(path, compression="zstd")
        relative = str(path.relative_to(output_dir))
        artifacts[relative] = artifact_record(path)
        positive = family.filter(pl.col("positive_mutation_association"))
        concordant = family.filter(pl.col("concordant_association"))
        summaries[relative] = {
            "tested_pairs": family.height,
            "targets": family["target"].n_unique(),
            "features": family["feature_id"].n_unique(),
            "concordant_associations": concordant.height,
            "positive_mutation_associations": positive.height,
            "best_auprc": float(family["best_auprc"].max()),
            "minimum_maximum_q": float(family["maximum_q"].min()),
        }
        if concordant.height:
            top_frames.append(concordant.head(100))

    capacity_frame = pl.DataFrame(capacity).sort(
        "block", "orientation", "position_status"
    )
    capacity_path = output_dir / "capacity_summary.parquet"
    capacity_frame.write_parquet(capacity_path, compression="zstd")
    artifacts[capacity_path.name] = artifact_record(capacity_path)
    top_hits = pl.concat(top_frames if top_frames else family_templates[:1]).sort(
        "maximum_q", "best_auprc", descending=[False, True]
    )
    top_path = output_dir / "top_hits.parquet"
    top_hits.write_parquet(top_path, compression="zstd")
    artifacts[top_path.name] = artifact_record(top_path)

    result = {
        "issue": ISSUE,
        "run_id": PAIRED_RUN_ID,
        "analysis_status": "frozen_paired_repeat_variant_delta_associations",
        "created_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "experiment_commit": experiment_commit,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "polars": pl.__version__,
        "inputs": {
            "variant_panel": {
                "run_id": VARIANT_PANEL_RUN_ID,
                "archive_manifest_sha256": VARIANT_PANEL_ARCHIVE_SHA256,
                "artifacts": len(panel_manifest["artifacts"]),
            },
            "reference_associations": {
                "run_id": ASSOCIATION_RUN_ID,
                "archive_manifest_sha256": REFERENCE_ASSOCIATION_ARCHIVE_SHA256,
                "artifacts": len(association_manifest["artifacts"]),
            },
            "paired_activations": {
                "run_id": ACTIVATION_RUN_ID,
                "manifest_sha256": PAIRED_ACTIVATION_MANIFEST_SHA256,
                "experiment_commit": activation_manifest["experiment_commit"],
            },
        },
        "protocol": {
            "layers": [1, 10, 19],
            "checkpoint": "25M",
            "orientations": list(ORIENTATIONS),
            "responses": list(RESPONSES),
            "global_support": MINIMUM_GLOBAL_SUPPORT,
            "stratified_support": MINIMUM_STRATIFIED_SUPPORT,
            "fdr": (
                "BH within layer/family/variant-sensitivity/response/statistic, "
                "joint across orientations and all target-feature pairs"
            ),
            "label_used": False,
            "heldout_split_used": False,
        },
        "summaries": summaries,
        "artifacts": artifacts,
    }
    write_json(output_dir / "results.json", result)
    artifacts["results.json"] = artifact_record(output_dir / "results.json")
    write_json(output_dir / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-archive", type=Path, required=True)
    parser.add_argument("--association-archive", type=Path, required=True)
    parser.add_argument("--activation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        args.panel_archive,
        args.association_archive,
        args.activation_root,
        args.output_dir,
    )
    print(json.dumps(result["summaries"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
