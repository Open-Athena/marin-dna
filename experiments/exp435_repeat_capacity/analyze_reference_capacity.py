"""Run the frozen repeat-capacity association families on sparse activations."""

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
import pyarrow.parquet as pq
import scipy
from scipy.sparse import csr_matrix

from association_common import bh_adjust, comparison_metrics
from common import ISSUE, assert_commit, write_json
from extract_common import (
    CONTEXTS,
    D_SAE,
    EXTRACTION_RUN_ID,
    ORIENTATIONS,
    PANEL_ARCHIVE_MANIFEST_SHA256,
    PANEL_RUN_ID,
    TRAINING_TOKENS,
    arm_label,
    sha256_file,
)

RUN_ID = "dna-exp435-repeat-reference-associations-r1"
BLOCKS = (1, 10, 19)
HIERARCHIES = ("repeat", "class", "family", "subfamily")
MINIMUM_SUPPORT = {"repeat": 64, "class": 16, "family": 16, "subfamily": 16}
FDR_THRESHOLD = 0.05


def verify_archive(root: Path, expected_sha256: str) -> dict[str, Any]:
    manifest_path = root / "archive_manifest.json"
    assert manifest_path.is_file() and len(expected_sha256) == 64
    assert sha256_file(manifest_path) == expected_sha256
    manifest = json.loads(manifest_path.read_text())
    assert manifest["issue"] == ISSUE and manifest["run_id"] == EXTRACTION_RUN_ID
    assert manifest["analysis_status"] == "frozen_reference_sae_extraction"
    for relative, expected in manifest["artifacts"].items():
        path = root / relative
        assert path.is_file() and path.stat().st_size == expected["bytes"]
        assert sha256_file(path) == expected["sha256"]
    return manifest


def verify_extraction(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["issue"] == ISSUE and manifest["run_id"] == EXTRACTION_RUN_ID
    assert manifest["analysis_status"] == "frozen_reference_sae_extraction"
    assert manifest["panel"]["run_id"] == PANEL_RUN_ID
    assert manifest["panel"]["archive_manifest_sha256"] == PANEL_ARCHIVE_MANIFEST_SHA256
    assert manifest["panel"]["contexts"] == CONTEXTS
    for relative, expected in manifest["artifacts"].items():
        path = root / relative
        assert path.is_file() and path.stat().st_size == expected["bytes"]
        assert sha256_file(path) == expected["sha256"]
    return manifest


def load_sparse(path: Path) -> csr_matrix:
    table = pq.read_table(
        path, columns=["context_id", "feature_id", "activation"], memory_map=True
    )
    context_ids = table["context_id"].to_numpy(zero_copy_only=False)
    feature_ids = table["feature_id"].to_numpy(zero_copy_only=False)
    activations = table["activation"].to_numpy(zero_copy_only=False)
    assert context_ids.dtype == feature_ids.dtype == np.uint32
    assert activations.dtype == np.float32
    assert context_ids.size == feature_ids.size == activations.size > 0
    assert int(context_ids.max()) < CONTEXTS and int(feature_ids.max()) < D_SAE
    assert np.isfinite(activations).all() and np.all(activations > 0)
    ordered = (context_ids[1:] > context_ids[:-1]) | (
        (context_ids[1:] == context_ids[:-1]) & (feature_ids[1:] > feature_ids[:-1])
    )
    assert ordered.all()
    matrix = csr_matrix(
        (activations, (context_ids, feature_ids)), shape=(CONTEXTS, D_SAE)
    )
    assert matrix.nnz == activations.size and matrix.has_sorted_indices
    return matrix


def add_family_metadata(
    frame: pl.DataFrame,
    *,
    arm: str,
    block: int,
    orientation: str,
    hierarchy: str,
    target: str,
) -> pl.DataFrame:
    assert hierarchy in HIERARCHIES and orientation in ORIENTATIONS
    return frame.with_columns(
        pl.lit(arm).alias("arm"),
        pl.lit(block, dtype=pl.UInt8).alias("block"),
        pl.lit(TRAINING_TOKENS, dtype=pl.UInt32).alias("budget"),
        pl.lit(orientation).alias("orientation"),
        pl.lit("focal").alias("pooling"),
        pl.lit(hierarchy).alias("hierarchy"),
        pl.lit(target).alias("target"),
        pl.lit(MINIMUM_SUPPORT[hierarchy], dtype=pl.UInt32).alias(
            "minimum_nonzero_support"
        ),
    ).select(
        "arm",
        "block",
        "budget",
        "orientation",
        "pooling",
        "hierarchy",
        "target",
        "minimum_nonzero_support",
        pl.all().exclude(
            "arm",
            "block",
            "budget",
            "orientation",
            "pooling",
            "hierarchy",
            "target",
            "minimum_nonzero_support",
        ),
    )


def correct_family(frames: list[pl.DataFrame]) -> pl.DataFrame:
    assert frames
    family = pl.concat(frames, how="vertical")
    welch_q = bh_adjust(family["welch_p"].to_numpy())
    mann_q = bh_adjust(family["mann_whitney_p"].to_numpy())
    return family.with_columns(
        pl.Series("welch_q", welch_q),
        pl.Series("mann_whitney_q", mann_q),
    ).with_columns(
        pl.max_horizontal("welch_q", "mann_whitney_q").alias("maximum_q"),
        (
            (pl.col("mean_difference") > 0)
            & (pl.col("welch_q") <= FDR_THRESHOLD)
            & (pl.col("mann_whitney_q") <= FDR_THRESHOLD)
        ).alias("concordant_positive_association"),
    )


def family_summary(frame: pl.DataFrame) -> dict[str, Any]:
    discoveries = frame.filter(pl.col("concordant_positive_association"))
    return {
        "feature_target_pairs": frame.height,
        "targets": frame["target"].n_unique(),
        "eligible_features": frame["feature_id"].n_unique(),
        "welch_discoveries_q05": frame.filter(
            pl.col("welch_q") <= FDR_THRESHOLD
        ).height,
        "mann_whitney_discoveries_q05": frame.filter(
            pl.col("mann_whitney_q") <= FDR_THRESHOLD
        ).height,
        "concordant_positive_pairs_q05": discoveries.height,
        "associated_features": discoveries["feature_id"].n_unique(),
        "minimum_maximum_q": float(frame["maximum_q"].min()),
        "maximum_best_auprc": float(frame["best_auprc"].max()),
    }


def capacity_row(
    matrix: csr_matrix,
    uniform_family: pl.DataFrame,
    repeat_ids: np.ndarray,
    control_ids: np.ndarray,
    *,
    arm: str,
    block: int,
    orientation: str,
) -> dict[str, Any]:
    repeat = matrix[repeat_ids, :]
    control = matrix[control_ids, :]
    associated = (
        uniform_family.filter(pl.col("concordant_positive_association"))["feature_id"]
        .unique()
        .to_numpy()
    )
    eligible = uniform_family["feature_id"].n_unique()
    repeat_nnz = int(repeat.nnz)
    control_nnz = int(control.nnz)
    repeat_mass = float(repeat.sum())
    control_mass = float(control.sum())
    associated_repeat_nnz = int(repeat[:, associated].nnz) if associated.size else 0
    associated_repeat_mass = (
        float(repeat[:, associated].sum()) if associated.size else 0.0
    )
    return {
        "arm": arm,
        "block": block,
        "orientation": orientation,
        "dictionary_features": D_SAE,
        "features_observed_in_reference_panel": int(np.unique(matrix.indices).size),
        "eligible_repeat_features": eligible,
        "associated_repeat_features": int(associated.size),
        "associated_fraction_dictionary": associated.size / D_SAE,
        "associated_fraction_eligible": associated.size / eligible,
        "repeat_nonzero_slots": repeat_nnz,
        "control_nonzero_slots": control_nnz,
        "repeat_fraction_paired_nonzero_slots": repeat_nnz / (repeat_nnz + control_nnz),
        "repeat_activation_mass": repeat_mass,
        "control_activation_mass": control_mass,
        "repeat_fraction_paired_activation_mass": repeat_mass
        / (repeat_mass + control_mass),
        "associated_feature_repeat_nonzero_slots": associated_repeat_nnz,
        "associated_feature_fraction_repeat_nonzero_slots": associated_repeat_nnz
        / repeat_nnz,
        "associated_feature_repeat_activation_mass": associated_repeat_mass,
        "associated_feature_fraction_repeat_activation_mass": associated_repeat_mass
        / repeat_mass,
    }


def analyze(
    *, extraction_archive: Path, output_dir: Path, expected_archive_sha256: str
) -> dict[str, Any]:
    assert not output_dir.exists()
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert_commit(experiment_commit)
    assert os.environ.get("RUN_ID") == RUN_ID
    started = time.monotonic()
    archive = verify_archive(extraction_archive, expected_archive_sha256)
    extraction_root = extraction_archive / "extraction"
    extraction_manifest = verify_extraction(extraction_root)
    panel_root = extraction_archive / "inputs" / "panel" / "panel"
    uniform_pairs = pl.read_parquet(panel_root / "uniform_pairs.parquet").sort(
        "pair_id"
    )
    comparisons = pl.read_parquet(panel_root / "category_comparisons.parquet").sort(
        "level", "label", "pair_id"
    )
    assert uniform_pairs.height == 32_768
    assert comparisons.height == 24_576
    repeat_ids = uniform_pairs["repeat_context_id"].to_numpy()
    control_ids = uniform_pairs["control_context_id"].to_numpy()

    output_dir.mkdir(parents=True)
    artifacts: dict[str, Any] = {}
    summaries: dict[str, Any] = {}
    capacity_rows: list[dict[str, Any]] = []
    top_frames: list[pl.DataFrame] = []
    for block in BLOCKS:
        arm = arm_label(block - 1)
        summaries[arm] = {}
        for orientation in ORIENTATIONS:
            sparse_path = extraction_root / arm / f"sae_focal_{orientation}.parquet"
            print(
                json.dumps(
                    {"stage": "load_sparse", "arm": arm, "orientation": orientation}
                ),
                flush=True,
            )
            matrix = load_sparse(sparse_path)
            expected_rows = extraction_manifest["outputs"][arm][orientation]["rows"]
            assert matrix.nnz == expected_rows
            summaries[arm][orientation] = {}
            for hierarchy in HIERARCHIES:
                target_frames: list[pl.DataFrame] = []
                if hierarchy == "repeat":
                    target_frames.append(
                        add_family_metadata(
                            comparison_metrics(
                                matrix,
                                repeat_ids,
                                control_ids,
                                minimum_nonzero_support=MINIMUM_SUPPORT[hierarchy],
                            ),
                            arm=arm,
                            block=block,
                            orientation=orientation,
                            hierarchy=hierarchy,
                            target="repeat_vs_repeat_free",
                        )
                    )
                else:
                    current = comparisons.filter(pl.col("level") == hierarchy)
                    for target in current["label"].unique(maintain_order=True):
                        target_pairs = current.filter(pl.col("label") == target).sort(
                            "pair_id"
                        )
                        target_frames.append(
                            add_family_metadata(
                                comparison_metrics(
                                    matrix,
                                    target_pairs["positive_context_id"].to_numpy(),
                                    target_pairs["negative_context_id"].to_numpy(),
                                    minimum_nonzero_support=MINIMUM_SUPPORT[hierarchy],
                                ),
                                arm=arm,
                                block=block,
                                orientation=orientation,
                                hierarchy=hierarchy,
                                target=str(target),
                            )
                        )
                family = correct_family(target_frames)
                relative = Path("families") / arm / orientation / f"{hierarchy}.parquet"
                path = output_dir / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                family.write_parquet(path, compression="zstd")
                artifacts[str(relative)] = {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                summaries[arm][orientation][hierarchy] = family_summary(family)
                ranked = family.with_columns(
                    (~pl.col("concordant_positive_association")).alias("not_concordant")
                ).sort(
                    "target",
                    "not_concordant",
                    "maximum_q",
                    "standardized_mean_difference",
                    "feature_id",
                    descending=[False, False, False, True, False],
                )
                top_frames.append(
                    ranked.group_by("target", maintain_order=True).head(25)
                )
                if hierarchy == "repeat":
                    capacity_rows.append(
                        capacity_row(
                            matrix,
                            family,
                            repeat_ids,
                            control_ids,
                            arm=arm,
                            block=block,
                            orientation=orientation,
                        )
                    )
                print(
                    json.dumps(
                        {
                            "stage": "family_complete",
                            "arm": arm,
                            "orientation": orientation,
                            "hierarchy": hierarchy,
                            **summaries[arm][orientation][hierarchy],
                        }
                    ),
                    flush=True,
                )
            del matrix

    capacity = pl.DataFrame(capacity_rows).sort("block", "orientation")
    capacity_path = output_dir / "capacity_summary.parquet"
    capacity.write_parquet(capacity_path, compression="zstd")
    artifacts[capacity_path.name] = {
        "bytes": capacity_path.stat().st_size,
        "sha256": sha256_file(capacity_path),
    }
    top_hits = (
        pl.concat(top_frames, how="vertical")
        .drop("not_concordant")
        .sort("block", "orientation", "hierarchy", "target", "maximum_q", "feature_id")
    )
    top_path = output_dir / "top_hits.parquet"
    top_hits.write_parquet(top_path, compression="zstd")
    artifacts[top_path.name] = {
        "bytes": top_path.stat().st_size,
        "sha256": sha256_file(top_path),
    }
    result: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "issue": ISSUE,
        "run_id": RUN_ID,
        "analysis_status": "frozen_reference_repeat_capacity_associations",
        "experiment_commit": experiment_commit,
        "elapsed_seconds": time.monotonic() - started,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "polars": pl.__version__,
        "scipy": scipy.__version__,
        "input": {
            "extraction_run_id": EXTRACTION_RUN_ID,
            "extraction_archive_manifest_sha256": expected_archive_sha256,
            "extraction_experiment_commit": extraction_manifest["experiment_commit"],
            "extraction_objects_excluding_manifest": archive[
                "object_count_excluding_this_manifest"
            ],
            "panel_run_id": PANEL_RUN_ID,
            "panel_archive_manifest_sha256": PANEL_ARCHIVE_MANIFEST_SHA256,
            "contexts": CONTEXTS,
        },
        "protocol": {
            "layers_reported": list(BLOCKS),
            "training_tokens_per_sae": TRAINING_TOKENS,
            "orientations": list(ORIENTATIONS),
            "pooling": "focal",
            "hierarchies": list(HIERARCHIES),
            "minimum_nonzero_support": MINIMUM_SUPPORT,
            "tests": ["Welch t", "Mann-Whitney U"],
            "mann_whitney": (
                "exact U from sparse nonzeros with analytical zero/equal-value ties; "
                "asymptotic two-sided p with continuity correction"
            ),
            "descriptive_metric": "AUPRC in raw and sign-reversed direction",
            "bh_family": (
                "within each layer x orientation x hierarchy x statistic, across all "
                "supported feature-target pairs"
            ),
            "fdr_threshold": FDR_THRESHOLD,
            "association_call": "positive mean effect and both BH q <= 0.05",
            "uses_all_frozen_contexts": True,
            "uses_outcome_for_feature_support": False,
        },
        "summaries": summaries,
        "capacity": capacity.to_dicts(),
        "artifacts": artifacts,
    }
    result_path = output_dir / "results.json"
    write_json(result_path, result)
    result["artifacts"]["results.json"] = {
        "bytes": result_path.stat().st_size,
        "sha256": sha256_file(result_path),
    }
    write_json(output_dir / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    args = parser.parse_args()
    result = analyze(
        extraction_archive=args.extraction_archive,
        output_dir=args.output_dir,
        expected_archive_sha256=args.expected_archive_sha256,
    )
    print(json.dumps(result["capacity"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
