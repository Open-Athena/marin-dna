"""Independently recompute Stage-9 summaries from its archived family tables."""

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
from polars.testing import assert_frame_equal

from analyze_repeat_label_sensitivity import (
    FDR_THRESHOLD,
    FEATURE_OF_INTEREST,
    GLOBAL_ASSOCIATION_MANIFEST_SHA256,
    PAIRED_REPEAT_ARCHIVE_SHA256,
    REFERENCE_ASSOCIATION_ARCHIVE_SHA256,
    RUN_ID,
    STRATA,
    add_stratum_calls,
    contingency_tables,
    inventory_overlap,
    load_aligned_panel,
    repeat_free_retention,
    repeat_inventory_sets,
    strand_overlap,
    target_definitions,
    target_summary,
    verify_inner_archive,
    verify_outer_archive,
)
from association_common import bh_adjust
from common import ISSUE, assert_commit, sha256_file, write_json
from variant_analysis_common import VARIANT_PANEL_ARCHIVE_SHA256

AUDIT_RUN_ID = "dna-exp435-repeat-label-sensitivity-audit-r1"


def verify_hash_complete_outer(root: Path, expected_sha256: str) -> dict[str, Any]:
    manifest_path = root / "archive_manifest.json"
    assert manifest_path.is_file() and sha256_file(manifest_path) == expected_sha256
    manifest = json.loads(manifest_path.read_text())
    assert manifest["run_id"] == RUN_ID
    assert manifest["analysis_status"] == (
        "frozen_repeat_aware_mendelian_label_sensitivity"
    )
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    assert len(paths) == manifest["object_count_excluding_this_manifest"] + 1
    for relative, metadata in manifest["artifacts"].items():
        path = root / relative
        assert path.is_file() and path.stat().st_size == metadata["bytes"]
        assert sha256_file(path) == metadata["sha256"]
    return manifest


def validate_family(frame: pl.DataFrame, stratum: str) -> pl.DataFrame:
    """Recompute corrected p values and the strict two-test discovery call."""

    expected_welch = bh_adjust(frame["welch_p"].to_numpy())
    expected_mann = bh_adjust(frame["mann_whitney_p"].to_numpy())
    np.testing.assert_allclose(frame["welch_q"].to_numpy(), expected_welch)
    np.testing.assert_allclose(frame["mann_whitney_q"].to_numpy(), expected_mann)
    base = frame.drop("repeat_stratum", "maximum_q", "concordant_discovery")
    expected = add_stratum_calls(base, stratum)
    assert_frame_equal(frame, expected, check_exact=True)
    assert frame.filter(
        pl.col("concordant_discovery")
        & (
            (pl.col("welch_q") > FDR_THRESHOLD)
            | (pl.col("mann_whitney_q") > FDR_THRESHOLD)
        )
    ).is_empty()
    return frame


def assert_recomputed(actual: pl.DataFrame, expected_path: Path) -> None:
    expected = pl.read_parquet(expected_path)
    assert_frame_equal(actual, expected, check_exact=False, rtol=1e-12, atol=1e-12)


def audit(
    *,
    result_root: Path,
    expected_outer_sha256: str,
    label_panel_path: Path,
    repeat_panel_root: Path,
    global_association_root: Path,
    reference_association_root: Path,
    paired_repeat_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    assert not output_dir.exists()
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert_commit(experiment_commit)
    assert os.environ.get("RUN_ID") == AUDIT_RUN_ID
    started = time.perf_counter()
    outer = verify_hash_complete_outer(result_root, expected_outer_sha256)
    analysis_root = result_root / "analysis"
    analysis_manifest = json.loads((analysis_root / "manifest.json").read_text())
    assert analysis_manifest["run_id"] == RUN_ID
    for relative, metadata in analysis_manifest["artifacts"].items():
        path = analysis_root / relative
        assert path.is_file() and path.stat().st_size == metadata["bytes"]
        assert sha256_file(path) == metadata["sha256"]

    verify_outer_archive(
        repeat_panel_root,
        VARIANT_PANEL_ARCHIVE_SHA256,
        "outcome_blind_paired_repeat_variant_panel",
    )
    verify_inner_archive(
        global_association_root,
        GLOBAL_ASSOCIATION_MANIFEST_SHA256,
        "dna-exp436-mendelian-focal-associations-seed288-r2",
    )
    verify_outer_archive(
        reference_association_root,
        REFERENCE_ASSOCIATION_ARCHIVE_SHA256,
        "frozen_reference_repeat_capacity_associations",
    )
    verify_outer_archive(
        paired_repeat_root,
        PAIRED_REPEAT_ARCHIVE_SHA256,
        "frozen_paired_repeat_variant_delta_associations",
    )

    family_paths = sorted(analysis_root.glob("families/*/*/*/*.parquet"))
    assert len(family_paths) == 48
    families: list[pl.DataFrame] = []
    for path in family_paths:
        arm, orientation, stratum, response = path.relative_to(
            analysis_root / "families"
        ).parts
        response = response.removesuffix(".parquet")
        assert stratum in STRATA
        frame = validate_family(pl.read_parquet(path), stratum)
        assert frame["arm"].unique().to_list() == [arm]
        assert frame["orientation"].unique().to_list() == [orientation]
        assert frame["repeat_stratum"].unique().to_list() == [stratum]
        assert frame["response"].unique().to_list() == [response]
        families.append(frame)
    combined = pl.concat(families, how="vertical")

    assert_recomputed(
        target_summary(combined), analysis_root / "target_summary.parquet"
    )
    assert_recomputed(
        strand_overlap(combined), analysis_root / "strand_overlap.parquet"
    )
    assert_recomputed(
        repeat_free_retention(combined),
        analysis_root / "repeat_free_retention.parquet",
    )
    feature = combined.filter(
        (pl.col("block") == 19) & (pl.col("feature_id") == FEATURE_OF_INTEREST)
    ).sort("orientation", "response", "repeat_stratum", "target_kind", "target")
    assert_recomputed(feature, analysis_root / "feature9086.parquet")

    panel = load_aligned_panel(label_panel_path, repeat_panel_root)
    _, counts = target_definitions(panel)
    omnibus, pairwise = contingency_tables(panel)
    assert_recomputed(counts, analysis_root / "stratum_target_counts.parquet")
    assert_recomputed(omnibus, analysis_root / "label_repeat_status_omnibus.parquet")
    assert_recomputed(pairwise, analysis_root / "label_repeat_status_pairwise.parquet")
    reference_sets, paired_sets = repeat_inventory_sets(
        reference_association_root, paired_repeat_root
    )
    assert_recomputed(
        inventory_overlap(combined, reference_sets, paired_sets),
        analysis_root / "inventory_overlap.parquet",
    )

    reproduction = analysis_manifest["global_reproduction"]
    assert len(reproduction) == 12
    assert all(
        item["maximum_absolute_error"] <= 1e-12 for item in reproduction.values()
    )
    output_dir.mkdir(parents=True)
    result = {
        "issue": ISSUE,
        "run_id": AUDIT_RUN_ID,
        "analysis_status": "independent_repeat_label_sensitivity_audit",
        "created_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "experiment_commit": experiment_commit,
        "platform": platform.platform(),
        "input": {
            "run_id": outer["run_id"],
            "outer_manifest_sha256": expected_outer_sha256,
            "objects": outer["object_count_excluding_this_manifest"],
            "bytes": outer["bytes_excluding_this_manifest"],
        },
        "checks": {
            "family_files": len(family_paths),
            "family_rows": combined.height,
            "all_artifact_hashes": True,
            "all_48_bh_families_recomputed": True,
            "all_summary_tables_recomputed": True,
            "panel_identity_reverified": True,
            "global_reproduction_families": len(reproduction),
            "maximum_global_reproduction_error": max(
                item["maximum_absolute_error"] for item in reproduction.values()
            ),
        },
    }
    write_json(output_dir / "audit.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--expected-outer-sha256", required=True)
    parser.add_argument("--label-panel", type=Path, required=True)
    parser.add_argument("--repeat-panel-root", type=Path, required=True)
    parser.add_argument("--global-association-root", type=Path, required=True)
    parser.add_argument("--reference-association-root", type=Path, required=True)
    parser.add_argument("--paired-repeat-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        result_root=args.result_root,
        expected_outer_sha256=args.expected_outer_sha256,
        label_panel_path=args.label_panel,
        repeat_panel_root=args.repeat_panel_root,
        global_association_root=args.global_association_root,
        reference_association_root=args.reference_association_root,
        paired_repeat_root=args.paired_repeat_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
