"""Annotate the official Mendelian panel for paired repeat-variant analysis.

Input positions are VCF-style 1-based coordinates. They are converted exactly
once to 0-based half-open focal coordinates before joining RepeatMasker.
No SAE activation or Mendelian label is read into the output panel.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from common import ISSUE, assert_commit, sha256_file, write_json
from panel_common import FOCAL_INDEX, WINDOW_BP
from prepare_reference_panel import (
    add_hierarchy_labels,
    annotate_points,
    merge_intervals,
    repeat_covered_bp,
    validate_inventory,
)
from variant_common import (
    EXPECTED_MATCH_GROUPS,
    EXPECTED_VARIANTS,
    MIN_CATEGORY_VARIANTS,
    POSITION_STATUSES,
    REPEAT_INTERIOR_BP,
    SOURCE_DATASET_ID,
    SOURCE_DATASET_REVISION,
    SOURCE_PANEL_SHA256,
    VARIANT_PANEL_RUN_ID,
)

NUCLEOTIDES = frozenset("ACGT")
PASSTHROUGH_COLUMNS = (
    "chrom",
    "pos",
    "ref",
    "alt",
    "subset",
    "match_group",
    "split",
)


def validate_source_panel(
    panel: pl.DataFrame, manifest: dict[str, Any], panel_path: Path
) -> None:
    required = set(PASSTHROUGH_COLUMNS) | {"label"}
    assert required <= set(panel.columns), required - set(panel.columns)
    assert manifest["dataset"] == {
        "id": SOURCE_DATASET_ID,
        "revision": SOURCE_DATASET_REVISION,
        "split": "train",
    }
    assert manifest["coordinate_boundary"] == "dataset pos1 -> pos0 = pos1 - 1"
    assert manifest["panel_sha256"] == SOURCE_PANEL_SHA256 == sha256_file(panel_path)
    assert manifest["row_count"] == panel.height == EXPECTED_VARIANTS
    assert (
        manifest["match_groups"]
        == panel["match_group"].n_unique()
        == EXPECTED_MATCH_GROUPS
    )
    assert (
        panel.select(pl.struct("chrom", "pos", "ref", "alt").n_unique()).item()
        == panel.height
    )
    assert panel.filter(pl.col("pos") < 1).is_empty()
    assert panel.filter(pl.col("ref") == pl.col("alt")).is_empty()
    assert panel.filter(
        ~pl.col("ref").is_in(sorted(NUCLEOTIDES))
        | ~pl.col("alt").is_in(sorted(NUCLEOTIDES))
    ).is_empty()
    assert panel.select(pl.col("label").unique().sort()).to_series().to_list() == [0, 1]


def union_index(repeat_union: pl.DataFrame) -> dict[str, tuple[Any, Any]]:
    result: dict[str, tuple[Any, Any]] = {}
    for current in repeat_union.partition_by("chrom", maintain_order=True):
        chrom = str(current["chrom"][0])
        ordered = current.sort("start0", "end0")
        result[chrom] = (
            ordered["start0"].to_numpy(),
            ordered["end0"].to_numpy(),
        )
    assert result
    return result


def annotate_variant_panel(
    panel: pl.DataFrame,
    annotations: pl.DataFrame,
    repeat_union: pl.DataFrame,
    *,
    expected_rows: int = EXPECTED_VARIANTS,
) -> pl.DataFrame:
    """Return an outcome-free row-aligned repeat annotation panel."""

    points = [
        (str(chrom), int(pos) - 1)
        for chrom, pos in panel.select("chrom", "pos").iter_rows()
    ]
    assert len(points) == panel.height
    point_annotations = annotate_points(points, annotations)
    coverage_index = union_index(repeat_union)

    rows: list[dict[str, Any]] = []
    for panel_row, source in enumerate(
        panel.select(PASSTHROUGH_COLUMNS).iter_rows(named=True)
    ):
        chrom = str(source["chrom"])
        pos1 = int(source["pos"])
        pos0 = pos1 - 1
        start0 = pos0 - FOCAL_INDEX
        end0 = pos0 + FOCAL_INDEX + 1
        assert start0 >= 0 and end0 - start0 == WINDOW_BP
        covered_bp = repeat_covered_bp(chrom, start0, end0, coverage_index)
        annotation = point_annotations[(chrom, pos0)]
        if annotation is not None:
            position_status = "focal_repeat"
        elif covered_bp == 0:
            position_status = "repeat_free_window"
        else:
            position_status = "near_repeat"

        row = {
            "panel_row": panel_row,
            **source,
            "pos0": pos0,
            "window_start0": start0,
            "window_end0": end0,
            "allele_change": f"{source['ref']}>{source['alt']}",
            "position_status": position_status,
            "repeat_covered_bp": covered_bp,
            "repeat_fraction": covered_bp / WINDOW_BP,
            "annotation_id": None,
            "repeat_start0": None,
            "repeat_end0": None,
            "sw_score": None,
            "milli_div": None,
            "repeat_strand": None,
            "repeat_name": None,
            "repeat_class": None,
            "repeat_family": None,
            "family_label": None,
            "subfamily_label": None,
            "boundary_distance": None,
            "overlap_count": 0,
            "overlap_annotation_ids": [],
            "overlap_subfamilies": [],
            "unique_repeat_overlap": False,
            "repeat_interior_32": False,
        }
        if annotation is not None:
            row.update(
                {
                    "annotation_id": annotation["annotation_id"],
                    "repeat_start0": annotation["start0"],
                    "repeat_end0": annotation["end0"],
                    "sw_score": annotation["sw_score"],
                    "milli_div": annotation["milli_div"],
                    "repeat_strand": annotation["repeat_strand"],
                    "repeat_name": annotation["repeat_name"],
                    "repeat_class": annotation["repeat_class"],
                    "repeat_family": annotation["repeat_family"],
                    "family_label": annotation["family_label"],
                    "subfamily_label": annotation["subfamily_label"],
                    "boundary_distance": annotation["boundary_distance"],
                    "overlap_count": annotation["overlap_count"],
                    "overlap_annotation_ids": annotation["overlap_annotation_ids"],
                    "overlap_subfamilies": annotation["overlap_subfamilies"],
                    "unique_repeat_overlap": annotation["overlap_count"] == 1,
                    "repeat_interior_32": (
                        annotation["boundary_distance"] >= REPEAT_INTERIOR_BP
                    ),
                }
            )
        rows.append(row)

    result = pl.DataFrame(
        rows,
        schema_overrides={
            "panel_row": pl.UInt32,
            "annotation_id": pl.UInt32,
            "overlap_annotation_ids": pl.List(pl.UInt32),
        },
    )
    assert result.height == panel.height == expected_rows
    assert "label" not in result.columns
    assert result["panel_row"].to_list() == list(range(expected_rows))
    assert set(result["position_status"].unique()) <= set(POSITION_STATUSES)
    assert result.filter(
        (pl.col("position_status") == "focal_repeat")
        != pl.col("annotation_id").is_not_null()
    ).is_empty()
    assert result.filter(
        (pl.col("position_status") == "repeat_free_window")
        != (pl.col("repeat_covered_bp") == 0)
    ).is_empty()
    assert result.filter(
        (pl.col("repeat_fraction") < 0) | (pl.col("repeat_fraction") > 1)
    ).is_empty()
    return result


def summarize_categories(panel: pl.DataFrame) -> pl.DataFrame:
    focal = panel.filter(pl.col("position_status") == "focal_repeat")
    frames: list[pl.DataFrame] = []
    for level, column in (
        ("class", "repeat_class"),
        ("family", "family_label"),
        ("subfamily", "subfamily_label"),
    ):
        frames.append(
            focal.group_by(column)
            .agg(
                pl.len().alias("variants"),
                pl.col("subset").n_unique().alias("subsets"),
                pl.col("chrom").n_unique().alias("chromosomes"),
                pl.col("repeat_interior_32").sum().alias("interior_32_variants"),
                pl.col("unique_repeat_overlap").sum().alias("unique_overlap_variants"),
            )
            .rename({column: "category"})
            .with_columns(
                pl.lit(level).alias("level"),
                (pl.col("variants") >= MIN_CATEGORY_VARIANTS).alias("eligible_32"),
            )
            .select(
                "level",
                "category",
                "variants",
                "subsets",
                "chromosomes",
                "interior_32_variants",
                "unique_overlap_variants",
                "eligible_32",
            )
        )
    return pl.concat(frames).sort(
        "level", "variants", "category", descending=[False, True, False]
    )


def materialize(
    panel_path: Path,
    panel_manifest_path: Path,
    inventory_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    assert panel_path.is_file() and panel_manifest_path.is_file()
    assert not output_dir.exists()
    panel = pl.read_parquet(panel_path)
    source_manifest = json.loads(panel_manifest_path.read_text())
    validate_source_panel(panel, source_manifest, panel_path)
    inventory_manifest, annotations, _ = validate_inventory(inventory_dir)
    annotations = add_hierarchy_labels(annotations)
    repeat_union = merge_intervals(annotations)
    variant_panel = annotate_variant_panel(panel, annotations, repeat_union)
    category_counts = summarize_categories(variant_panel)
    status_by_subset = (
        variant_panel.group_by("subset", "position_status")
        .agg(
            pl.len().alias("variants"),
            pl.col("match_group").n_unique().alias("match_groups"),
        )
        .sort("subset", "position_status")
    )

    output_dir.mkdir(parents=True)
    panel_output = output_dir / "variant_panel.parquet"
    category_output = output_dir / "category_counts.parquet"
    subset_output = output_dir / "status_by_subset.parquet"
    variant_panel.write_parquet(panel_output, compression="zstd")
    category_counts.write_parquet(category_output, compression="zstd")
    status_by_subset.write_parquet(subset_output, compression="zstd")

    status_counts = {
        str(status): int(count)
        for status, count in variant_panel.group_by("position_status")
        .len()
        .sort("position_status")
        .iter_rows()
    }
    eligible_counts = {
        level: category_counts.filter(
            (pl.col("level") == level) & pl.col("eligible_32")
        ).height
        for level in ("class", "family", "subfamily")
    }
    artifacts = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in (panel_output, category_output, subset_output)
    }
    result = {
        "issue": ISSUE,
        "run_id": VARIANT_PANEL_RUN_ID,
        "analysis_status": "outcome_blind_paired_repeat_variant_panel",
        "created_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "experiment_commit": os.environ.get("EXPERIMENT_COMMIT", ""),
        "platform": platform.platform(),
        "polars": pl.__version__,
        "coordinate_system": "source pos is 1-based; pos0/start0/end0 are 0-based half-open",
        "source_panel": {
            "dataset": source_manifest["dataset"],
            "bytes": panel_path.stat().st_size,
            "sha256": sha256_file(panel_path),
            "rows": panel.height,
            "match_groups": panel["match_group"].n_unique(),
            "label_policy": "validated at input, excluded from every output artifact",
        },
        "repeat_inventory": {
            "run_id": inventory_manifest["run_id"],
            "primary_records": inventory_manifest["primary_records"],
            "source_sha256": inventory_manifest["source"]["sha256"],
        },
        "definitions": {
            "focal_repeat": "variant pos0 overlaps at least one RepeatMasker record",
            "near_repeat": "focal base is non-repeat but the 255 bp window overlaps repeat union",
            "repeat_free_window": "the complete 255 bp window has zero repeat-union overlap",
            "primary_annotation": (
                "highest SW score, then lower divergence, longer span, stable annotation ID"
            ),
            "category_eligibility": f"at least {MIN_CATEGORY_VARIANTS} focal-repeat variants",
        },
        "status_counts": status_counts,
        "eligible_categories": eligible_counts,
        "artifacts": artifacts,
    }
    assert_commit(result["experiment_commit"])
    write_json(output_dir / "results.json", result)
    result["artifacts"]["results.json"] = {
        "bytes": (output_dir / "results.json").stat().st_size,
        "sha256": sha256_file(output_dir / "results.json"),
    }
    write_json(output_dir / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--inventory-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = materialize(
        args.panel,
        args.panel_manifest,
        args.inventory_dir,
        args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
