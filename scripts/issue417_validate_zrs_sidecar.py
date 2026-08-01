"""Validate the issue #417 ZRS projection-QC sidecar.

The two named loci are positive controls only. They must demonstrate broad
cross-clade recovery without entering the conservation-filtered training grid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import polars as pl


EXPECTED_ZRS = {
    "zrs_EH38E2604086": ("chr7", 156_791_360, 156_791_615),
    "zrs_EH38E2604087": ("chr7", 156_792_658, 156_792_913),
}
TARGET_LENGTH = 255
MINIMUM_NON_MAMMAL_CLADES = 2


def _single_row(scan: pl.LazyFrame, *expressions: pl.Expr) -> dict[str, Any]:
    return scan.select(*expressions).collect(engine="streaming").row(0, named=True)


def validate_zrs_sidecar(
    results: Path,
    *,
    expected_pipeline_commit: str,
) -> dict[str, object]:
    assert results.is_dir(), f"missing sidecar results directory: {results}"
    assert re.fullmatch(r"[0-9a-f]{40}", expected_pipeline_commit)

    manifest = pl.read_csv(results / "metadata/species_active.tsv", separator="\t")
    assert manifest["selected"].all()
    backend_counts = {
        str(backend): int(count)
        for backend, count in manifest.group_by("backend").len().iter_rows()
    }
    assert backend_counts == {"zoonomia_cactus": 2, "ucsc_multiz100way": 5}

    sequences = pl.scan_parquet(results / "sequences/all_sources.parquet")
    zrs = sequences.filter(
        pl.col("query_name").str.to_lowercase().str.starts_with("zrs_")
    )
    query_names = set(
        zrs.select("query_name")
        .unique()
        .collect(engine="streaming")["query_name"]
        .to_list()
    )
    assert query_names == set(EXPECTED_ZRS)

    coordinates = (
        zrs.select("query_name", "source_chrom", "source_start", "source_end")
        .unique()
        .collect(engine="streaming")
    )
    assert coordinates.height == len(EXPECTED_ZRS)
    observed_coordinates = {
        str(query_name): (str(chrom), int(start), int(end))
        for query_name, chrom, start, end in coordinates.iter_rows()
    }
    assert observed_coordinates == EXPECTED_ZRS

    invalid = _single_row(
        zrs,
        (pl.col("sequence").str.len_bytes() != TARGET_LENGTH)
        .sum()
        .alias("invalid_sequence_length"),
        (pl.col("source_end") - pl.col("source_start") != TARGET_LENGTH)
        .sum()
        .alias("invalid_source_span"),
        (pl.col("t_end") - pl.col("t_start") != TARGET_LENGTH)
        .sum()
        .alias("invalid_target_span"),
        (~pl.col("t_strand").is_in(["+", "-"])).sum().alias("invalid_strand"),
    )
    assert all(int(value) == 0 for value in invalid.values()), invalid

    recovery = (
        zrs.group_by("query_name")
        .agg(
            pl.len().alias("rows"),
            (pl.col("alignment_source") == "human_reference").sum().alias("human_rows"),
            (pl.col("alignment_source") == "zoonomia_cactus")
            .sum()
            .alias("mammal_target_rows"),
            (pl.col("alignment_source") == "ucsc_multiz100way")
            .sum()
            .alias("non_mammal_rows"),
            pl.col("clade")
            .filter(pl.col("alignment_source") == "ucsc_multiz100way")
            .n_unique()
            .alias("non_mammal_clades"),
        )
        .sort("query_name")
        .collect(engine="streaming")
    )
    assert recovery.height == len(EXPECTED_ZRS)
    assert (recovery["human_rows"] == 1).all()
    assert (recovery["mammal_target_rows"] >= 1).all()
    assert (recovery["non_mammal_rows"] >= MINIMUM_NON_MAMMAL_CLADES).all()
    assert (recovery["non_mammal_clades"] >= MINIMUM_NON_MAMMAL_CLADES).all()

    per_anchor = pl.read_parquet(results / "qc/per_anchor.parquet")
    zrs_qc = per_anchor.filter(pl.col("query_name").is_in(list(EXPECTED_ZRS)))
    assert zrs_qc.height == len(EXPECTED_ZRS)
    assert (zrs_qc["accepted_non_mammal_projections"] >= 2).all()

    sample = pl.read_csv(results / "qc/manual_inspection_sample.tsv", separator="\t")
    assert set(EXPECTED_ZRS) <= set(sample["query_name"].to_list())
    report = (results / "qc/manual_inspection.md").read_text().lower()
    assert "pending human review" in report
    assert "required zrs anchors" in report
    assert all(name.lower() in report for name in EXPECTED_ZRS)

    return {
        "results": str(results.resolve()),
        "expected_pipeline_commit": expected_pipeline_commit,
        "coordinate_system": "0-based half-open",
        "active_target_species": manifest.height,
        "backend_species": backend_counts,
        "zrs_recovery": recovery.to_dicts(),
        "manual_inspection": "pending human review",
        "status": "automated ZRS sidecar validation passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--expected-pipeline-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = validate_zrs_sidecar(
        args.results,
        expected_pipeline_commit=args.expected_pipeline_commit,
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
