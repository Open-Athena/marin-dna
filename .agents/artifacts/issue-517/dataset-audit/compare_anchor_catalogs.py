#!/usr/bin/env python3
"""Compare issue-517 and issue-473 CDS anchor catalogs without sequence rows."""

from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import json
import sys
from typing import Any

import polars as pl


WINDOW_SIZE = 255
DISTANCE_THRESHOLDS = (0, 1, 32, 64, 127, 128, 254, 255, 512, 1024, 4096)


def _quantiles(values: list[float]) -> dict[str, float]:
    series = pl.Series(values)
    return {
        str(q): float(series.quantile(q, interpolation="nearest"))
        for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    }


def _column_summary(frame: pl.DataFrame, column: str) -> dict[str, Any]:
    values = frame[column].cast(pl.Float64).to_list()
    return {
        "mean": sum(values) / len(values),
        "quantiles": _quantiles(values),
    }


def _starts_by_chrom(frame: pl.DataFrame) -> dict[str, list[int]]:
    starts: dict[str, list[int]] = defaultdict(list)
    for chrom, start in frame.select("source_chrom", "source_start").iter_rows():
        starts[str(chrom)].append(int(start))
    for chrom in starts:
        starts[chrom].sort()
    return starts


def _nearest_distances(
    source: dict[str, list[int]], target: dict[str, list[int]]
) -> list[float]:
    distances: list[float] = []
    for chrom, starts in source.items():
        candidates = target.get(chrom, [])
        for start in starts:
            position = bisect_left(candidates, start)
            neighbors = candidates[max(0, position - 1) : position + 1]
            distances.append(
                float(min(abs(start - candidate) for candidate in neighbors))
                if neighbors
                else float("inf")
            )
    return distances


def _distance_summary(distances: list[float]) -> dict[str, Any]:
    finite = [value for value in distances if value != float("inf")]
    return {
        "quantiles_bp": _quantiles(finite),
        "fraction_at_or_below_bp": {
            str(threshold): sum(value <= threshold for value in distances)
            / len(distances)
            for threshold in DISTANCE_THRESHOLDS
        },
        "mean_nearest_window_overlap_fraction": sum(
            max(0.0, WINDOW_SIZE - value) / WINDOW_SIZE for value in distances
        )
        / len(distances),
    }


def _merged_intervals(frame: pl.DataFrame) -> dict[str, list[tuple[int, int]]]:
    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for chrom, start, end in frame.select(
        "source_chrom", "source_start", "source_end"
    ).iter_rows():
        grouped[str(chrom)].append((int(start), int(end)))

    merged: dict[str, list[tuple[int, int]]] = {}
    for chrom, intervals in grouped.items():
        output: list[tuple[int, int]] = []
        for start, end in sorted(intervals):
            if not output or start > output[-1][1]:
                output.append((start, end))
            else:
                output[-1] = (output[-1][0], max(output[-1][1], end))
        merged[chrom] = output
    return merged


def _covered_bases(intervals: dict[str, list[tuple[int, int]]]) -> int:
    return sum(end - start for values in intervals.values() for start, end in values)


def _intersection_bases(
    left: dict[str, list[tuple[int, int]]],
    right: dict[str, list[tuple[int, int]]],
) -> int:
    total = 0
    for chrom, left_values in left.items():
        right_values = right.get(chrom, [])
        left_index = 0
        right_index = 0
        while left_index < len(left_values) and right_index < len(right_values):
            left_start, left_end = left_values[left_index]
            right_start, right_end = right_values[right_index]
            total += max(0, min(left_end, right_end) - max(left_start, right_start))
            if left_end <= right_end:
                left_index += 1
            else:
                right_index += 1
    return total


def _chromosome_counts(frame: pl.DataFrame) -> dict[str, int]:
    return dict(Counter(frame["source_chrom"].to_list()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", required=True)
    parser.add_argument("--historical", required=True)
    parser.add_argument("--historical-labels", required=True)
    parser.add_argument("--current-ownership", required=True)
    parser.add_argument("--current-construction-drops", required=True)
    args = parser.parse_args()

    current_columns = [
        "source_chrom",
        "source_start",
        "source_end",
        "region_label",
        "proportion_conserved",
        "raw_cds_fraction",
        "owned_cds_fraction",
        "source_arm_owned_fraction",
        "union_functional_fraction",
        "exon_fraction",
        "contributing_feature_count",
    ]
    historical_columns = [
        "source_chrom",
        "source_start",
        "source_end",
        "region_label",
        "proportion_conserved",
    ]
    current = (
        pl.scan_parquet(args.current)
        .select(current_columns)
        .filter(pl.col("region_label") == "cds")
        .sort("source_chrom", "source_start")
        .collect()
    )
    historical = (
        pl.scan_parquet(args.historical)
        .select(historical_columns)
        .filter(pl.col("region_label") == "cds")
        .sort("source_chrom", "source_start")
        .collect()
    )
    historical_labels = (
        pl.scan_parquet(args.historical_labels)
        .select(
            "label",
            "functional_frac",
            "cds_frac",
            "gene_body_frac",
            "intron_frac",
            "intergenic_frac",
        )
        .filter(pl.col("label") == "cds")
        .collect()
    )
    current_ownership = (
        pl.scan_parquet(args.current_ownership)
        .select(
            "source_arm",
            "ownership_winner",
            "passes_ownership_gate",
            "raw_cds_fraction",
            "owned_cds_fraction",
            "union_functional_fraction",
            "exon_fraction",
        )
        .filter(pl.col("source_arm") == "cds")
        .collect()
    )
    current_construction_drops = (
        pl.scan_parquet(args.current_construction_drops)
        .select("source_arm", "drop_reason")
        .filter(pl.col("source_arm") == "cds")
        .collect()
    )
    for frame in (current, historical):
        assert ((frame["source_end"] - frame["source_start"]) == WINDOW_SIZE).all()

    current_coordinates = set(
        current.select("source_chrom", "source_start", "source_end").iter_rows()
    )
    historical_coordinates = set(
        historical.select("source_chrom", "source_start", "source_end").iter_rows()
    )
    exact_intersection = len(current_coordinates & historical_coordinates)

    current_starts = _starts_by_chrom(current)
    historical_starts = _starts_by_chrom(historical)
    current_to_historical = _nearest_distances(current_starts, historical_starts)
    historical_to_current = _nearest_distances(historical_starts, current_starts)

    current_union = _merged_intervals(current)
    historical_union = _merged_intervals(historical)
    current_bases = _covered_bases(current_union)
    historical_bases = _covered_bases(historical_union)
    intersection_bases = _intersection_bases(current_union, historical_union)

    result = {
        "coordinate_system": "0-based half-open",
        "window_size": WINDOW_SIZE,
        "anchor_counts": {
            "current": current.height,
            "historical": historical.height,
            "current_minus_historical": current.height - historical.height,
            "current_over_historical": current.height / historical.height,
        },
        "exact_coordinate_overlap": {
            "count": exact_intersection,
            "fraction_of_current": exact_intersection / current.height,
            "fraction_of_historical": exact_intersection / historical.height,
        },
        "nearest_start_distance": {
            "current_to_historical": _distance_summary(current_to_historical),
            "historical_to_current": _distance_summary(historical_to_current),
        },
        "union_base_coverage": {
            "current_bases": current_bases,
            "historical_bases": historical_bases,
            "intersection_bases": intersection_bases,
            "current_covered_by_historical_fraction": intersection_bases
            / current_bases,
            "historical_covered_by_current_fraction": intersection_bases
            / historical_bases,
            "jaccard": intersection_bases
            / (current_bases + historical_bases - intersection_bases),
        },
        "chromosome_counts": {
            "current": _chromosome_counts(current),
            "historical": _chromosome_counts(historical),
        },
        "proportion_conserved": {
            "current": _column_summary(current, "proportion_conserved"),
            "historical": _column_summary(historical, "proportion_conserved"),
        },
        "anchor_purity": {
            "current_training": {
                column: _column_summary(current, column)
                for column in (
                    "raw_cds_fraction",
                    "owned_cds_fraction",
                    "union_functional_fraction",
                    "exon_fraction",
                )
            },
            "historical_training": {
                column: _column_summary(historical_labels, column)
                for column in (
                    "cds_frac",
                    "functional_frac",
                    "gene_body_frac",
                    "intron_frac",
                    "intergenic_frac",
                )
            },
        },
        "current_preconservation_gates": {
            "construction_valid": current_ownership.height,
            "construction_drops": {
                reason: count
                for reason, count in current_construction_drops.group_by(
                    "drop_reason"
                ).len().iter_rows()
            },
            "ownership_outcomes": {
                ("retained" if passes else f"lost_to_{winner}"): count
                for passes, winner, count in current_ownership.group_by(
                    "passes_ownership_gate", "ownership_winner"
                ).len().iter_rows()
            },
            "retained_purity": {
                column: _column_summary(
                    current_ownership.filter(pl.col("passes_ownership_gate")), column
                )
                for column in (
                    "raw_cds_fraction",
                    "owned_cds_fraction",
                    "union_functional_fraction",
                    "exon_fraction",
                )
            },
            "dropped_purity": {
                column: _column_summary(
                    current_ownership.filter(~pl.col("passes_ownership_gate")), column
                )
                for column in (
                    "raw_cds_fraction",
                    "owned_cds_fraction",
                    "union_functional_fraction",
                    "exon_fraction",
                )
            },
        },
        "current_only_anchor_attributes": {
            column: _column_summary(current, column)
            for column in (
                "raw_cds_fraction",
                "owned_cds_fraction",
                "source_arm_owned_fraction",
                "union_functional_fraction",
                "exon_fraction",
                "contributing_feature_count",
            )
        },
    }
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
