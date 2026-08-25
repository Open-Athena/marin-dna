#!/usr/bin/env python3
"""Audit development VEP point coverage across CDS anchor constructions."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
import json
import sys
from typing import Any

import polars as pl


SUBSETS = ("missense_variant", "splicing", "synonymous_variant")


def _normalized_chrom(value: object) -> str:
    chrom = str(value)
    return chrom if chrom.startswith("chr") else f"chr{chrom}"


class IntervalIndex:
    """Merged 0-based, half-open intervals indexed by chromosome."""

    def __init__(self, frame: pl.DataFrame, chrom: str, start: str, end: str) -> None:
        grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for chrom_value, start_value, end_value in frame.select(
            chrom, start, end
        ).iter_rows():
            grouped[_normalized_chrom(chrom_value)].append(
                (int(start_value), int(end_value))
            )

        self._intervals: dict[str, tuple[list[int], list[int]]] = {}
        for chrom_value, intervals in grouped.items():
            merged: list[tuple[int, int]] = []
            for start_value, end_value in sorted(intervals):
                if not merged or start_value > merged[-1][1]:
                    merged.append((start_value, end_value))
                else:
                    merged[-1] = (
                        merged[-1][0],
                        max(merged[-1][1], end_value),
                    )
            self._intervals[chrom_value] = (
                [value[0] for value in merged],
                [value[1] for value in merged],
            )

    def contains(self, chrom: str, position: int) -> bool:
        values = self._intervals.get(chrom)
        if values is None:
            return False
        starts, ends = values
        index = bisect_right(starts, position) - 1
        return index >= 0 and position < ends[index]


def _coverage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n_rows = len(rows)
    assert n_rows > 0

    def count(column: str) -> int:
        return sum(bool(row[column]) for row in rows)

    current = count("current")
    historical = count("historical")
    both = sum(row["current"] and row["historical"] for row in rows)
    old_only = sum(not row["current"] and row["historical"] for row in rows)
    current_only = sum(row["current"] and not row["historical"] for row in rows)
    neither = n_rows - both - old_only - current_only

    counts = {
        "current": current,
        "historical": historical,
        "both": both,
        "historical_only": old_only,
        "current_only": current_only,
        "neither": neither,
        "current_preconservation_retained": count(
            "current_preconservation_retained"
        ),
        "current_ownership_lost_any": count("current_ownership_lost_any"),
    }
    historical_only_rows = [
        row for row in rows if row["historical"] and not row["current"]
    ]
    historical_only_due_conservation = sum(
        row["current_preconservation_retained"] for row in historical_only_rows
    )
    historical_only_due_ownership = sum(
        not row["current_preconservation_retained"]
        and row["current_ownership_lost_any"]
        for row in historical_only_rows
    )
    counts.update(
        {
            "historical_only_due_conservation": historical_only_due_conservation,
            "historical_only_due_ownership": historical_only_due_ownership,
            "historical_only_due_construction": (
                old_only
                - historical_only_due_conservation
                - historical_only_due_ownership
            ),
        }
    )
    for winner in ("enhancer", "tss_region", "utr3", "ncrna"):
        counts[f"current_lost_to_{winner}"] = count(f"current_lost_to_{winner}")

    return {
        "n_rows": n_rows,
        "counts": counts,
        "fractions": {key: value / n_rows for key, value in counts.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", required=True)
    parser.add_argument("--historical", required=True)
    parser.add_argument("--current-ownership", required=True)
    parser.add_argument("--vep-scores", required=True)
    args = parser.parse_args()

    current = (
        pl.scan_parquet(args.current)
        .select("source_chrom", "source_start", "source_end", "region_label")
        .filter(pl.col("region_label") == "cds")
        .collect()
    )
    historical = (
        pl.scan_parquet(args.historical)
        .select("source_chrom", "source_start", "source_end", "region_label")
        .filter(pl.col("region_label") == "cds")
        .collect()
    )
    ownership = (
        pl.scan_parquet(args.current_ownership)
        .select(
            "source_arm",
            "chrom",
            "start",
            "end",
            "passes_ownership_gate",
            "ownership_winner",
        )
        .filter(pl.col("source_arm") == "cds")
        .collect()
    )
    vep = (
        pl.scan_parquet(args.vep_scores)
        .select("chrom", "pos", "ref", "alt", "label", "subset", "match_group")
        .filter(pl.col("subset").is_in(SUBSETS))
        .collect()
    )
    assert vep["pos"].min() >= 1

    indexes = {
        "current": IntervalIndex(
            current, "source_chrom", "source_start", "source_end"
        ),
        "historical": IntervalIndex(
            historical, "source_chrom", "source_start", "source_end"
        ),
        "current_preconservation_retained": IntervalIndex(
            ownership.filter(pl.col("passes_ownership_gate")),
            "chrom",
            "start",
            "end",
        ),
        "current_ownership_lost_any": IntervalIndex(
            ownership.filter(~pl.col("passes_ownership_gate")),
            "chrom",
            "start",
            "end",
        ),
    }
    for winner in ("enhancer", "tss_region", "utr3", "ncrna"):
        indexes[f"current_lost_to_{winner}"] = IntervalIndex(
            ownership.filter(
                (~pl.col("passes_ownership_gate"))
                & (pl.col("ownership_winner") == winner)
            ),
            "chrom",
            "start",
            "end",
        )

    rows: list[dict[str, Any]] = []
    for values in vep.iter_rows(named=True):
        # Normalize the Ensembl-style benchmark and ownership names (for
        # example, "1") and UCSC-style scored-anchor names ("chr1").
        chrom = _normalized_chrom(values["chrom"])
        # Evaluation inputs use 1-based VCF-like positions. Convert at this
        # boundary before testing the 0-based, half-open anchor intervals.
        position = int(values["pos"]) - 1
        row = {
            "subset": str(values["subset"]),
            "label": int(values["label"]),
        }
        row.update(
            {
                name: index.contains(chrom, position)
                for name, index in indexes.items()
            }
        )
        rows.append(row)

    result: dict[str, Any] = {
        "coordinate_system": {
            "vep_input": "1-based position",
            "anchor_input": "0-based half-open",
            "conversion": "position_0 = pos - 1",
            "sequence_name_conversion": "VEP N -> anchor chrN",
        },
        "subsets": {},
    }
    for subset in SUBSETS:
        subset_rows = [row for row in rows if row["subset"] == subset]
        result["subsets"][subset] = {
            "all": _coverage_summary(subset_rows),
            "positive": _coverage_summary(
                [row for row in subset_rows if row["label"] == 1]
            ),
            "negative": _coverage_summary(
                [row for row in subset_rows if row["label"] == 0]
            ),
        }

    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
