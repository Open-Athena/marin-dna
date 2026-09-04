#!/usr/bin/env python3
"""Stream cross-arm anchor coverage over development Mendelian VEP loci."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import json
import sys
from typing import Any

import pyarrow.parquet as pq


ARMS = ("cds", "utr3", "tss_region", "ncrna", "enhancer")
CURRENT_LABELS = {arm: arm for arm in ARMS}
HISTORICAL_LABELS = {
    "cds": "cds",
    "utr3": "utr3",
    "tss_region": "tss_region_and_utr5",
    "ncrna": "ncrna_exon",
    "enhancer": "ccre_enhancer_centered",
}
OWNERSHIP_ARMS = ("utr3", "tss_region", "ncrna")


def _normalized_chrom(value: object) -> str:
    chrom = str(value)
    return chrom if chrom.startswith("chr") else f"chr{chrom}"


class StreamingIntervalIndex:
    """Merged 0-based, half-open intervals added in chromosome/start order."""

    def __init__(self) -> None:
        self._merged: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._last_start: dict[str, int] = {}
        self._search: dict[str, tuple[list[int], list[int]]] | None = None
        self.input_intervals = 0

    def add(self, chrom: object, start: object, end: object) -> None:
        if self._search is not None:
            raise RuntimeError("cannot add intervals after finalization")
        chrom_value = _normalized_chrom(chrom)
        start_value = int(start)
        end_value = int(end)
        if start_value < 0 or end_value <= start_value:
            raise ValueError((chrom_value, start_value, end_value))
        previous_start = self._last_start.get(chrom_value)
        if previous_start is not None and start_value < previous_start:
            raise ValueError(
                f"input is not start-sorted within {chrom_value}: "
                f"{start_value} < {previous_start}"
            )
        self._last_start[chrom_value] = start_value
        intervals = self._merged[chrom_value]
        if not intervals or start_value > intervals[-1][1]:
            intervals.append((start_value, end_value))
        else:
            intervals[-1] = (
                intervals[-1][0],
                max(intervals[-1][1], end_value),
            )
        self.input_intervals += 1

    def finalize(self) -> None:
        self._search = {
            chrom: (
                [interval[0] for interval in intervals],
                [interval[1] for interval in intervals],
            )
            for chrom, intervals in self._merged.items()
        }

    def contains(self, chrom: str, position: int) -> bool:
        if self._search is None:
            raise RuntimeError("index must be finalized before lookup")
        values = self._search.get(chrom)
        if values is None:
            return False
        starts, ends = values
        index = bisect_right(starts, position) - 1
        return index >= 0 and position < ends[index]

    def metadata(self) -> dict[str, int]:
        if self._search is None:
            raise RuntimeError("index must be finalized before metadata")
        return {
            "input_intervals": self.input_intervals,
            "merged_intervals": sum(len(values[0]) for values in self._search.values()),
            "union_bases": sum(
                end - start
                for starts, ends in self._search.values()
                for start, end in zip(starts, ends, strict=True)
            ),
        }


def _load_catalog(
    path: str,
    *,
    labels: dict[str, str],
    chrom_column: str,
    start_column: str,
    end_column: str,
    label_column: str,
) -> dict[str, StreamingIntervalIndex]:
    inverse = {label: arm for arm, label in labels.items()}
    indexes = {arm: StreamingIntervalIndex() for arm in labels}
    parquet = pq.ParquetFile(path)
    columns = [label_column, chrom_column, start_column, end_column]
    for batch in parquet.iter_batches(batch_size=65_536, columns=columns):
        values = [batch.column(index).to_pylist() for index in range(len(columns))]
        for label, chrom, start, end in zip(*values, strict=True):
            arm = inverse.get(str(label))
            if arm is not None:
                indexes[arm].add(chrom, start, end)
    for index in indexes.values():
        index.finalize()
    return indexes


def _load_ownership(path: str) -> dict[str, StreamingIntervalIndex]:
    indexes: dict[str, StreamingIntervalIndex] = {}
    for arm in OWNERSHIP_ARMS:
        for outcome in ("all", "retained", "lost"):
            indexes[f"candidate:{arm}:{outcome}"] = StreamingIntervalIndex()
        for winner in ARMS:
            if winner != arm:
                indexes[f"candidate:{arm}:lost_to:{winner}"] = StreamingIntervalIndex()

    columns = [
        "source_arm",
        "chrom",
        "start",
        "end",
        "passes_ownership_gate",
        "ownership_winner",
    ]
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=65_536, columns=columns):
        values = [batch.column(index).to_pylist() for index in range(len(columns))]
        for arm, chrom, start, end, retained, winner in zip(*values, strict=True):
            arm_value = str(arm)
            if arm_value not in OWNERSHIP_ARMS:
                continue
            indexes[f"candidate:{arm_value}:all"].add(chrom, start, end)
            outcome = "retained" if bool(retained) else "lost"
            indexes[f"candidate:{arm_value}:{outcome}"].add(chrom, start, end)
            if not retained:
                indexes[f"candidate:{arm_value}:lost_to:{winner}"].add(
                    chrom, start, end
                )
    for index in indexes.values():
        index.finalize()
    return indexes


def _joint_category(left: bool, right: bool, left_name: str, right_name: str) -> str:
    if left and right:
        return "both"
    if left:
        return f"{left_name}_only"
    if right:
        return f"{right_name}_only"
    return "neither"


def _transition(current: bool, historical: bool) -> str:
    if current and historical:
        return "both"
    if current:
        return "current_only"
    if historical:
        return "historical_only"
    return "neither"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", required=True)
    parser.add_argument("--historical", required=True)
    parser.add_argument("--ownership", required=True)
    parser.add_argument("--vep-scores", required=True)
    args = parser.parse_args()

    current = _load_catalog(
        args.current,
        labels=CURRENT_LABELS,
        chrom_column="source_chrom",
        start_column="source_start",
        end_column="source_end",
        label_column="region_label",
    )
    historical = _load_catalog(
        args.historical,
        labels=HISTORICAL_LABELS,
        chrom_column="source_chrom",
        start_column="source_start",
        end_column="source_end",
        label_column="region_label",
    )
    ownership = _load_ownership(args.ownership)
    coverage_indexes = {
        **{f"current:{arm}": index for arm, index in current.items()},
        **{f"historical:{arm}": index for arm, index in historical.items()},
        **ownership,
    }

    summaries: dict[str, dict[str, dict[str, Any]]] = defaultdict(
        lambda: defaultdict(lambda: {"n_rows": 0, "counts": Counter()})
    )
    parquet = pq.ParquetFile(args.vep_scores)
    columns = ["chrom", "pos", "label", "subset"]
    for batch in parquet.iter_batches(batch_size=65_536, columns=columns):
        values = [batch.column(index).to_pylist() for index in range(len(columns))]
        for chrom, position, label, subset in zip(*values, strict=True):
            chrom_value = _normalized_chrom(chrom)
            position_value = int(position)
            if position_value < 1:
                raise ValueError("VEP positions must be 1-based and positive")
            position_zero = position_value - 1
            label_value = int(label)
            if label_value not in {0, 1}:
                raise ValueError(f"unexpected binary label {label_value}")
            subset_value = str(subset)
            covered = {
                name: index.contains(chrom_value, position_zero)
                for name, index in coverage_indexes.items()
            }

            extra_counts: list[str] = []
            for catalog in ("current", "historical"):
                tss = covered[f"{catalog}:tss_region"]
                ncrna = covered[f"{catalog}:ncrna"]
                category = _joint_category(tss, ncrna, "tss_region", "ncrna")
                extra_counts.append(f"joint:{catalog}:tss_ncrna:{category}")
            for arm in ARMS:
                category = _transition(
                    covered[f"current:{arm}"], covered[f"historical:{arm}"]
                )
                extra_counts.append(f"transition:{arm}:{category}")

            strata = ("all", "positive" if label_value == 1 else "negative")
            for stratum in strata:
                summary = summaries[subset_value][stratum]
                summary["n_rows"] += 1
                for name, is_covered in covered.items():
                    if is_covered:
                        summary["counts"][name] += 1
                summary["counts"].update(extra_counts)

    serialized: dict[str, Any] = {}
    for subset, strata in sorted(summaries.items()):
        serialized[subset] = {}
        for stratum, summary in sorted(strata.items()):
            n_rows = int(summary["n_rows"])
            counts = dict(sorted(summary["counts"].items()))
            serialized[subset][stratum] = {
                "n_rows": n_rows,
                "counts": counts,
                "fractions": {key: value / n_rows for key, value in counts.items()},
            }

    json.dump(
        {
            "coordinate_system": {
                "vep_input": "1-based position",
                "anchor_input": "0-based half-open",
                "conversion": "position_0 = pos - 1",
                "sequence_name_conversion": "VEP N -> anchor chrN",
            },
            "catalog_metadata": {
                "current": {arm: index.metadata() for arm, index in current.items()},
                "historical": {
                    arm: index.metadata() for arm, index in historical.items()
                },
                "ownership": {
                    name: index.metadata() for name, index in ownership.items()
                },
            },
            "subsets": serialized,
        },
        sys.stdout,
        indent=2,
        sort_keys=True,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
