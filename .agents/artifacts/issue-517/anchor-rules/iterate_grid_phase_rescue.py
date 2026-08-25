#!/usr/bin/env python3
"""Evaluate additive >=20% grid-phase anchors on development VEP loci.

The current issue-517 training catalog is always retained.
Each experimental policy adds same-arm anchors from the historical uniform
128-bp grid only when their conserved-base fraction is at least 0.20.
Development labels are used only after rule construction for coverage readout.
"""

from __future__ import annotations

import argparse
import json
import sys
from array import array
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from typing import Any

import pyarrow.parquet as pq

ARMS = ("cds", "utr3", "tss_region")
HISTORICAL_LABEL_TO_ARM = {
    "cds": "cds",
    "utr3": "utr3",
    "tss_region_and_utr5": "tss_region",
}
SCOPES = (
    "nearest_grid_selected",
    "nearest_grid_pass",
    "nearest_grid_all_candidates",
    "grid_all",
    "grid_overlap",
    "grid_overlap_novel_1",
    "grid_overlap_novel_64",
    "grid_overlap_novel_128",
    "grid_any_candidate_novel_64",
    "grid_any_candidate_novel_128",
    "grid_novel_64",
    "grid_novel_128",
    "grid_center",
    "grid_contained",
)
HOME_SUBSETS = {
    "cds": ("missense_variant", "splicing", "synonymous_variant"),
    "utr3": ("3_prime_UTR_variant",),
    "tss_region": ("5_prime_UTR_variant", "tss_proximal"),
}
DEVELOPMENT_SUBSETS = (
    "missense_variant",
    "splicing",
    "synonymous_variant",
    "5_prime_UTR_variant",
    "tss_proximal",
    "3_prime_UTR_variant",
    "distal",
    "non_coding_transcript_exon_variant",
)
WINDOW_SIZE = 255
GRID_STEP = 128
TRAINING_MIN_CONSERVED = 0.20


def _normalized_chrom(value: object) -> str:
    chrom = str(value)
    return chrom if chrom.startswith("chr") else f"chr{chrom}"


class StreamingIntervalIndex:
    """Merged 0-based half-open intervals added in chromosome/start order."""

    def __init__(self) -> None:
        self._merged: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._last_start: dict[str, int] = {}
        self._search: dict[str, tuple[list[int], list[int]]] | None = None

    def add(self, chrom: object, start: object, end: object) -> None:
        if self._search is not None:
            raise RuntimeError("cannot add intervals after finalization")
        chrom_value = _normalized_chrom(chrom)
        start_value = int(start)
        end_value = int(end)
        if start_value < 0 or end_value - start_value != WINDOW_SIZE:
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

    def finalize(self) -> None:
        self._search = {
            chrom: (
                [interval[0] for interval in intervals],
                [interval[1] for interval in intervals],
            )
            for chrom, intervals in self._merged.items()
        }

    def _values(self, chrom: object) -> tuple[list[int], list[int]] | None:
        if self._search is None:
            raise RuntimeError("index must be finalized before lookup")
        return self._search.get(_normalized_chrom(chrom))

    def contains_point(self, chrom: object, position: int) -> bool:
        values = self._values(chrom)
        if values is None:
            return False
        starts, ends = values
        index = bisect_right(starts, position) - 1
        return index >= 0 and position < ends[index]

    def overlaps(self, chrom: object, start: int, end: int) -> bool:
        values = self._values(chrom)
        if values is None:
            return False
        starts, ends = values
        index = bisect_right(starts, end - 1) - 1
        return index >= 0 and ends[index] > start

    def contains_interval(self, chrom: object, start: int, end: int) -> bool:
        values = self._values(chrom)
        if values is None:
            return False
        starts, ends = values
        index = bisect_right(starts, start) - 1
        return index >= 0 and ends[index] >= end

    def overlap_bases(self, chrom: object, start: int, end: int) -> int:
        values = self._values(chrom)
        if values is None:
            return 0
        starts, ends = values
        index = max(0, bisect_right(starts, start) - 1)
        total = 0
        while index < len(starts) and starts[index] < end:
            total += max(0, min(end, ends[index]) - max(start, starts[index]))
            index += 1
        return total

    def metadata(self) -> dict[str, int]:
        if self._search is None:
            raise RuntimeError("index must be finalized before metadata")
        return {
            "merged_intervals": sum(len(starts) for starts, _ in self._search.values()),
            "union_bases": sum(
                end - start
                for starts, ends in self._search.values()
                for start, end in zip(starts, ends, strict=True)
            ),
        }


class CandidateStartIndex:
    """Compact current-candidate starts for nearest-grid membership tests."""

    def __init__(self) -> None:
        self._values: dict[
            tuple[str, str], tuple[array[int], array[int], array[int]]
        ] = {}
        self._last_start: dict[tuple[str, str], int] = {}

    def add(
        self,
        arm: str,
        chrom: object,
        start: object,
        *,
        passes_ownership: bool,
        selected: bool,
    ) -> None:
        chrom_value = _normalized_chrom(chrom)
        start_value = int(start)
        key = (arm, chrom_value)
        previous = self._last_start.get(key)
        if previous is not None and start_value <= previous:
            raise ValueError(
                f"candidate starts are not unique/sorted for {key}: "
                f"{start_value} <= {previous}"
            )
        self._last_start[key] = start_value
        starts, passes, selected_values = self._values.setdefault(
            key, (array("q"), array("B"), array("B"))
        )
        starts.append(start_value)
        passes.append(passes_ownership)
        selected_values.append(selected)

    def has_nearest_grid_source(
        self,
        arm: str,
        chrom: object,
        grid_start: int,
        *,
        scope: str,
    ) -> bool:
        values = self._values.get((arm, _normalized_chrom(chrom)))
        if values is None:
            return False
        starts, passes, selected = values
        # With ties assigned to the lower grid point, candidate starts in this
        # inclusive range map to grid_start as their nearest 128-bp grid start.
        left = bisect_left(starts, grid_start - 63)
        right = bisect_right(starts, grid_start + 64)
        if scope == "nearest_grid_all_candidates":
            return left < right
        flags = passes if scope == "nearest_grid_pass" else selected
        return any(flags[index] for index in range(left, right))

    def overlaps_candidate(self, arm: str, chrom: object, start: int, end: int) -> bool:
        values = self._values.get((arm, _normalized_chrom(chrom)))
        if values is None:
            return False
        starts, _, _ = values
        left = bisect_left(starts, start - WINDOW_SIZE + 1)
        right = bisect_left(starts, end)
        return left < right


def _load_current(
    path: str,
) -> tuple[
    dict[str, StreamingIntervalIndex],
    dict[str, set[tuple[str, int, int]]],
    Counter[str],
]:
    indexes = {arm: StreamingIntervalIndex() for arm in ARMS}
    coordinates: dict[str, set[tuple[str, int, int]]] = {arm: set() for arm in ARMS}
    counts: Counter[str] = Counter()
    columns = ["region_label", "source_chrom", "source_start", "source_end"]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=65_536, columns=columns):
        values = [batch.column(index).to_pylist() for index in range(len(columns))]
        for arm, chrom, start, end in zip(*values, strict=True):
            arm_value = str(arm)
            if arm_value not in ARMS:
                continue
            chrom_value = _normalized_chrom(chrom)
            start_value = int(start)
            end_value = int(end)
            indexes[arm_value].add(chrom_value, start_value, end_value)
            coordinates[arm_value].add((chrom_value, start_value, end_value))
            counts[arm_value] += 1
    for arm, index in indexes.items():
        index.finalize()
        if counts[arm] != len(coordinates[arm]):
            raise AssertionError(
                f"current {arm} catalog contains duplicate coordinates"
            )
    return indexes, coordinates, counts


def _load_candidate_territories(
    path: str,
    *,
    current_coordinates: dict[str, set[tuple[str, int, int]]],
) -> tuple[dict[str, StreamingIntervalIndex], CandidateStartIndex]:
    indexes = {arm: StreamingIntervalIndex() for arm in ARMS}
    starts = CandidateStartIndex()
    columns = ["source_arm", "chrom", "start", "end", "passes_ownership_gate"]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=65_536, columns=columns):
        values = [batch.column(index).to_pylist() for index in range(len(columns))]
        for arm, chrom, start, end, retained in zip(*values, strict=True):
            arm_value = str(arm)
            if arm_value not in ARMS:
                continue
            chrom_value = _normalized_chrom(chrom)
            start_value = int(start)
            end_value = int(end)
            retained_value = bool(retained)
            if retained_value:
                indexes[arm_value].add(chrom_value, start_value, end_value)
            starts.add(
                arm_value,
                chrom_value,
                start_value,
                passes_ownership=retained_value,
                selected=(chrom_value, start_value, end_value)
                in current_coordinates[arm_value],
            )
    for index in indexes.values():
        index.finalize()
    return indexes, starts


def _scope_membership(
    territory: StreamingIntervalIndex,
    current: StreamingIntervalIndex,
    candidates: CandidateStartIndex,
    arm: str,
    chrom: str,
    start: int,
    end: int,
) -> dict[str, bool]:
    center = start + (end - start) // 2
    overlaps_territory = territory.overlaps(chrom, start, end)
    overlaps_any_candidate = candidates.overlaps_candidate(arm, chrom, start, end)
    novel_bases = WINDOW_SIZE - current.overlap_bases(chrom, start, end)
    return {
        scope: candidates.has_nearest_grid_source(arm, chrom, start, scope=scope)
        for scope in (
            "nearest_grid_selected",
            "nearest_grid_pass",
            "nearest_grid_all_candidates",
        )
    } | {
        "grid_all": True,
        "grid_overlap": overlaps_territory,
        "grid_overlap_novel_1": overlaps_territory and novel_bases >= 1,
        "grid_overlap_novel_64": overlaps_territory and novel_bases >= 64,
        "grid_overlap_novel_128": overlaps_territory and novel_bases >= 128,
        "grid_any_candidate_novel_64": (overlaps_any_candidate and novel_bases >= 64),
        "grid_any_candidate_novel_128": (overlaps_any_candidate and novel_bases >= 128),
        "grid_novel_64": novel_bases >= 64,
        "grid_novel_128": novel_bases >= 128,
        "grid_center": territory.contains_point(chrom, center),
        "grid_contained": territory.contains_interval(chrom, start, end),
    }


def _load_rescue_rules(
    path: str,
    *,
    current_coordinates: dict[str, set[tuple[str, int, int]]],
    current_indexes: dict[str, StreamingIntervalIndex],
    territories: dict[str, StreamingIntervalIndex],
    candidates: CandidateStartIndex,
) -> tuple[
    dict[str, dict[str, StreamingIntervalIndex]],
    dict[str, dict[str, dict[str, int | float]]],
]:
    indexes = {
        arm: {scope: StreamingIntervalIndex() for scope in SCOPES} for arm in ARMS
    }
    stats: dict[str, dict[str, dict[str, int | float]]] = {
        arm: {
            scope: {
                "new_anchor_count": 0,
                "min_proportion_conserved": 1.0,
                "max_proportion_conserved": 0.0,
            }
            for scope in SCOPES
        }
        for arm in ARMS
    }
    columns = [
        "region_label",
        "source_chrom",
        "source_start",
        "source_end",
        "proportion_conserved",
    ]
    skipped_below_training_min = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=65_536, columns=columns):
        values = [batch.column(index).to_pylist() for index in range(len(columns))]
        for label, chrom, start, end, conserved in zip(*values, strict=True):
            arm = HISTORICAL_LABEL_TO_ARM.get(str(label))
            if arm is None:
                continue
            chrom_value = _normalized_chrom(chrom)
            start_value = int(start)
            end_value = int(end)
            conserved_value = float(conserved)
            if end_value - start_value != WINDOW_SIZE:
                raise AssertionError("historical anchor is not 255 bp")
            if start_value % GRID_STEP != 0:
                raise AssertionError(
                    "historical functional anchor is off the 128-bp grid"
                )
            if conserved_value + 1e-12 < TRAINING_MIN_CONSERVED:
                skipped_below_training_min += 1
                continue
            coordinate = (chrom_value, start_value, end_value)
            if coordinate in current_coordinates[arm]:
                continue
            membership = _scope_membership(
                territories[arm],
                current_indexes[arm],
                candidates,
                arm,
                chrom_value,
                start_value,
                end_value,
            )
            for scope, keep in membership.items():
                if not keep:
                    continue
                indexes[arm][scope].add(chrom_value, start_value, end_value)
                rule_stats = stats[arm][scope]
                rule_stats["new_anchor_count"] = int(rule_stats["new_anchor_count"]) + 1
                rule_stats["min_proportion_conserved"] = min(
                    float(rule_stats["min_proportion_conserved"]), conserved_value
                )
                rule_stats["max_proportion_conserved"] = max(
                    float(rule_stats["max_proportion_conserved"]), conserved_value
                )
    if skipped_below_training_min:
        raise AssertionError(
            f"historical training catalog had {skipped_below_training_min} anchors below 0.20"
        )
    for arm in ARMS:
        for scope in SCOPES:
            indexes[arm][scope].finalize()
            if int(stats[arm][scope]["new_anchor_count"]) == 0:
                raise AssertionError(f"empty rescue rule {arm}/{scope}")
    return indexes, stats


def _read_development_rows(path: str) -> list[dict[str, Any]]:
    columns = ["chrom", "pos", "label", "subset", "match_group"]
    table = pq.read_table(path, columns=columns)
    values = {name: table[name].to_pylist() for name in columns}
    excluded_groups = {
        int(values["match_group"][index])
        for index in range(table.num_rows)
        if values["subset"][index] == "mature_miRNA_variant"
    }
    rows: list[dict[str, Any]] = []
    allowed_chroms = {f"chr{chrom}" for chrom in range(1, 23, 2)} | {"chrX"}
    for index in range(table.num_rows):
        if int(values["match_group"][index]) in excluded_groups:
            continue
        subset = str(values["subset"][index])
        if subset not in DEVELOPMENT_SUBSETS:
            raise AssertionError(f"unexpected development subset {subset}")
        chrom = _normalized_chrom(values["chrom"][index])
        if chrom not in allowed_chroms:
            raise AssertionError(f"held-out labeled chromosome was read: {chrom}")
        position = int(values["pos"][index])
        if position < 1:
            raise ValueError("development VEP positions must be 1-based positive")
        label = int(values["label"][index])
        if label not in {0, 1}:
            raise ValueError(f"unexpected binary label {label}")
        rows.append(
            {
                "chrom": chrom,
                "position_zero": position - 1,
                "label": label,
                "subset": subset,
            }
        )
    if len(rows) != 16_100:
        raise AssertionError(f"unexpected post-exclusion row count {len(rows)}")
    return rows


def _empty_coverage() -> dict[str, dict[str, Counter[str]]]:
    return {
        subset: {
            "all": Counter(),
            "positive": Counter(),
            "negative": Counter(),
        }
        for subset in DEVELOPMENT_SUBSETS
    }


def _coverage_summary(
    rows: list[dict[str, Any]],
    *,
    current: dict[str, StreamingIntervalIndex],
    rescue: dict[str, dict[str, StreamingIntervalIndex]],
) -> dict[str, dict[str, Any]]:
    counts = {arm: _empty_coverage() for arm in ARMS}
    for row in rows:
        subset = str(row["subset"])
        stratum = "positive" if int(row["label"]) == 1 else "negative"
        chrom = str(row["chrom"])
        position = int(row["position_zero"])
        for arm in ARMS:
            current_covered = current[arm].contains_point(chrom, position)
            policies = {"current": current_covered}
            for scope in SCOPES:
                policies[scope] = current_covered or rescue[arm][scope].contains_point(
                    chrom, position
                )
            for policy, covered in policies.items():
                counts[arm][subset]["all"][(policy, "rows")] += 1
                counts[arm][subset][stratum][(policy, "rows")] += 1
                if covered:
                    counts[arm][subset]["all"][(policy, "covered")] += 1
                    counts[arm][subset][stratum][(policy, "covered")] += 1

    output: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        output[arm] = {}
        for subset in DEVELOPMENT_SUBSETS:
            output[arm][subset] = {}
            for stratum in ("all", "positive", "negative"):
                counter = counts[arm][subset][stratum]
                current_rows = counter[("current", "rows")]
                current_covered = counter[("current", "covered")]
                policies: dict[str, Any] = {}
                for policy in ("current", *SCOPES):
                    rows_count = counter[(policy, "rows")]
                    covered = counter[(policy, "covered")]
                    if rows_count != current_rows:
                        raise AssertionError("policy support differs from baseline")
                    policies[policy] = {
                        "n_rows": rows_count,
                        "covered": covered,
                        "fraction": covered / rows_count,
                        "gain_count_vs_current": covered - current_covered,
                        "gain_fraction_vs_current": (covered - current_covered)
                        / rows_count,
                    }
                output[arm][subset][stratum] = policies
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", required=True)
    parser.add_argument("--historical-grid", required=True)
    parser.add_argument("--current-ownership", required=True)
    parser.add_argument("--vep-scores", required=True)
    args = parser.parse_args()

    current, current_coordinates, current_counts = _load_current(args.current)
    territories, candidates = _load_candidate_territories(
        args.current_ownership,
        current_coordinates=current_coordinates,
    )
    rescue, rule_stats = _load_rescue_rules(
        args.historical_grid,
        current_coordinates=current_coordinates,
        current_indexes=current,
        territories=territories,
        candidates=candidates,
    )
    rows = _read_development_rows(args.vep_scores)
    coverage = _coverage_summary(rows, current=current, rescue=rescue)

    for arm in ARMS:
        for scope in SCOPES:
            rule_stats[arm][scope]["union_anchor_count"] = current_counts[arm] + int(
                rule_stats[arm][scope]["new_anchor_count"]
            )
            rule_stats[arm][scope]["rescue_footprint"] = rescue[arm][scope].metadata()

    json.dump(
        {
            "evaluation_boundary": {
                "dataset": "Mendelian development split only",
                "excluded_complete_groups": "mature_miRNA_variant",
                "allowed_chromosomes": "odd autosomes and X",
                "heldout_accessed": False,
                "coordinate_conversion": "1-based pos -> 0-based pos - 1",
            },
            "rule_contract": {
                "composition": "current training anchors union additive rescue",
                "conservation_minimum": TRAINING_MIN_CONSERVED,
                "historical_grid_step": GRID_STEP,
                "window_size": WINDOW_SIZE,
                "development_labels_used_to_construct_rules": False,
                "scopes": {
                    "nearest_grid_selected": (
                        "historical >=20% grid anchor is the nearest phase-0 grid "
                        "coordinate of at least one current selected same-arm anchor"
                    ),
                    "nearest_grid_pass": (
                        "historical >=20% grid anchor is the nearest phase-0 grid "
                        "coordinate of at least one ownership-passing current same-arm candidate"
                    ),
                    "nearest_grid_all_candidates": (
                        "historical >=20% grid anchor is the nearest phase-0 grid "
                        "coordinate of at least one current same-arm candidate, including ownership losses"
                    ),
                    "grid_all": "all same-arm historical >=20% grid anchors",
                    "grid_overlap": "grid anchor overlaps current ownership-passing same-arm candidate territory",
                    "grid_overlap_novel_1": (
                        "grid_overlap and anchor contributes at least 1 base outside the current same-arm training union"
                    ),
                    "grid_overlap_novel_64": (
                        "grid_overlap and anchor contributes at least 64 bases outside the current same-arm training union"
                    ),
                    "grid_overlap_novel_128": (
                        "grid_overlap and anchor contributes at least 128 bases outside the current same-arm training union"
                    ),
                    "grid_any_candidate_novel_64": (
                        "grid anchor overlaps any current same-arm candidate, including ownership losses, and contributes at least 64 new bases"
                    ),
                    "grid_any_candidate_novel_128": (
                        "grid anchor overlaps any current same-arm candidate, including ownership losses, and contributes at least 128 new bases"
                    ),
                    "grid_novel_64": (
                        "same-arm historical >=20% grid anchor contributes at least 64 bases outside the current same-arm training union"
                    ),
                    "grid_novel_128": (
                        "same-arm historical >=20% grid anchor contributes at least 128 bases outside the current same-arm training union"
                    ),
                    "grid_center": "grid anchor center lies in current ownership-passing same-arm candidate territory",
                    "grid_contained": "grid anchor is contained in current ownership-passing same-arm candidate territory",
                },
            },
            "current_catalog": {
                arm: {
                    "anchor_count": current_counts[arm],
                    **current[arm].metadata(),
                }
                for arm in ARMS
            },
            "rule_stats": rule_stats,
            "coverage": coverage,
            "home_subsets": HOME_SUBSETS,
        },
        sys.stdout,
        indent=2,
        sort_keys=True,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
