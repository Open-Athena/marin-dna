#!/usr/bin/env python3
"""Trace historical >=20% functional anchors through issue-517 selection.

The diagnostic distinguishes exact-coordinate construction, ownership, and
conservation losses before inspecting coverage of pathogenic development loci.
All anchor coordinates are 0-based, half-open. VEP positions are converted
from 1-based coordinates at the input boundary.
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
HOME_SUBSETS = {
    "cds": ("missense_variant", "splicing", "synonymous_variant"),
    "utr3": ("3_prime_UTR_variant",),
    "tss_region": ("5_prime_UTR_variant", "tss_proximal"),
}
DEVELOPMENT_SUBSETS = {
    subset for subsets in HOME_SUBSETS.values() for subset in subsets
} | {"distal", "non_coding_transcript_exon_variant"}
WINDOW_SIZE = 255
GRID_STEP = 128
TRAINING_MIN = 0.20
PROJECTION_MIN = 0.10

STATUS_NAMES = (
    "selected_ge_0.20",
    "projection_0.10_to_0.20",
    "conservation_below_0.10",
    "lost_ownership_to_cds",
    "lost_ownership_to_utr3",
    "lost_ownership_to_tss_region",
    "lost_ownership_to_ncrna",
    "lost_ownership_to_enhancer",
    "lost_ownership_to_other",
)
STATUS_TO_CODE = {name: index for index, name in enumerate(STATUS_NAMES)}


def _normalized_chrom(value: object) -> str:
    chrom = str(value)
    return chrom if chrom.startswith("chr") else f"chr{chrom}"


class StreamingIntervalIndex:
    """Merged intervals added in chromosome/start order."""

    def __init__(self) -> None:
        self._merged: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._last_start: dict[str, int] = {}
        self._search: dict[str, tuple[list[int], list[int]]] | None = None

    def add(self, chrom: object, start: object, end: object) -> None:
        if self._search is not None:
            raise RuntimeError("cannot add after finalization")
        chrom_value = _normalized_chrom(chrom)
        start_value = int(start)
        end_value = int(end)
        if start_value < 0 or end_value - start_value != WINDOW_SIZE:
            raise ValueError((chrom_value, start_value, end_value))
        previous = self._last_start.get(chrom_value)
        if previous is not None and start_value < previous:
            raise ValueError(
                f"input is not start-sorted within {chrom_value}: "
                f"{start_value} < {previous}"
            )
        self._last_start[chrom_value] = start_value
        intervals = self._merged[chrom_value]
        if not intervals or start_value > intervals[-1][1]:
            intervals.append((start_value, end_value))
        else:
            intervals[-1] = (intervals[-1][0], max(intervals[-1][1], end_value))

    def finalize(self) -> None:
        self._search = {
            chrom: (
                [interval[0] for interval in intervals],
                [interval[1] for interval in intervals],
            )
            for chrom, intervals in self._merged.items()
        }

    def contains(self, chrom: object, position: int) -> bool:
        if self._search is None:
            raise RuntimeError("index is not finalized")
        values = self._search.get(_normalized_chrom(chrom))
        if values is None:
            return False
        starts, ends = values
        index = bisect_right(starts, position) - 1
        return index >= 0 and position < ends[index]


class CandidateIndex:
    """Compact index of current candidates and their final selection stage."""

    def __init__(self) -> None:
        self._values: dict[tuple[str, str], tuple[array[int], array[int]]] = {}
        self._last_start: dict[tuple[str, str], int] = {}
        self.counts: Counter[tuple[str, str]] = Counter()

    def add(self, arm: str, chrom: object, start: object, status: str) -> None:
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
        starts, statuses = self._values.setdefault(key, (array("q"), array("B")))
        starts.append(start_value)
        statuses.append(STATUS_TO_CODE[status])
        self.counts[(arm, status)] += 1

    def exact_status(self, arm: str, chrom: object, start: int) -> str | None:
        values = self._values.get((arm, _normalized_chrom(chrom)))
        if values is None:
            return None
        starts, statuses = values
        index = bisect_left(starts, start)
        if index == len(starts) or starts[index] != start:
            return None
        return STATUS_NAMES[statuses[index]]

    def nearest(self, arm: str, chrom: object, start: int) -> dict[str, Any] | None:
        values = self._values.get((arm, _normalized_chrom(chrom)))
        if values is None:
            return None
        starts, statuses = values
        index = bisect_left(starts, start)
        candidates = range(max(0, index - 1), min(len(starts), index + 1))
        best = min(candidates, key=lambda item: (abs(starts[item] - start), item))
        delta = int(starts[best]) - start
        return {
            "start_delta_bp": delta,
            "absolute_start_delta_bp": abs(delta),
            "status": STATUS_NAMES[statuses[best]],
        }

    def covering_statuses(self, arm: str, chrom: object, position: int) -> Counter[str]:
        values = self._values.get((arm, _normalized_chrom(chrom)))
        if values is None:
            return Counter()
        starts, statuses = values
        left = bisect_left(starts, position - WINDOW_SIZE + 1)
        right = bisect_right(starts, position)
        return Counter(STATUS_NAMES[statuses[index]] for index in range(left, right))

    def overlapping_statuses(
        self, arm: str, chrom: object, start: int, end: int
    ) -> Counter[str]:
        values = self._values.get((arm, _normalized_chrom(chrom)))
        if values is None:
            return Counter()
        starts, statuses = values
        left = bisect_left(starts, start - WINDOW_SIZE + 1)
        right = bisect_left(starts, end)
        return Counter(STATUS_NAMES[statuses[index]] for index in range(left, right))


def _load_catalog_starts(
    path: str,
) -> tuple[
    dict[str, dict[str, set[int]]],
    dict[str, StreamingIntervalIndex],
    Counter[tuple[str, int]],
]:
    starts: dict[str, dict[str, set[int]]] = {arm: defaultdict(set) for arm in ARMS}
    indexes = {arm: StreamingIntervalIndex() for arm in ARMS}
    phases: Counter[tuple[str, int]] = Counter()
    columns = ["region_label", "source_chrom", "source_start", "source_end"]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=65_536, columns=columns):
        values = [batch.column(index).to_pylist() for index in range(len(columns))]
        for label, chrom, start, end in zip(*values, strict=True):
            arm = str(label)
            if arm not in ARMS:
                continue
            chrom_value = _normalized_chrom(chrom)
            start_value = int(start)
            end_value = int(end)
            if start_value in starts[arm][chrom_value]:
                raise AssertionError(
                    f"duplicate catalog coordinate {arm}/{chrom_value}/{start_value}"
                )
            starts[arm][chrom_value].add(start_value)
            indexes[arm].add(chrom_value, start_value, end_value)
            phases[(arm, start_value % GRID_STEP)] += 1
    for index in indexes.values():
        index.finalize()
    return starts, indexes, phases


def _load_candidates(
    path: str,
    *,
    training: dict[str, dict[str, set[int]]],
    projection: dict[str, dict[str, set[int]]],
) -> CandidateIndex:
    index = CandidateIndex()
    columns = [
        "source_arm",
        "chrom",
        "start",
        "end",
        "passes_ownership_gate",
        "ownership_winner",
    ]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=65_536, columns=columns):
        values = {column: batch.column(column).to_pylist() for column in columns}
        for row in range(batch.num_rows):
            arm = str(values["source_arm"][row])
            if arm not in ARMS:
                continue
            chrom = _normalized_chrom(values["chrom"][row])
            start = int(values["start"][row])
            end = int(values["end"][row])
            if end - start != WINDOW_SIZE:
                raise AssertionError("candidate is not 255 bp")
            if bool(values["passes_ownership_gate"][row]):
                if start in training[arm].get(chrom, set()):
                    status = "selected_ge_0.20"
                elif start in projection[arm].get(chrom, set()):
                    status = "projection_0.10_to_0.20"
                else:
                    status = "conservation_below_0.10"
            else:
                winner = str(values["ownership_winner"][row])
                status = f"lost_ownership_to_{winner}"
                if status not in STATUS_TO_CODE:
                    status = "lost_ownership_to_other"
            index.add(arm, chrom, start, status)
    return index


def _read_development_positives(
    path: str,
    *,
    current: dict[str, StreamingIntervalIndex],
) -> list[dict[str, Any]]:
    columns = ["chrom", "pos", "label", "subset", "match_group"]
    table = pq.read_table(path, columns=columns)
    values = {column: table[column].to_pylist() for column in columns}
    excluded_groups = {
        int(values["match_group"][row])
        for row in range(table.num_rows)
        if values["subset"][row] == "mature_miRNA_variant"
    }
    allowed_chroms = {f"chr{chrom}" for chrom in range(1, 23, 2)} | {"chrX"}
    retained_rows = 0
    positives: list[dict[str, Any]] = []
    subset_to_arm = {
        subset: arm for arm, subsets in HOME_SUBSETS.items() for subset in subsets
    }
    for row in range(table.num_rows):
        if int(values["match_group"][row]) in excluded_groups:
            continue
        retained_rows += 1
        subset = str(values["subset"][row])
        if subset not in DEVELOPMENT_SUBSETS:
            raise AssertionError(f"unexpected development subset {subset}")
        chrom = _normalized_chrom(values["chrom"][row])
        if chrom not in allowed_chroms:
            raise AssertionError(f"held-out labeled chromosome was read: {chrom}")
        position = int(values["pos"][row])
        if position < 1:
            raise ValueError("VEP positions must be 1-based positive")
        if not bool(values["label"][row]) or subset not in subset_to_arm:
            continue
        arm = subset_to_arm[subset]
        position_zero = position - 1
        positives.append(
            {
                "arm": arm,
                "subset": subset,
                "chrom": chrom,
                "position_zero": position_zero,
                "current_covered": current[arm].contains(chrom, position_zero),
                "historical_anchors": [],
            }
        )
    if retained_rows != 16_100:
        raise AssertionError(f"unexpected post-exclusion row count {retained_rows}")
    return positives


def _weighted_distribution(counts: Counter[int]) -> dict[str, Any]:
    total = sum(counts.values())
    if total == 0:
        return {"n": 0}
    ordered = sorted(counts.items())
    quantiles: dict[str, int] = {}
    for quantile in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0):
        target = max(1, int(quantile * (total - 1)) + 1)
        cumulative = 0
        for value, count in ordered:
            cumulative += count
            if cumulative >= target:
                quantiles[str(quantile)] = value
                break
    return {
        "n": total,
        "mean": sum(value * count for value, count in ordered) / total,
        "quantiles": quantiles,
        "fraction_at_or_below_bp": {
            str(threshold): sum(count for value, count in ordered if value <= threshold)
            / total
            for threshold in (0, 1, 32, 64, 127, 128, 254, 255, 512, 1024)
        },
    }


def _locus_loss_reason(statuses: Counter[str]) -> str:
    if statuses["selected_ge_0.20"]:
        raise AssertionError(
            "historical-only locus is covered by a current selected anchor"
        )
    if statuses["projection_0.10_to_0.20"]:
        return "current_phase_conservation_0.10_to_0.20"
    if statuses["conservation_below_0.10"]:
        return "current_phase_conservation_below_0.10"
    ownership = sorted(
        status for status in statuses if status.startswith("lost_ownership_to_")
    )
    if ownership:
        return "+".join(ownership)
    return "no_current_candidate_covers_locus"


def _historical_anchor_context(statuses: Counter[str]) -> str:
    passing = (
        statuses["selected_ge_0.20"]
        + statuses["projection_0.10_to_0.20"]
        + statuses["conservation_below_0.10"]
    )
    if passing:
        return "overlaps_ownership_passing_current_candidate_territory"
    if any(status.startswith("lost_ownership_to_") for status in statuses):
        return "overlaps_only_ownership_lost_current_candidate_territory"
    return "no_current_same_arm_candidate_overlaps_historical_anchor"


def _serialize_phase_counts(counts: Counter[tuple[str, int]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for arm in ARMS:
        arm_counts = {str(phase): counts[(arm, phase)] for phase in range(GRID_STEP)}
        nonzero = {phase: count for phase, count in arm_counts.items() if count}
        output[arm] = {
            "n_occupied_phases": len(nonzero),
            "counts": nonzero,
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", required=True)
    parser.add_argument("--projection", required=True)
    parser.add_argument("--current-ownership", required=True)
    parser.add_argument("--historical", required=True)
    parser.add_argument("--vep-scores", required=True)
    args = parser.parse_args()

    training, current_indexes, current_phases = _load_catalog_starts(args.current)
    projection, _, _ = _load_catalog_starts(args.projection)
    candidates = _load_candidates(
        args.current_ownership,
        training=training,
        projection=projection,
    )
    positives = _read_development_positives(
        args.vep_scores,
        current=current_indexes,
    )
    positive_positions: dict[tuple[str, str], tuple[list[int], list[int]]] = {}
    position_rows: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for row_index, row in enumerate(positives):
        position_rows[(str(row["arm"]), str(row["chrom"]))].append(
            (int(row["position_zero"]), row_index)
        )
    for key, rows in position_rows.items():
        rows.sort()
        positive_positions[key] = (
            [position for position, _ in rows],
            [row_index for _, row_index in rows],
        )

    exact_status_counts: Counter[tuple[str, str]] = Counter()
    nearest_distance_counts: dict[str, Counter[int]] = {arm: Counter() for arm in ARMS}
    historical_phases: Counter[tuple[str, int]] = Counter()
    historical_counts: Counter[str] = Counter()
    columns = [
        "region_label",
        "source_chrom",
        "source_start",
        "source_end",
        "proportion_conserved",
    ]
    for batch in pq.ParquetFile(args.historical).iter_batches(
        batch_size=65_536, columns=columns
    ):
        values = {column: batch.column(column).to_pylist() for column in columns}
        for row in range(batch.num_rows):
            arm = HISTORICAL_LABEL_TO_ARM.get(str(values["region_label"][row]))
            if arm is None:
                continue
            chrom = _normalized_chrom(values["source_chrom"][row])
            start = int(values["source_start"][row])
            end = int(values["source_end"][row])
            conserved = float(values["proportion_conserved"][row])
            if end - start != WINDOW_SIZE or start % GRID_STEP != 0:
                raise AssertionError(
                    "historical functional anchor violates grid contract"
                )
            if conserved + 1e-12 < TRAINING_MIN:
                raise AssertionError("historical catalog contains an anchor below 0.20")
            historical_counts[arm] += 1
            historical_phases[(arm, start % GRID_STEP)] += 1
            exact_status = candidates.exact_status(arm, chrom, start)
            exact_status_counts[(arm, exact_status or "not_constructed")] += 1
            nearest = candidates.nearest(arm, chrom, start)
            if nearest is not None:
                nearest_distance_counts[arm][
                    int(nearest["absolute_start_delta_bp"])
                ] += 1
            locus_values = positive_positions.get((arm, chrom))
            if locus_values is None:
                continue
            positions, row_indexes = locus_values
            left = bisect_left(positions, start)
            right = bisect_left(positions, end)
            for match in range(left, right):
                positives[row_indexes[match]]["historical_anchors"].append(
                    {
                        "start": start,
                        "conserved": conserved,
                        "exact_status": exact_status or "not_constructed",
                    }
                )

    subset_summaries: dict[str, Any] = {}
    old_only_anchor_keys: set[tuple[str, str, int]] = set()
    old_only_anchor_statuses: Counter[tuple[str, str]] = Counter()
    for arm, subsets in HOME_SUBSETS.items():
        for subset in subsets:
            rows = [row for row in positives if row["subset"] == subset]
            transitions: Counter[str] = Counter()
            reasons: Counter[str] = Counter()
            anchor_contexts: Counter[str] = Counter()
            reason_by_anchor_context: Counter[str] = Counter()
            locus_has_status: Counter[str] = Counter()
            historical_anchor_counts: Counter[int] = Counter()
            historical_anchor_conservation_bins: Counter[str] = Counter()
            for row in rows:
                current_covered = bool(row["current_covered"])
                historical_covered = bool(row["historical_anchors"])
                if current_covered and historical_covered:
                    transition = "both"
                elif current_covered:
                    transition = "current_only"
                elif historical_covered:
                    transition = "historical_only"
                else:
                    transition = "neither"
                transitions[transition] += 1
                if transition != "historical_only":
                    continue
                statuses = candidates.covering_statuses(
                    arm, str(row["chrom"]), int(row["position_zero"])
                )
                reason = _locus_loss_reason(statuses)
                reasons[reason] += 1
                for status in statuses:
                    locus_has_status[status] += 1
                overlapping_statuses: Counter[str] = Counter()
                for anchor in row["historical_anchors"]:
                    overlapping_statuses.update(
                        candidates.overlapping_statuses(
                            arm,
                            str(row["chrom"]),
                            int(anchor["start"]),
                            int(anchor["start"]) + WINDOW_SIZE,
                        )
                    )
                anchor_context = _historical_anchor_context(overlapping_statuses)
                anchor_contexts[anchor_context] += 1
                reason_by_anchor_context[f"{reason}|{anchor_context}"] += 1
                historical_anchor_counts[len(row["historical_anchors"])] += 1
                for anchor in row["historical_anchors"]:
                    conserved = float(anchor["conserved"])
                    if conserved < 0.25:
                        conserved_bin = "0.20_to_0.25"
                    elif conserved < 0.30:
                        conserved_bin = "0.25_to_0.30"
                    elif conserved < 0.50:
                        conserved_bin = "0.30_to_0.50"
                    else:
                        conserved_bin = "ge_0.50"
                    historical_anchor_conservation_bins[conserved_bin] += 1
                    key = (arm, str(row["chrom"]), int(anchor["start"]))
                    if key not in old_only_anchor_keys:
                        old_only_anchor_keys.add(key)
                        old_only_anchor_statuses[
                            (arm, str(anchor["exact_status"]))
                        ] += 1
            subset_summaries[subset] = {
                "arm": arm,
                "n_pathogenic": len(rows),
                "transitions": dict(sorted(transitions.items())),
                "historical_only_locus_reason": dict(sorted(reasons.items())),
                "historical_only_historical_anchor_context": dict(
                    sorted(anchor_contexts.items())
                ),
                "historical_only_reason_by_historical_anchor_context": dict(
                    sorted(reason_by_anchor_context.items())
                ),
                "historical_only_locus_has_candidate_status": dict(
                    sorted(locus_has_status.items())
                ),
                "historical_only_number_of_covering_historical_anchors": dict(
                    sorted(historical_anchor_counts.items())
                ),
                "historical_only_covering_anchor_conservation_bins": dict(
                    sorted(historical_anchor_conservation_bins.items())
                ),
            }

    global_status: dict[str, Any] = {}
    for arm in ARMS:
        counts = {
            status: exact_status_counts[(arm, status)]
            for status in (*STATUS_NAMES, "not_constructed")
            if exact_status_counts[(arm, status)]
        }
        total = historical_counts[arm]
        global_status[arm] = {
            "historical_anchor_count": total,
            "counts": counts,
            "fractions": {status: count / total for status, count in counts.items()},
            "nearest_current_candidate_start": _weighted_distribution(
                nearest_distance_counts[arm]
            ),
        }

    old_only_status_by_arm: dict[str, Any] = {}
    for arm in ARMS:
        counts = {
            status: old_only_anchor_statuses[(arm, status)]
            for status in (*STATUS_NAMES, "not_constructed")
            if old_only_anchor_statuses[(arm, status)]
        }
        total = sum(counts.values())
        old_only_status_by_arm[arm] = {
            "unique_historical_anchors": total,
            "counts": counts,
            "fractions": {status: count / total for status, count in counts.items()}
            if total
            else {},
        }

    json.dump(
        {
            "coordinate_contract": {
                "anchors": "0-based half-open",
                "vep_input": "1-based position",
                "vep_conversion": "position_zero = pos - 1",
                "window_size": WINDOW_SIZE,
            },
            "evaluation_boundary": {
                "dataset": "Mendelian development split only",
                "excluded_complete_groups": "mature_miRNA_variant",
                "allowed_chromosomes": "odd autosomes and X",
                "heldout_accessed": False,
                "labels_used_to_construct_anchor_rules": False,
            },
            "selection_contract": {
                "training_minimum": TRAINING_MIN,
                "projection_minimum": PROJECTION_MIN,
                "exact_status_precedence": [
                    "selected_ge_0.20",
                    "projection_0.10_to_0.20",
                    "conservation_below_0.10",
                    "lost_ownership",
                    "not_constructed",
                ],
            },
            "current_candidate_status_counts": {
                arm: {
                    status: candidates.counts[(arm, status)]
                    for status in STATUS_NAMES
                    if candidates.counts[(arm, status)]
                }
                for arm in ARMS
            },
            "historical_anchor_exact_current_stage": global_status,
            "historical_anchors_covering_historical_only_pathogenic_loci": (
                old_only_status_by_arm
            ),
            "pathogenic_development_coverage": subset_summaries,
            "start_phase_mod_128": {
                "historical": _serialize_phase_counts(historical_phases),
                "current_selected": _serialize_phase_counts(current_phases),
            },
        },
        sys.stdout,
        indent=2,
        sort_keys=True,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
