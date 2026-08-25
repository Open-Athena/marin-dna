"""Evaluation helpers that preserve chromosome and footprint contracts."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Interval:
    chromosome: str
    start: int
    end: int

    def __post_init__(self) -> None:
        assert self.chromosome
        assert 0 <= self.start < self.end


def collapse_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    """Merge overlapping or touching 0-based, half-open intervals per chromosome."""
    ordered = sorted(
        intervals,
        key=lambda interval: (interval.chromosome, interval.start, interval.end),
    )
    collapsed: list[Interval] = []
    for interval in ordered:
        if (
            collapsed
            and collapsed[-1].chromosome == interval.chromosome
            and interval.start <= collapsed[-1].end
        ):
            previous = collapsed[-1]
            collapsed[-1] = Interval(
                previous.chromosome, previous.start, max(previous.end, interval.end)
            )
        else:
            collapsed.append(interval)
    return collapsed


def unique_bases(intervals: Iterable[Interval]) -> int:
    """Count unique bases covered by intervals."""
    return sum(
        interval.end - interval.start for interval in collapse_intervals(intervals)
    )


class _CoverageTree:
    """Incrementally track covered length over coordinate-compressed segments."""

    def __init__(self, intervals: Sequence[Interval]) -> None:
        self.coordinates = sorted(
            {
                coordinate
                for interval in intervals
                for coordinate in (interval.start, interval.end)
            }
        )
        assert len(self.coordinates) >= 2
        self.coordinate_index = {
            coordinate: index for index, coordinate in enumerate(self.coordinates)
        }
        segment_count = len(self.coordinates) - 1
        self.cover_counts = [0] * (4 * segment_count)
        self.covered_lengths = [0] * (4 * segment_count)
        self.segment_count = segment_count

    @property
    def covered_length(self) -> int:
        return self.covered_lengths[1]

    def add(self, interval: Interval) -> None:
        left = self.coordinate_index[interval.start]
        right = self.coordinate_index[interval.end] - 1
        self._add(
            node=1,
            segment_left=0,
            segment_right=self.segment_count - 1,
            left=left,
            right=right,
        )

    def _add(
        self,
        *,
        node: int,
        segment_left: int,
        segment_right: int,
        left: int,
        right: int,
    ) -> None:
        if left <= segment_left and segment_right <= right:
            self.cover_counts[node] += 1
        else:
            midpoint = (segment_left + segment_right) // 2
            if left <= midpoint:
                self._add(
                    node=node * 2,
                    segment_left=segment_left,
                    segment_right=midpoint,
                    left=left,
                    right=right,
                )
            if right > midpoint:
                self._add(
                    node=node * 2 + 1,
                    segment_left=midpoint + 1,
                    segment_right=segment_right,
                    left=left,
                    right=right,
                )
        if self.cover_counts[node] > 0:
            self.covered_lengths[node] = (
                self.coordinates[segment_right + 1] - self.coordinates[segment_left]
            )
        elif segment_left == segment_right:
            self.covered_lengths[node] = 0
        else:
            self.covered_lengths[node] = (
                self.covered_lengths[node * 2] + self.covered_lengths[node * 2 + 1]
            )


def ranked_footprint(
    windows: Sequence[tuple[Interval, float]],
) -> list[tuple[int, int, float]]:
    """Return rank, cumulative unique bases, and score after each ranked window."""
    ranked = sorted(
        windows,
        key=lambda item: (
            -item[1],
            item[0].chromosome,
            item[0].start,
            item[0].end,
        ),
    )
    by_chromosome: dict[str, list[Interval]] = {}
    for interval, _ in ranked:
        by_chromosome.setdefault(interval.chromosome, []).append(interval)
    coverage = {
        chromosome: _CoverageTree(intervals)
        for chromosome, intervals in by_chromosome.items()
    }
    cumulative_bases = 0
    curve: list[tuple[int, int, float]] = []
    for rank, (interval, score) in enumerate(ranked, start=1):
        chromosome_coverage = coverage[interval.chromosome]
        before = chromosome_coverage.covered_length
        chromosome_coverage.add(interval)
        cumulative_bases += chromosome_coverage.covered_length - before
        curve.append((rank, cumulative_bases, score))
    return curve
