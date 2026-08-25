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
    selected: list[Interval] = []
    curve: list[tuple[int, int, float]] = []
    for rank, (interval, score) in enumerate(ranked, start=1):
        selected.append(interval)
        curve.append((rank, unique_bases(selected), score))
    return curve
