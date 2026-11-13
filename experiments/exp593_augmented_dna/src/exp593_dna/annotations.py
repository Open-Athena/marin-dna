"""Train-row annotation sidecars and deterministic landmark-guided gap selection."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np

LANDMARK_CLASSES = (
    "translation_start",
    "translation_stop",
    "splice_donor",
    "splice_acceptor",
)
Placement = Literal["overlap", "nearby"]


@dataclass(frozen=True)
class Landmark:
    """One motif interval in the original stored window, zero-based half-open."""

    start: int
    end: int
    kind: str
    source: str
    feature_id: str = ""

    def __post_init__(self) -> None:
        if self.kind not in LANDMARK_CLASSES or not 0 <= self.start < self.end <= 8192:
            raise ValueError("Invalid stored-window landmark")
        if not self.source:
            raise ValueError("Annotation provenance is required")


@dataclass(frozen=True)
class RowAnnotations:
    """Bind uncertain feature locations to exact, existing source DNA."""

    row: int
    sequence_sha256: str
    landmarks: tuple[Landmark, ...]
    split: str = "train"

    def verify(self, sequence: str, split: str = "train") -> None:
        """Reject sidecars for another sequence or source split."""
        if self.split != split or len(self.sequence_sha256) != 64:
            raise ValueError("Annotation split/hash binding is invalid")
        if hashlib.sha256(sequence.encode("ascii")).hexdigest() != self.sequence_sha256:
            raise ValueError("Annotation sidecar DNA identity differs")


@dataclass(frozen=True)
class GapChoice:
    """Selection metadata; a missing start means retain the random control gap."""

    start: int | None = None
    eligible: bool = False
    guided: bool = False
    landmark_class: str = ""
    mode: str = ""
    offset: int | None = None
    source: str = ""
    reason: str = "no_annotation"


def load_annotations(
    path: Path, *, split: str = "train", rows: set[int] | None = None
) -> Mapping[int, RowAnnotations]:
    """Load a frozen split-specific JSONL; never silently accept duplicate rows."""
    if path.suffix == ".parquet":
        return AnnotationIndex(path, split=split, rows=rows)
    with path.open() as stream:
        lines = (json.loads(line) for line in stream)
        return _load_rows(lines, split, rows)


class AnnotationIndex(Mapping[int, RowAnnotations]):
    """Keep the full annotation bank in Arrow, materializing only queried rows.

    A dense integer lookup is about 11 MB for the original training split.
    This avoids millions of nested Python feature objects on every GPU process.
    """

    def __init__(
        self, path: Path, *, split: str = "train", rows: set[int] | None = None
    ) -> None:
        import pyarrow.parquet as pq

        from exp593_dna.pipeline import DATASET_REVISION

        metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
        if (
            metadata.get(b"dataset_revision") != DATASET_REVISION.encode()
            or metadata.get(b"split") != split.encode()
        ):
            raise ValueError("Annotation source revision or split differs")
        self.split = split
        self.table = pq.read_table(
            path, filters=[("row", "in", sorted(rows))] if rows is not None else None
        )
        self.rows = self.table.column("row").to_numpy()
        maximum = {"train": 2_638_656, "validation": 329_832}[split]
        if np.any((self.rows < 0) | (self.rows >= maximum)) or len(
            np.unique(self.rows)
        ) != len(self.rows):
            raise ValueError("Invalid or duplicated annotation row")
        self.lookup = np.full(maximum, -1, dtype=np.int32)
        self.lookup[self.rows] = np.arange(len(self.rows), dtype=np.int32)
        self._cached_row = lru_cache(maxsize=4096)(self._read_row)

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[int]:
        return (int(row) for row in self.rows)

    def __getitem__(self, row: int) -> RowAnnotations:
        if row < 0 or row >= len(self.lookup) or self.lookup[row] < 0:
            raise KeyError(row)
        return self._cached_row(int(self.lookup[row]))

    def _read_row(self, offset: int) -> RowAnnotations:
        record = self.table.slice(offset, 1).to_pylist()[0]
        return RowAnnotations(
            int(record["row"]),
            record["sequence_sha256"],
            tuple(Landmark(**m) for m in record["landmarks"]),
            self.split,
        )


def _load_rows(
    records: Iterable[dict[str, Any]], split: str, rows: set[int] | None
) -> dict[int, RowAnnotations]:
    """Stream a sidecar without retaining its serialized text in memory."""
    from exp593_dna.pipeline import DATASET_REVISION

    result = {}
    for rec in records:
        if rec["dataset_revision"] != DATASET_REVISION or rec["split"] != split:
            raise ValueError("Annotation source revision or split differs")
        row = int(rec["row"])
        if rows is not None and row not in rows:
            continue
        if row < 0 or row in result:
            raise ValueError("Invalid or duplicated annotation row")
        result[row] = RowAnnotations(
            row,
            rec["sequence_sha256"],
            tuple(Landmark(**x) for x in rec["landmarks"]),
            split,
        )
    return result


def candidate_starts(
    start: int,
    end: int,
    gap_length: int,
    dna_length: int,
    mode: Placement,
    *,
    neighborhood: int = 32,
    minimum_flank: int = 128,
) -> tuple[int, ...]:
    """Find valid starts with maximal motif overlap, or fully outside it nearby."""
    if not 0 <= start < end <= dna_length or gap_length < 1:
        return ()
    lo, hi = minimum_flank, dna_length - minimum_flank - gap_length
    if lo > hi:
        return ()
    if mode == "overlap":
        candidates = list(range(max(lo, start - gap_length + 1), min(hi, end - 1) + 1))
        if not candidates:
            return ()
        overlap = lambda p: min(p + gap_length, end) - max(p, start)
        best = max(map(overlap, candidates))
        return tuple(p for p in candidates if overlap(p) == best)
    if mode != "nearby":
        raise ValueError("Unknown guided placement mode")
    left = range(max(lo, start - neighborhood), min(hi, start - gap_length) + 1)
    right = range(max(lo, end), min(hi, end + neighborhood - gap_length) + 1)
    return (*left, *right)


def choose_guided_gap(
    landmarks: tuple[Landmark, ...],
    rng: np.random.Generator,
    *,
    reverse_complemented: bool,
    crop_start: int,
    dna_length: int,
    gap_length: int,
    guided_probability: float = 0.75,
    overlap_probability: float = 0.50,
) -> GapChoice:
    """Choose a class uniformly before a deduplicated site, with random fallback."""
    mode: Placement = "overlap" if rng.random() < overlap_probability else "nearby"
    by_class: dict[str, list[tuple[Landmark, int, tuple[int, ...]]]] = defaultdict(list)
    seen = set()
    for feature in landmarks:
        start, end = feature.start, feature.end
        if reverse_complemented:
            start, end = 8192 - end, 8192 - start
        start, end = start - crop_start, end - crop_start
        key = (feature.kind, start, end)
        if key in seen:
            continue
        seen.add(key)
        starts = candidate_starts(start, end, gap_length, dna_length, mode)
        if starts:
            by_class[feature.kind].append((feature, start, starts))
    if not by_class:
        return GapChoice(
            mode=mode, reason="no_valid_site" if landmarks else "no_annotation"
        )
    if rng.random() >= guided_probability:
        return GapChoice(eligible=True, mode=mode, reason="random_control")
    classes = sorted(by_class)
    kind = classes[int(rng.integers(len(classes)))]
    sites = by_class[kind]
    feature, motif_start, starts = sites[int(rng.integers(len(sites)))]
    chosen = starts[int(rng.integers(len(starts)))]
    return GapChoice(
        chosen, True, True, kind, mode, chosen - motif_start, feature.source, ""
    )
