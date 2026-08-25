"""Construct fixed genomic windows without crossing ambiguous sequence."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum

DNA = frozenset("ACGTacgt")


class RejectionReason(StrEnum):
    AMBIGUOUS_BASE = "ambiguous_base"
    MAJORITY_SOFT_MASKED = "majority_soft_masked"


@dataclass(frozen=True, slots=True)
class Window:
    accession: str
    sequence_name: str
    start: int
    end: int
    strand: str
    sequence: str
    repeat_fraction: float

    @property
    def record_id(self) -> str:
        return encode_record_id(
            self.accession,
            self.sequence_name,
            self.start,
            self.end,
            self.strand,
        )


@dataclass(frozen=True, slots=True)
class RejectedWindow:
    accession: str
    sequence_name: str
    start: int
    end: int
    reason: RejectionReason


def encode_record_id(
    accession: str,
    sequence_name: str,
    start: int,
    end: int,
    strand: str,
) -> str:
    """Encode a reversible, whitespace-free genomic window ID."""
    assert accession and sequence_name
    assert all(character not in accession for character in "|\t\n ")
    assert all(character not in sequence_name for character in "|\t\n ")
    assert 0 <= start < end
    assert strand in {"+", "-"}
    return f"{accession}|{sequence_name}|{start}|{end}|{strand}"


def decode_record_id(record_id: str) -> tuple[str, str, int, int, str]:
    """Decode and validate a record ID created by :func:`encode_record_id`."""
    fields = record_id.split("|")
    assert len(fields) == 5, f"invalid record ID: {record_id!r}"
    accession, sequence_name, start_text, end_text, strand = fields
    start = int(start_text)
    end = int(end_text)
    assert encode_record_id(accession, sequence_name, start, end, strand) == record_id
    return accession, sequence_name, start, end, strand


def iter_windows(
    sequence: str,
    *,
    accession: str,
    sequence_name: str,
    window_length: int = 255,
    stride: int = 128,
    max_repeat_fraction: float = 0.5,
) -> Iterator[Window | RejectedWindow]:
    """Yield retained or rejected 0-based, half-open windows from one sequence."""
    assert window_length > 0
    assert stride > 0
    assert 0.0 <= max_repeat_fraction <= 1.0
    if len(sequence) < window_length:
        return
    for start in range(0, len(sequence) - window_length + 1, stride):
        end = start + window_length
        window_sequence = sequence[start:end]
        assert len(window_sequence) == window_length
        yield classify_window(
            window_sequence,
            accession=accession,
            sequence_name=sequence_name,
            start=start,
            max_repeat_fraction=max_repeat_fraction,
        )


def classify_window(
    sequence: str,
    *,
    accession: str,
    sequence_name: str,
    start: int,
    max_repeat_fraction: float = 0.5,
) -> Window | RejectedWindow:
    """Classify one already-extracted genomic window at its source coordinate."""
    assert sequence
    assert start >= 0
    assert 0.0 <= max_repeat_fraction <= 1.0
    end = start + len(sequence)
    if any(base not in DNA for base in sequence):
        return RejectedWindow(
            accession,
            sequence_name,
            start,
            end,
            RejectionReason.AMBIGUOUS_BASE,
        )
    repeat_fraction = sum(base.islower() for base in sequence) / len(sequence)
    if repeat_fraction > max_repeat_fraction:
        return RejectedWindow(
            accession,
            sequence_name,
            start,
            end,
            RejectionReason.MAJORITY_SOFT_MASKED,
        )
    return Window(
        accession=accession,
        sequence_name=sequence_name,
        start=start,
        end=end,
        strand="+",
        sequence=sequence,
        repeat_fraction=repeat_fraction,
    )
