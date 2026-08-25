"""Validate NCBI sequence roles and sample tiled source intervals."""

from __future__ import annotations

import bisect
import json
import random
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ALLOWED_ROLES = frozenset(
    {"assembled-molecule", "unlocalized-scaffold", "unplaced-scaffold"}
)
NON_NUCLEAR_ASSEMBLY_UNIT = "non-nuclear"
MITOCHONDRIAL_LOCATION_TYPE = "Mitochondrion"


@dataclass(frozen=True, slots=True)
class SourceSequence:
    assembly_accession: str
    sequence_accession: str
    length: int
    role: str
    assembly_unit: str
    assigned_molecule_location_type: str

    @property
    def is_mitochondrial(self) -> bool:
        return self.assigned_molecule_location_type == MITOCHONDRIAL_LOCATION_TYPE


@dataclass(frozen=True, slots=True)
class SampledInterval:
    sequence_accession: str
    start: int
    end: int


def is_primary_nuclear_record(record: Mapping[str, Any]) -> bool:
    """Identify principal nuclear sequence roles across NCBI assembly-unit labels."""
    role = str(record["role"])
    assembly_unit = str(record["assembly_unit"])
    location_type = str(record.get("assigned_molecule_location_type", ""))
    return (
        role in ALLOWED_ROLES
        and assembly_unit != NON_NUCLEAR_ASSEMBLY_UNIT
        and location_type != MITOCHONDRIAL_LOCATION_TYPE
    )


def parse_sequence_records(
    records: Iterable[Mapping[str, Any]],
) -> list[SourceSequence]:
    """Return primary-assembly sequence records accepted by the experiment."""
    sequences: list[SourceSequence] = []
    for record in records:
        assembly_unit = str(record["assembly_unit"])
        role = str(record["role"])
        if not is_primary_nuclear_record(record):
            continue
        sequence_accession = record.get("refseq_accession")
        assert sequence_accession, (
            f"{record.get('assembly_accession')}: primary RefSeq sequence lacks "
            "refseq_accession"
        )
        length = int(record["length"])
        assert length > 0
        sequences.append(
            SourceSequence(
                assembly_accession=str(record["assembly_accession"]),
                sequence_accession=str(sequence_accession),
                length=length,
                role=role,
                assembly_unit=assembly_unit,
                assigned_molecule_location_type=str(
                    record.get("assigned_molecule_location_type", "")
                ),
            )
        )
    keys = [
        (sequence.assembly_accession, sequence.sequence_accession)
        for sequence in sequences
    ]
    assert len(keys) == len(set(keys)), "duplicate sequence accessions in NCBI report"
    return sorted(
        sequences,
        key=lambda sequence: (
            sequence.assembly_accession,
            sequence.sequence_accession,
        ),
    )


def read_sequence_report(path: str | Path) -> list[SourceSequence]:
    """Read NCBI sequence-report JSONL and apply the experiment role filter."""
    records: list[dict[str, Any]] = []
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            assert isinstance(record, dict), f"{path}:{line_number}: expected object"
            records.append(record)
    sequences = parse_sequence_records(records)
    assert sequences, "NCBI sequence report has no eligible primary sequences"
    return sequences


def tiled_interval_count(
    sequences: Iterable[SourceSequence],
    *,
    window_length: int,
    stride: int,
) -> int:
    """Count fixed-grid intervals without materializing them."""
    assert window_length > 0
    assert stride > 0
    return sum(
        max(0, (sequence.length - window_length) // stride + 1)
        for sequence in sequences
    )


def iter_tiled_intervals(
    sequences: Iterable[SourceSequence],
    *,
    window_length: int,
    stride: int,
) -> Iterator[SampledInterval]:
    """Yield every fixed-grid interval in deterministic sequence order."""
    assert window_length > 0
    assert stride > 0
    for sequence in sorted(
        sequences,
        key=lambda source: source.sequence_accession,
    ):
        for start in range(0, sequence.length - window_length + 1, stride):
            yield SampledInterval(
                sequence_accession=sequence.sequence_accession,
                start=start,
                end=start + window_length,
            )


def sample_tiled_intervals(
    sequences: Iterable[SourceSequence],
    *,
    window_length: int,
    stride: int,
    sample_size: int,
    seed: int,
) -> list[SampledInterval]:
    """Sample tiled intervals uniformly without materializing a whole genome grid."""
    assert window_length > 0
    assert stride > 0
    assert sample_size > 0
    normalized = sorted(sequences, key=lambda sequence: sequence.sequence_accession)
    tile_counts = [
        tiled_interval_count(
            [sequence],
            window_length=window_length,
            stride=stride,
        )
        for sequence in normalized
    ]
    cumulative: list[int] = []
    total = 0
    for count in tile_counts:
        total += count
        cumulative.append(total)
    assert total >= sample_size, (
        f"requested {sample_size} intervals from a {total}-interval population"
    )
    sampled_indices = sorted(random.Random(seed).sample(range(total), sample_size))
    sampled: list[SampledInterval] = []
    for global_index in sampled_indices:
        sequence_index = bisect.bisect_right(cumulative, global_index)
        previous_total = cumulative[sequence_index - 1] if sequence_index else 0
        local_tile_index = global_index - previous_total
        start = local_tile_index * stride
        sampled.append(
            SampledInterval(
                sequence_accession=normalized[sequence_index].sequence_accession,
                start=start,
                end=start + window_length,
            )
        )
    assert len(sampled) == sample_size
    assert len(set(sampled)) == sample_size
    return sampled
