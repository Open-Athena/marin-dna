"""Stream large Linclust assignment tables into sensitivity metrics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TextIO


@dataclass(frozen=True, slots=True)
class AssignmentSummary:
    """Bounded-memory aggregate of a representative-grouped assignment table."""

    assignment_count: int
    cluster_count: int
    cross_genome_cluster_count: int
    cross_genome_member_count: int
    distinct_genome_histogram: dict[str, int]
    max_cluster_size: int
    max_distinct_genomes: int
    member_count_by_accession: dict[str, int]
    size_bucket_histogram: dict[str, int]
    singleton_cluster_count: int


def _size_bucket(size: int) -> str:
    if size == 1:
        return "1"
    if size == 2:
        return "2"
    if size == 3:
        return "3"
    if size <= 10:
        return "4-10"
    if size <= 100:
        return "11-100"
    if size <= 1_000:
        return "101-1000"
    return "1001+"


def summarize_grouped_assignments(
    lines: Iterable[str],
    *,
    expected_accessions: set[str],
) -> AssignmentSummary:
    """Summarize `createtsv` rows, which MMseqs emits grouped by representative."""
    assert expected_accessions
    assignment_count = 0
    cluster_count = 0
    cross_genome_cluster_count = 0
    cross_genome_member_count = 0
    singleton_cluster_count = 0
    max_cluster_size = 0
    max_distinct_genomes = 0
    member_count_by_accession: Counter[str] = Counter()
    size_bucket_histogram: Counter[str] = Counter()
    distinct_genome_histogram: Counter[str] = Counter()
    current_representative: str | None = None
    current_size = 0
    current_genomes: set[str] = set()
    current_has_self_row = False

    def finish_cluster() -> None:
        nonlocal cluster_count
        nonlocal cross_genome_cluster_count
        nonlocal cross_genome_member_count
        nonlocal singleton_cluster_count
        nonlocal max_cluster_size
        nonlocal max_distinct_genomes
        if current_representative is None:
            return
        assert current_size > 0
        assert current_has_self_row, (
            f"cluster {current_representative!r} lacks its representative self row"
        )
        distinct_genomes = len(current_genomes)
        cluster_count += 1
        singleton_cluster_count += int(current_size == 1)
        cross_genome_cluster_count += int(distinct_genomes > 1)
        if distinct_genomes > 1:
            cross_genome_member_count += current_size
        max_cluster_size = max(max_cluster_size, current_size)
        max_distinct_genomes = max(max_distinct_genomes, distinct_genomes)
        size_bucket_histogram[_size_bucket(current_size)] += 1
        distinct_genome_histogram[str(distinct_genomes)] += 1

    for line_number, line in enumerate(lines, start=1):
        stripped = line.rstrip("\n")
        if not stripped:
            continue
        fields = stripped.split("\t")
        assert len(fields) == 2, f"assignment line {line_number}: expected 2 fields"
        representative, member = fields
        assert representative and member
        if current_representative != representative:
            finish_cluster()
            current_representative = representative
            current_size = 0
            current_genomes = set()
            current_has_self_row = False
        accession = member.split("|", maxsplit=1)[0]
        assert accession in expected_accessions, (
            f"assignment line {line_number}: unexpected accession {accession!r}"
        )
        assignment_count += 1
        current_size += 1
        current_genomes.add(accession)
        current_has_self_row |= representative == member
        member_count_by_accession[accession] += 1
    finish_cluster()
    assert assignment_count > 0, "cluster assignment TSV is empty"
    assert cluster_count > 0
    return AssignmentSummary(
        assignment_count=assignment_count,
        cluster_count=cluster_count,
        cross_genome_cluster_count=cross_genome_cluster_count,
        cross_genome_member_count=cross_genome_member_count,
        distinct_genome_histogram=dict(sorted(distinct_genome_histogram.items())),
        max_cluster_size=max_cluster_size,
        max_distinct_genomes=max_distinct_genomes,
        member_count_by_accession=dict(sorted(member_count_by_accession.items())),
        size_bucket_histogram=dict(sorted(size_bucket_histogram.items())),
        singleton_cluster_count=singleton_cluster_count,
    )


def summarize_assignment_handle(
    handle: TextIO,
    *,
    expected_accessions: set[str],
) -> AssignmentSummary:
    """Summarize an open text handle without taking ownership of it."""
    return summarize_grouped_assignments(
        handle,
        expected_accessions=expected_accessions,
    )
