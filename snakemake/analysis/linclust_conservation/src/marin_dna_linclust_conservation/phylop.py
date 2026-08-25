"""Human assembly mapping and per-window phyloP labels."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

PHYLOP_THRESHOLD = 2.2162
WINDOW_LENGTH = 255

# RefSeq GRCh38.p14 assembled molecules to UCSC hg38 canonical autosomes.
REFSEQ_GRCH38_TO_UCSC: dict[str, str] = {
    "NC_000001.11": "chr1",
    "NC_000002.12": "chr2",
    "NC_000003.12": "chr3",
    "NC_000004.12": "chr4",
    "NC_000005.10": "chr5",
    "NC_000006.12": "chr6",
    "NC_000007.14": "chr7",
    "NC_000008.11": "chr8",
    "NC_000009.12": "chr9",
    "NC_000010.11": "chr10",
    "NC_000011.10": "chr11",
    "NC_000012.12": "chr12",
    "NC_000013.11": "chr13",
    "NC_000014.9": "chr14",
    "NC_000015.10": "chr15",
    "NC_000016.10": "chr16",
    "NC_000017.11": "chr17",
    "NC_000018.10": "chr18",
    "NC_000019.10": "chr19",
    "NC_000020.11": "chr20",
    "NC_000021.9": "chr21",
    "NC_000022.11": "chr22",
}


def autosome_number(chromosome: str) -> int:
    """Return a canonical autosome number for `chrN` or `N`."""
    normalized = chromosome.removeprefix("chr")
    assert normalized.isdigit(), f"not a canonical autosome: {chromosome!r}"
    number = int(normalized)
    assert 1 <= number <= 22, f"not a canonical autosome: {chromosome!r}"
    return number


def chromosome_split(chromosome: str) -> str:
    """Assign odd autosomes to tuning and even autosomes to sealed evaluation."""
    return "tuning" if autosome_number(chromosome) % 2 else "held_out"


def phyloP_fraction(
    values: Sequence[float | None],
    *,
    threshold: float = PHYLOP_THRESHOLD,
    window_length: int = WINDOW_LENGTH,
) -> float:
    """Count passing bases; missing and NaN values contribute zero."""
    assert len(values) == window_length, (
        f"expected {window_length} phyloP values, received {len(values)}"
    )
    passing = 0
    for value in values:
        if value is None:
            continue
        numeric = float(value)
        if not math.isnan(numeric) and numeric >= threshold:
            passing += 1
    return passing / window_length


def validate_human_mapping(
    *,
    refseq_lengths: Mapping[str, int],
    ucsc_lengths: Mapping[str, int],
    sampled_sequences: Iterable[tuple[str, int, int, str, str]],
    mapping: Mapping[str, str] = REFSEQ_GRCH38_TO_UCSC,
) -> None:
    """Validate mapped contig lengths and sampled sequence equality.

    Sample tuples contain `(refseq_name, start, end, refseq_sequence,
    ucsc_sequence)` using 0-based, half-open coordinates.
    """
    assert len(mapping) == 22
    assert len(set(mapping.values())) == 22
    for refseq_name, ucsc_name in mapping.items():
        assert refseq_name in refseq_lengths, f"missing RefSeq contig {refseq_name}"
        assert ucsc_name in ucsc_lengths, f"missing UCSC contig {ucsc_name}"
        assert refseq_lengths[refseq_name] == ucsc_lengths[ucsc_name], (
            f"length mismatch: {refseq_name} vs {ucsc_name}"
        )
    sample_count = 0
    for refseq_name, start, end, refseq_sequence, ucsc_sequence in sampled_sequences:
        assert refseq_name in mapping
        assert 0 <= start < end <= refseq_lengths[refseq_name]
        assert len(refseq_sequence) == end - start
        assert len(ucsc_sequence) == end - start
        assert refseq_sequence.upper() == ucsc_sequence.upper(), (
            f"sequence mismatch for {refseq_name}:{start}-{end}"
        )
        sample_count += 1
    assert sample_count > 0, "at least one sampled sequence comparison is required"
