"""CDS-only secondary genomic strata for issue #478.

All coordinates are 0-based, half-open. GTF conversion happens at the loader
boundary. Overlapping transcripts are retained; a base is marked ambiguous
unless every overlapping annotation agrees on both label and gene strand.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import polars as pl

from marin_dna.data.utils import load_annotation


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    strand: str
    label: int


def load_cds_and_exons(gtf_path: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load RefSeq GTF and return transcript-keyed CDS segments and exons."""
    annotation = load_annotation(gtf_path).with_columns(
        pl.col("attribute").str.extract(r'transcript_id "(.*?)"').alias("transcript_id")
    )
    cds = (
        annotation.filter(
            (pl.col("feature") == "CDS") & pl.col("transcript_id").is_not_null()
        )
        .select(["chrom", "start", "end", "strand", "transcript_id", "frame"])
        .with_columns(pl.col("frame").cast(pl.Int8, strict=False))
    )
    valid = cds["frame"].is_in([0, 1, 2]).fill_null(False)
    assert valid.all(), (
        "every CDS segment needs GTF phase 0/1/2; "
        f"{int((~valid).sum())} invalid segment(s)"
    )
    exons = annotation.filter(
        (pl.col("feature") == "exon") & pl.col("transcript_id").is_not_null()
    ).select(["chrom", "start", "end", "strand", "transcript_id"])
    assert len(cds) > 0 and len(exons) > 0, "empty CDS/exon annotation"
    return cds, exons


def codon_segments(cds: pl.DataFrame) -> dict[str, list[tuple[int, int, str, int]]]:
    """Convert a CDS frame to per-segment records for window-local annotation."""
    required = {"chrom", "start", "end", "strand", "frame"}
    missing = required - set(cds.columns)
    assert not missing, f"CDS missing columns {sorted(missing)}"
    out: dict[str, list[tuple[int, int, str, int]]] = defaultdict(list)
    for chrom, start, end, strand, frame in cds.select(
        ["chrom", "start", "end", "strand", "frame"]
    ).iter_rows():
        assert strand in {"+", "-"}
        assert int(end) > int(start)
        assert int(frame) in {0, 1, 2}
        out[str(chrom)].append((int(start), int(end), str(strand), int(frame)))
    return {chrom: sorted(records) for chrom, records in out.items()}


def canonical_splice_segments(exons: pl.DataFrame) -> dict[str, list[Segment]]:
    """Return the canonical two intronic bases at each donor/acceptor.

    Labels are ``1=donor`` and ``2=acceptor``. Donor/acceptor orientation is
    strand-aware. A short intron may carry both labels and is resolved as
    ambiguous downstream.
    """
    required = {"chrom", "start", "end", "strand", "transcript_id"}
    missing = required - set(exons.columns)
    assert not missing, f"exons missing columns {sorted(missing)}"
    out: dict[str, list[Segment]] = defaultdict(list)
    for group in exons.partition_by("transcript_id", maintain_order=False):
        rows = sorted(group.iter_rows(named=True), key=lambda row: int(row["start"]))
        for left, right in pairwise(rows):
            assert left["chrom"] == right["chrom"]
            assert left["strand"] == right["strand"]
            intron_start = int(left["end"])
            intron_end = int(right["start"])
            if intron_end <= intron_start:
                continue
            low_end = min(intron_start + 2, intron_end)
            high_start = max(intron_end - 2, intron_start)
            strand = str(left["strand"])
            low_label, high_label = (1, 2) if strand == "+" else (2, 1)
            chrom = str(left["chrom"])
            out[chrom].append(Segment(intron_start, low_end, strand, low_label))
            out[chrom].append(Segment(high_start, intron_end, strand, high_label))
    return {
        chrom: sorted(records, key=lambda record: record.start)
        for chrom, records in out.items()
    }


def _bin_records(
    records: Iterable[tuple[int, int, object]],
    *,
    bin_size: int = 1_000_000,
) -> dict[int, list[tuple[int, int, object]]]:
    bins: dict[int, list[tuple[int, int, object]]] = defaultdict(list)
    for start, end, payload in records:
        assert end > start
        for bin_id in range(start // bin_size, (end - 1) // bin_size + 1):
            bins[bin_id].append((start, end, payload))
    return dict(bins)


def _overlapping(
    bins: dict[int, list[tuple[int, int, object]]],
    start: int,
    end: int,
    *,
    bin_size: int = 1_000_000,
) -> Iterable[tuple[int, int, object]]:
    seen: set[tuple[int, int, object]] = set()
    for bin_id in range(start // bin_size, (end - 1) // bin_size + 1):
        for record in bins.get(bin_id, []):
            if record not in seen and record[0] < end and record[1] > start:
                seen.add(record)
                yield record


def annotate_cds_windows(
    windows: list[tuple[str, int, int]],
    cds: pl.DataFrame,
    exons: pl.DataFrame,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Annotate CDS windows with resolved codon and canonical-splice labels.

    Returns four forward-coordinate int8 arrays per window:
    ``codon_position`` (0 none, 1/2/3, -1 ambiguous), ``codon_strand``
    (0 none, +1 plus, -1 minus, 2 ambiguous), ``splice_class`` (0 none,
    1 donor, 2 acceptor, -1 ambiguous), and ``splice_strand`` with the same
    strand encoding.
    """
    cds_by_chrom = codon_segments(cds)
    splice_by_chrom = canonical_splice_segments(exons)
    cds_bins = {
        chrom: _bin_records(
            (start, end, (strand, phase)) for start, end, strand, phase in records
        )
        for chrom, records in cds_by_chrom.items()
    }
    splice_bins = {
        chrom: _bin_records(
            (record.start, record.end, (record.strand, record.label))
            for record in records
        )
        for chrom, records in splice_by_chrom.items()
    }

    codon_arrays: list[np.ndarray] = []
    codon_strands: list[np.ndarray] = []
    splice_arrays: list[np.ndarray] = []
    splice_strands: list[np.ndarray] = []
    for chrom, window_start, window_end in windows:
        width = window_end - window_start
        codon: list[set[tuple[int, str]]] = [set() for _ in range(width)]
        splice: list[set[tuple[int, str]]] = [set() for _ in range(width)]
        for start, end, payload in _overlapping(
            cds_bins.get(chrom, {}), window_start, window_end
        ):
            strand, phase = payload
            lo, hi = max(start, window_start), min(end, window_end)
            for pos in range(lo, hi):
                offset = pos - start if strand == "+" else end - 1 - pos
                codon[pos - window_start].add((((offset - phase) % 3) + 1, strand))
        for start, end, payload in _overlapping(
            splice_bins.get(chrom, {}), window_start, window_end
        ):
            strand, label = payload
            lo, hi = max(start, window_start), min(end, window_end)
            for pos in range(lo, hi):
                splice[pos - window_start].add((label, strand))

        def resolve(
            values: list[set[tuple[int, str]]],
            array_width: int,
        ) -> tuple[np.ndarray, np.ndarray]:

            label_out = np.zeros(array_width, dtype=np.int8)
            strand_out = np.zeros(array_width, dtype=np.int8)
            for index, labels in enumerate(values):
                if not labels:
                    continue
                if len(labels) != 1:
                    label_out[index] = -1
                    strand_out[index] = 2
                    continue
                label, strand = next(iter(labels))
                label_out[index] = label
                strand_out[index] = 1 if strand == "+" else -1
            return label_out, strand_out

        codon_out, codon_strand = resolve(codon, width)
        splice_out, splice_strand = resolve(splice, width)

        codon_arrays.append(codon_out)
        codon_strands.append(codon_strand)
        splice_arrays.append(splice_out)
        splice_strands.append(splice_strand)
    return codon_arrays, codon_strands, splice_arrays, splice_strands
