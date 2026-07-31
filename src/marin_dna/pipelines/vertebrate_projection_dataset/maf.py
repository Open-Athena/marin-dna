"""Streaming UCSC MAF adapter for human-anchored interval projection.

MAF ``start`` coordinates on ``-`` rows are measured in the reverse-
complemented source.  This module converts them to forward-strand, 0-based,
half-open coordinates at parse/use time.  Nothing downstream is allowed to
carry MAF-native coordinates.
"""

from __future__ import annotations

import gzip
from bisect import bisect_left
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import polars as pl

from marin_dna.pipelines.vertebrate_projection_dataset.manifest import (
    validate_species_manifest,
)


FRAGMENT_SCHEMA = pl.Schema(
    {
        "query_name": pl.String,
        "source_chrom": pl.String,
        "source_start": pl.Int64,
        "source_end": pl.Int64,
        "region_label": pl.String,
        "source_fragment_start": pl.Int64,
        "source_fragment_end": pl.Int64,
        "species": pl.String,
        "alignment_name": pl.String,
        "assembly": pl.String,
        "taxonomy_id": pl.Int64,
        "family": pl.String,
        "clade": pl.String,
        "phylogenetic_rank": pl.Int64,
        "alignment_source": pl.String,
        "t_chrom": pl.String,
        "t_start": pl.Int64,
        "t_end": pl.Int64,
        "t_strand": pl.String,
        "t_src_size": pl.Int64,
        "mapping_id": pl.String,
        "fragment_id": pl.String,
        "aligned_bases": pl.Int64,
    }
)


def _as_int(value: object) -> int:
    assert isinstance(value, int | str) and not isinstance(value, bool)
    return int(value)


@dataclass(frozen=True)
class MafSequence:
    """One ``s`` row from a MAF block."""

    src: str
    start: int
    size: int
    strand: str
    src_size: int
    text: str

    @property
    def alignment_name(self) -> str:
        """Return the UCSC assembly/database prefix of ``src``."""
        assert "." in self.src, f"MAF src must be assembly.chrom: {self.src!r}"
        return self.src.split(".", 1)[0]

    @property
    def chrom(self) -> str:
        """Return the chromosome component of ``src`` without truncating dots."""
        assert "." in self.src, f"MAF src must be assembly.chrom: {self.src!r}"
        return self.src.split(".", 1)[1]

    @property
    def forward_interval(self) -> tuple[int, int]:
        """Convert the MAF-native interval to forward 0-based half-open coords."""
        assert self.strand in {"+", "-"}
        if self.strand == "+":
            forward_start = self.start
        else:
            forward_start = self.src_size - self.start - self.size
        forward_end = forward_start + self.size
        assert 0 <= forward_start <= forward_end <= self.src_size, (
            f"invalid MAF interval for {self.src}: start={self.start} "
            f"size={self.size} strand={self.strand} src_size={self.src_size}"
        )
        return forward_start, forward_end

    def forward_coordinates(self) -> list[int | None]:
        """Return a forward genomic coordinate (or gap) per alignment column."""
        assert self.strand in {"+", "-"}
        expected_bases = sum(base != "-" for base in self.text)
        assert expected_bases == self.size, (
            f"MAF size/text mismatch for {self.src}: size={self.size}, "
            f"non_gap={expected_bases}"
        )
        if self.strand == "+":
            coordinate = self.start
            step = 1
        else:
            coordinate = self.src_size - self.start - 1
            step = -1

        result: list[int | None] = []
        for base in self.text:
            if base == "-":
                result.append(None)
            else:
                assert 0 <= coordinate < self.src_size
                result.append(coordinate)
                coordinate += step
        return result


@dataclass(frozen=True)
class MafBlock:
    """A parsed MAF alignment block."""

    block_id: int
    sequences: tuple[MafSequence, ...]


@contextmanager
def _open_maf(path: str | Path) -> Iterator[TextIO]:
    maf_path = Path(path)
    if maf_path.suffix == ".gz":
        with gzip.open(maf_path, "rt") as handle:
            yield handle
    else:
        with maf_path.open() as handle:
            yield handle


def iter_maf_blocks(path: str | Path) -> Iterator[MafBlock]:
    """Yield schema-checked MAF blocks without loading the file into memory."""
    block_id = 0
    sequences: list[MafSequence] = []

    def finish_block() -> MafBlock | None:
        nonlocal block_id, sequences
        if not sequences:
            return None
        text_lengths = {len(sequence.text) for sequence in sequences}
        assert len(text_lengths) == 1, (
            f"MAF block {block_id} has inconsistent alignment text lengths"
        )
        block = MafBlock(block_id=block_id, sequences=tuple(sequences))
        block_id += 1
        sequences = []
        return block

    with _open_maf(path) as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                block = finish_block()
                if block is not None:
                    yield block
                continue
            if line.startswith("#"):
                continue
            fields = line.split()
            record_type = fields[0]
            if record_type == "a":
                block = finish_block()
                if block is not None:
                    yield block
                continue
            if record_type != "s":
                # UCSC primary MAFs include i/e/q annotations.  They carry gap
                # context or quality but do not change s-row coordinates.
                continue
            assert len(fields) == 7, (
                f"malformed MAF s row at line {line_number}: {line!r}"
            )
            sequence = MafSequence(
                src=fields[1],
                start=int(fields[2]),
                size=int(fields[3]),
                strand=fields[4],
                src_size=int(fields[5]),
                text=fields[6],
            )
            # Force coordinate-convention assertions at the file boundary.
            sequence.forward_interval
            sequences.append(sequence)

    block = finish_block()
    if block is not None:
        yield block


def _fragment_runs(
    source: MafSequence,
    target: MafSequence,
    source_start: int,
    source_end: int,
) -> list[list[tuple[int, int]]]:
    """Return contiguous aligned source/target coordinate runs in an interval."""
    source_coordinates = source.forward_coordinates()
    target_coordinates = target.forward_coordinates()
    assert len(source_coordinates) == len(target_coordinates)

    pairs = [
        (source_coordinate, target_coordinate)
        for source_coordinate, target_coordinate in zip(
            source_coordinates, target_coordinates
        )
        if source_coordinate is not None
        and target_coordinate is not None
        and source_start <= source_coordinate < source_end
    ]
    if not pairs:
        return []

    source_step = 1 if source.strand == "+" else -1
    target_step = 1 if target.strand == "+" else -1
    runs: list[list[tuple[int, int]]] = [[pairs[0]]]
    for pair in pairs[1:]:
        previous = runs[-1][-1]
        if (
            pair[0] == previous[0] + source_step
            and pair[1] == previous[1] + target_step
        ):
            runs[-1].append(pair)
        else:
            runs.append([pair])
    return runs


def iter_projected_anchor_fragments(
    maf_path: str | Path,
    anchors: pl.DataFrame,
    species_manifest: pl.DataFrame,
    *,
    source_alignment_name: str = "hg38",
) -> Iterator[dict[str, object]]:
    """Yield common fragments projected through a human-referenced MAF.

    Rows are *candidates*, not accepted projections. Split blocks, gaps,
    duplicated source coverage, and target ambiguity remain explicit for the
    shared projection contract to adjudicate. The iterator keeps full-size MAF
    parsing bounded in memory; callers that need a DataFrame may materialize it.
    """
    required_anchor_columns = {
        "query_name",
        "source_chrom",
        "source_start",
        "source_end",
        "region_label",
    }
    missing = required_anchor_columns - set(anchors.columns)
    assert not missing, f"anchors missing columns: {sorted(missing)}"
    assert anchors["query_name"].n_unique() == anchors.height
    assert (anchors["source_start"] >= 0).all()
    assert (anchors["source_end"] > anchors["source_start"]).all()

    validate_species_manifest(species_manifest)
    targets = species_manifest.filter(
        (pl.col("backend") == "ucsc_multiz100way") & pl.col("selected")
    )
    target_metadata: dict[str, dict[str, object]] = {
        str(row["alignment_name"]): row for row in targets.to_dicts()
    }
    assert target_metadata, "no selected MultiZ targets"

    anchors_by_chrom: dict[str, list[dict[str, object]]] = {}
    starts_by_chrom: dict[str, list[int]] = {}
    max_length_by_chrom: dict[str, int] = {}
    for chrom, anchor_frame in anchors.partition_by(
        "source_chrom", as_dict=True, maintain_order=True
    ).items():
        chrom_name = str(chrom[0] if isinstance(chrom, tuple) else chrom)
        sorted_anchors = anchor_frame.sort("source_start", "source_end")
        rows = sorted_anchors.to_dicts()
        anchors_by_chrom[chrom_name] = rows
        starts_by_chrom[chrom_name] = [int(row["source_start"]) for row in rows]
        max_length_by_chrom[chrom_name] = max(
            int(row["source_end"]) - int(row["source_start"]) for row in rows
        )

    for block in iter_maf_blocks(maf_path):
        source_rows = [
            row
            for row in block.sequences
            if row.alignment_name == source_alignment_name
        ]
        assert len(source_rows) <= 1, (
            f"MAF block {block.block_id} contains duplicate source rows"
        )
        if not source_rows:
            continue
        source = source_rows[0]
        if source.chrom not in anchors_by_chrom:
            continue
        block_start, block_end = source.forward_interval
        chrom_anchor_rows = anchors_by_chrom[source.chrom]
        anchor_starts = starts_by_chrom[source.chrom]
        max_anchor_length = max_length_by_chrom[source.chrom]
        first = bisect_left(anchor_starts, block_start - max_anchor_length)
        last = bisect_left(anchor_starts, block_end)
        overlapping_anchors = [
            anchor
            for anchor in chrom_anchor_rows[first:last]
            if _as_int(anchor["source_end"]) > block_start
            and _as_int(anchor["source_start"]) < block_end
        ]
        if not overlapping_anchors:
            continue

        block_targets: dict[str, MafSequence] = {}
        for row in block.sequences:
            if row.alignment_name not in target_metadata:
                continue
            assert row.alignment_name not in block_targets, (
                f"MAF block {block.block_id} contains duplicate rows for "
                f"{row.alignment_name}"
            )
            block_targets[row.alignment_name] = row

        for alignment_name, target in block_targets.items():
            metadata = target_metadata[alignment_name]
            target_strand = "+" if source.strand == target.strand else "-"
            for anchor in overlapping_anchors:
                anchor_start = _as_int(anchor["source_start"])
                anchor_end = _as_int(anchor["source_end"])
                runs = _fragment_runs(source, target, anchor_start, anchor_end)
                mapping_id = f"maf:{block.block_id}:{target.src}"
                for run_number, run in enumerate(runs):
                    source_coordinates = [pair[0] for pair in run]
                    target_coordinates = [pair[1] for pair in run]
                    yield {
                        "query_name": str(anchor["query_name"]),
                        "source_chrom": str(anchor["source_chrom"]),
                        "source_start": anchor_start,
                        "source_end": anchor_end,
                        "region_label": str(anchor["region_label"]),
                        "source_fragment_start": min(source_coordinates),
                        "source_fragment_end": max(source_coordinates) + 1,
                        "species": str(metadata["scientific_name"]),
                        "alignment_name": alignment_name,
                        "assembly": str(metadata["assembly"]),
                        "taxonomy_id": _as_int(metadata["taxonomy_id"]),
                        "family": str(metadata["family"]),
                        "clade": str(metadata["clade"]),
                        "phylogenetic_rank": _as_int(metadata["phylogenetic_rank"]),
                        "alignment_source": "ucsc_multiz100way",
                        "t_chrom": target.chrom,
                        "t_start": min(target_coordinates),
                        "t_end": max(target_coordinates) + 1,
                        "t_strand": target_strand,
                        "t_src_size": target.src_size,
                        "mapping_id": mapping_id,
                        "fragment_id": f"{mapping_id}:{run_number}",
                        "aligned_bases": len(run),
                    }


def project_anchors_from_maf(
    maf_path: str | Path,
    anchors: pl.DataFrame,
    species_manifest: pl.DataFrame,
    *,
    source_alignment_name: str = "hg38",
) -> pl.DataFrame:
    """Materialize projected MAF candidates for tests and small inputs."""
    fragments = list(
        iter_projected_anchor_fragments(
            maf_path,
            anchors,
            species_manifest,
            source_alignment_name=source_alignment_name,
        )
    )

    if not fragments:
        return pl.DataFrame(schema=FRAGMENT_SCHEMA)
    result = pl.DataFrame(fragments, schema=FRAGMENT_SCHEMA)
    assert (result["source_fragment_start"] >= result["source_start"]).all()
    assert (result["source_fragment_end"] <= result["source_end"]).all()
    assert (result["t_start"] >= 0).all()
    assert (result["t_end"] <= result["t_src_size"]).all()
    return result.sort("query_name", "species", "mapping_id", "fragment_id")
