"""Build a bounded positive-control fixture from projected orthologous windows."""

from __future__ import annotations

import csv
import json
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import polars as pl

from marin_dna_linclust_conservation.staging import (
    download_staged_genome,
    sha256_file,
)


@dataclass(frozen=True, slots=True)
class ProjectionSource:
    label: str
    species: str
    assembly: str
    uri: str
    etag: str
    size_bytes: int
    genome_uri: str | None = None
    genome_etag: str | None = None
    genome_size_bytes: int | None = None

    @classmethod
    def from_dict(cls, row: dict[str, object]) -> ProjectionSource:
        source = cls(
            label=str(row["label"]),
            species=str(row["species"]),
            assembly=str(row["assembly"]),
            uri=str(row["uri"]),
            etag=str(row["etag"]),
            size_bytes=int(row["size_bytes"]),
            genome_uri=(
                str(row["genome_uri"]) if row.get("genome_uri") is not None else None
            ),
            genome_etag=(
                str(row["genome_etag"]) if row.get("genome_etag") is not None else None
            ),
            genome_size_bytes=(
                int(row["genome_size_bytes"])
                if row.get("genome_size_bytes") is not None
                else None
            ),
        )
        genome_fields = (
            source.genome_uri,
            source.genome_etag,
            source.genome_size_bytes,
        )
        assert all(value is None for value in genome_fields) or all(
            value is not None for value in genome_fields
        ), f"incomplete genome declaration for {source.label}"
        return source


FIXTURE_COLUMNS = [
    "query_name",
    "source_chrom",
    "source_start",
    "source_end",
    "region_label",
    "species",
    "assembly",
    "sequence",
]

EXPANDED_FIXTURE_COLUMNS = [
    *FIXTURE_COLUMNS,
    "t_chrom",
    "t_strand",
    "t_src_size",
    "pre_resize_t_start",
    "pre_resize_t_end",
]


def _common_query_names(paths: list[Path]) -> set[str]:
    common: set[str] | None = None
    for path in paths:
        names = set(
            pl.scan_parquet(path)
            .select("query_name")
            .collect(engine="streaming")["query_name"]
            .to_list()
        )
        common = names if common is None else common & names
        assert common, f"projection sources share no anchors after reading {path}"
    assert common is not None
    return common


def _sequence_rejection(sequence: str, *, window_length: int) -> str | None:
    if len(sequence) != window_length:
        return "wrong_length"
    if any(base not in "ACGT" for base in sequence.upper()):
        return "ambiguous_base"
    if sum(base.islower() for base in sequence) * 2 > window_length:
        return "majority_lowercase"
    return None


def build_projection_fixture(
    *,
    sources: list[ProjectionSource],
    paths: list[Path],
    max_anchors: int,
    candidate_anchors: int,
    window_length: int,
    fasta_path: str | Path,
    truth_path: str | Path,
) -> dict[str, object]:
    """Select complete projected anchor groups and write FASTA plus truth metadata."""
    assert len(sources) >= 2
    assert len(sources) == len(paths)
    assert max_anchors > 0
    assert candidate_anchors >= max_anchors
    assert window_length > 0
    assert len({source.label for source in sources}) == len(sources)
    assert len({source.species for source in sources}) == len(sources)
    assert all(path.is_file() for path in paths)

    common = _common_query_names(paths)
    candidates = sorted(common)[:candidate_anchors]
    candidate_set = set(candidates)
    rows_by_source: dict[str, dict[str, dict[str, object]]] = {}
    for source, path in zip(sources, paths, strict=True):
        frame = (
            pl.scan_parquet(path)
            .filter(pl.col("query_name").is_in(sorted(candidate_set)))
            .select(FIXTURE_COLUMNS)
            .collect(engine="streaming")
        )
        assert frame["query_name"].n_unique() == frame.height
        assert set(frame["species"].unique().to_list()) == {source.species}
        assert set(frame["assembly"].unique().to_list()) == {source.assembly}
        rows_by_source[source.label] = {
            str(row["query_name"]): row for row in frame.to_dicts()
        }

    selected: list[tuple[str, list[dict[str, object]]]] = []
    rejection_counts: dict[str, int] = {}
    for query_name in candidates:
        rows = [rows_by_source[source.label][query_name] for source in sources]
        coordinates = {
            (
                str(row["source_chrom"]),
                int(row["source_start"]),
                int(row["source_end"]),
                str(row["region_label"]),
            )
            for row in rows
        }
        assert len(coordinates) == 1, f"source metadata differs for {query_name}"
        reasons = {
            reason
            for row in rows
            if (
                reason := _sequence_rejection(
                    str(row["sequence"]), window_length=window_length
                )
            )
            is not None
        }
        if reasons:
            for reason in reasons:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            continue
        selected.append((query_name, rows))
        if len(selected) == max_anchors:
            break
    assert len(selected) == max_anchors, {
        "selected": len(selected),
        "requested": max_anchors,
        "candidate_anchors": candidate_anchors,
        "rejections": rejection_counts,
    }

    fasta_path = Path(fasta_path)
    truth_path = Path(truth_path)
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    truth_fields = [
        "anchor_index",
        "query_name",
        "record_id",
        "source_label",
        "species",
        "assembly",
        "region_label",
        "source_chrom",
        "source_start",
        "source_end",
    ]
    with fasta_path.open("w") as fasta, truth_path.open("w", newline="") as truth:
        writer = csv.DictWriter(truth, fieldnames=truth_fields, delimiter="\t")
        writer.writeheader()
        for anchor_index, (query_name, rows) in enumerate(selected):
            for source, row in zip(sources, rows, strict=True):
                record_id = f"anchor{anchor_index:06d}__{source.label}"
                sequence = str(row["sequence"])
                fasta.write(f">{record_id}\n{sequence}\n")
                writer.writerow(
                    {
                        "anchor_index": anchor_index,
                        "query_name": query_name,
                        "record_id": record_id,
                        "source_label": source.label,
                        "species": source.species,
                        "assembly": source.assembly,
                        "region_label": row["region_label"],
                        "source_chrom": row["source_chrom"],
                        "source_start": row["source_start"],
                        "source_end": row["source_end"],
                    }
                )
    return {
        "candidate_anchor_limit": candidate_anchors,
        "common_anchor_count": len(common),
        "input_sequence_count": len(selected) * len(sources),
        "rejected_candidate_anchor_counts": dict(sorted(rejection_counts.items())),
        "selected_anchor_count": len(selected),
        "source_count": len(sources),
        "truth_sha256": sha256_file(truth_path),
        "fasta_sha256": sha256_file(fasta_path),
        "window_length": window_length,
    }


def _centered_interval(
    *, start: int, end: int, window_length: int, sequence_length: int
) -> tuple[int, int] | None:
    """Return a centered 0-based half-open interval, or ``None`` at an edge."""
    assert end > start
    assert window_length > 0
    midpoint = (start + end) // 2
    centered_start = midpoint - window_length // 2
    centered_end = centered_start + window_length
    if centered_start < 0 or centered_end > sequence_length:
        return None
    return centered_start, centered_end


def _read_fasta_records(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    name: str | None = None
    chunks: list[str] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if stripped.startswith(">"):
                if name is not None:
                    assert chunks, f"empty FASTA sequence for {name!r}"
                    records.append((name, "".join(chunks)))
                name = stripped[1:].split(maxsplit=1)[0]
                assert name, f"empty FASTA header at line {line_number}"
                chunks = []
            else:
                assert name is not None, (
                    f"sequence before FASTA header at line {line_number}"
                )
                assert stripped, f"empty FASTA line at line {line_number}"
                chunks.append(stripped)
    if name is not None:
        assert chunks, f"empty FASTA sequence for {name!r}"
        records.append((name, "".join(chunks)))
    return records


def _extract_twobit_intervals(
    *,
    genome_path: Path,
    intervals: list[tuple[str, str, int, int, str]],
    temporary: Path,
    two_bit_to_fa: str,
    window_length: int,
) -> dict[str, str]:
    bed_path = temporary / f"{genome_path.stem}.bed"
    fasta_path = temporary / f"{genome_path.stem}.fasta"
    with bed_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        for record_id, chrom, start, end, strand in intervals:
            assert end - start == window_length
            assert strand in {"+", "-"}
            writer.writerow((chrom, start, end, record_id, 0, strand))
    subprocess.run(
        [two_bit_to_fa, str(genome_path), str(fasta_path), f"-bed={bed_path}"],
        check=True,
    )
    records = _read_fasta_records(fasta_path)
    expected_ids = [record_id for record_id, *_ in intervals]
    observed_ids = [record_id for record_id, _ in records]
    assert observed_ids == expected_ids, (
        "twoBitToFa changed BED record order or names: "
        f"expected {expected_ids[:3]}, observed {observed_ids[:3]}"
    )
    sequences = {record_id: sequence for record_id, sequence in records}
    assert all(len(sequence) == window_length for sequence in sequences.values())
    return sequences


def build_center_expanded_projection_fixture(
    *,
    sources: list[ProjectionSource],
    projection_paths: list[Path],
    genome_paths: list[Path],
    max_anchors: int,
    candidate_anchors: int,
    selection_window_length: int,
    window_length: int,
    fasta_path: str | Path,
    truth_path: str | Path,
    two_bit_to_fa: str = "twoBitToFa",
) -> dict[str, object]:
    """Expand projected centers and extract a matched longer-context fixture."""
    assert len(sources) >= 2
    assert len(sources) == len(projection_paths) == len(genome_paths)
    assert max_anchors > 0
    assert candidate_anchors >= max_anchors
    assert window_length > selection_window_length > 0
    assert all(path.is_file() for path in projection_paths)
    assert all(path.is_file() for path in genome_paths)

    common = _common_query_names(projection_paths)
    candidates = sorted(common)[:candidate_anchors]
    candidate_set = set(candidates)
    rows_by_source: dict[str, dict[str, dict[str, object]]] = {}
    for source, path in zip(sources, projection_paths, strict=True):
        frame = (
            pl.scan_parquet(path)
            .filter(pl.col("query_name").is_in(sorted(candidate_set)))
            .select(EXPANDED_FIXTURE_COLUMNS)
            .collect(engine="streaming")
        )
        assert frame["query_name"].n_unique() == frame.height
        assert set(frame["species"].unique().to_list()) == {source.species}
        assert set(frame["assembly"].unique().to_list()) == {source.assembly}
        rows_by_source[source.label] = {
            str(row["query_name"]): row for row in frame.to_dicts()
        }

    eligible: list[tuple[str, list[dict[str, object]]]] = []
    selection_rejections: dict[str, int] = {}
    centered_coordinates: dict[tuple[str, str], tuple[int, int]] = {}
    for query_name in candidates:
        rows = [rows_by_source[source.label][query_name] for source in sources]
        reasons = {
            reason
            for row in rows
            if (
                reason := _sequence_rejection(
                    str(row["sequence"]), window_length=selection_window_length
                )
            )
            is not None
        }
        intervals: list[tuple[int, int] | None] = []
        for row in rows:
            intervals.append(
                _centered_interval(
                    start=int(row["pre_resize_t_start"]),
                    end=int(row["pre_resize_t_end"]),
                    window_length=window_length,
                    sequence_length=int(row["t_src_size"]),
                )
            )
        if any(interval is None for interval in intervals):
            reasons.add("expanded_window_out_of_bounds")
        if reasons:
            for reason in reasons:
                selection_rejections[reason] = selection_rejections.get(reason, 0) + 1
            continue
        for source, interval in zip(sources, intervals, strict=True):
            assert interval is not None
            centered_coordinates[(query_name, source.label)] = interval
        eligible.append((query_name, rows))

    assert len(eligible) >= max_anchors, {
        "eligible": len(eligible),
        "requested": max_anchors,
        "candidate_anchors": candidate_anchors,
        "selection_rejections": selection_rejections,
    }

    extracted: dict[tuple[str, str], str] = {}
    with tempfile.TemporaryDirectory(prefix="linclust_twobit_extract_") as directory:
        temporary = Path(directory)
        for source, genome_path in zip(sources, genome_paths, strict=True):
            intervals = []
            for candidate_index, (query_name, rows) in enumerate(eligible):
                row_by_label = {
                    candidate_source.label: row
                    for candidate_source, row in zip(sources, rows, strict=True)
                }
                row = row_by_label[source.label]
                start, end = centered_coordinates[(query_name, source.label)]
                record_id = f"candidate{candidate_index:06d}__{source.label}"
                intervals.append(
                    (
                        record_id,
                        str(row["t_chrom"]),
                        start,
                        end,
                        str(row["t_strand"]),
                    )
                )
            source_sequences = _extract_twobit_intervals(
                genome_path=genome_path,
                intervals=intervals,
                temporary=temporary,
                two_bit_to_fa=two_bit_to_fa,
                window_length=window_length,
            )
            for candidate_index, (query_name, _) in enumerate(eligible):
                record_id = f"candidate{candidate_index:06d}__{source.label}"
                extracted[(query_name, source.label)] = source_sequences[record_id]

    selected: list[tuple[str, list[dict[str, object]]]] = []
    expanded_rejections: dict[str, int] = {}
    embedded_prefix = {query_name for query_name, _ in eligible[:max_anchors]}
    for query_name, rows in eligible:
        reasons = {
            reason
            for source in sources
            if (
                reason := _sequence_rejection(
                    extracted[(query_name, source.label)], window_length=window_length
                )
            )
            is not None
        }
        if reasons:
            for reason in reasons:
                expanded_rejections[reason] = expanded_rejections.get(reason, 0) + 1
            continue
        selected.append((query_name, rows))
        if len(selected) == max_anchors:
            break
    assert len(selected) == max_anchors, {
        "selected": len(selected),
        "requested": max_anchors,
        "expanded_rejections": expanded_rejections,
    }

    fasta_path = Path(fasta_path)
    truth_path = Path(truth_path)
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    truth_fields = [
        "anchor_index",
        "query_name",
        "record_id",
        "source_label",
        "species",
        "assembly",
        "region_label",
        "source_chrom",
        "source_start",
        "source_end",
        "target_chrom",
        "target_start",
        "target_end",
        "target_strand",
    ]
    with fasta_path.open("w") as fasta, truth_path.open("w", newline="") as truth:
        writer = csv.DictWriter(truth, fieldnames=truth_fields, delimiter="\t")
        writer.writeheader()
        for anchor_index, (query_name, rows) in enumerate(selected):
            for source, row in zip(sources, rows, strict=True):
                record_id = f"anchor{anchor_index:06d}__{source.label}"
                sequence = extracted[(query_name, source.label)]
                target_start, target_end = centered_coordinates[
                    (query_name, source.label)
                ]
                fasta.write(f">{record_id}\n{sequence}\n")
                writer.writerow(
                    {
                        "anchor_index": anchor_index,
                        "query_name": query_name,
                        "record_id": record_id,
                        "source_label": source.label,
                        "species": source.species,
                        "assembly": source.assembly,
                        "region_label": row["region_label"],
                        "source_chrom": row["source_chrom"],
                        "source_start": row["source_start"],
                        "source_end": row["source_end"],
                        "target_chrom": row["t_chrom"],
                        "target_start": target_start,
                        "target_end": target_end,
                        "target_strand": row["t_strand"],
                    }
                )

    selected_names = {query_name for query_name, _ in selected}
    return {
        "candidate_anchor_limit": candidate_anchors,
        "common_anchor_count": len(common),
        "eligible_selection_anchor_count": len(eligible),
        "expanded_rejected_candidate_anchor_counts": dict(
            sorted(expanded_rejections.items())
        ),
        "input_sequence_count": len(selected) * len(sources),
        "matched_embedded_prefix_anchor_count": len(selected_names & embedded_prefix),
        "selection_rejected_candidate_anchor_counts": dict(
            sorted(selection_rejections.items())
        ),
        "selected_anchor_count": len(selected),
        "selection_window_length": selection_window_length,
        "source_count": len(sources),
        "truth_sha256": sha256_file(truth_path),
        "fasta_sha256": sha256_file(fasta_path),
        "window_length": window_length,
    }


def download_and_build_projection_fixture(
    *,
    sources: list[ProjectionSource],
    s3_client: Any,
    max_anchors: int,
    candidate_anchors: int,
    window_length: int,
    fasta_path: str | Path,
    truth_path: str | Path,
) -> dict[str, object]:
    """Download exact projection objects, build the fixture, and return a receipt."""
    with tempfile.TemporaryDirectory(
        prefix="linclust_projection_fixture_"
    ) as directory:
        temporary = Path(directory)
        paths: list[Path] = []
        source_receipts: list[dict[str, object]] = []
        for source in sources:
            path = temporary / f"{source.label}.parquet"
            sha256, size_bytes = download_staged_genome(
                receipt={
                    "destination_uri": source.uri,
                    "destination_etag": source.etag,
                    "destination_size_bytes": source.size_bytes,
                },
                destination_path=path,
                s3_client=s3_client,
            )
            paths.append(path)
            source_receipts.append(
                {
                    **asdict(source),
                    "sha256": sha256,
                    "observed_size_bytes": size_bytes,
                }
            )
        receipt = build_projection_fixture(
            sources=sources,
            paths=paths,
            max_anchors=max_anchors,
            candidate_anchors=candidate_anchors,
            window_length=window_length,
            fasta_path=fasta_path,
            truth_path=truth_path,
        )
    receipt["sources"] = source_receipts
    return receipt


def download_and_build_center_expanded_projection_fixture(
    *,
    sources: list[ProjectionSource],
    s3_client: Any,
    max_anchors: int,
    candidate_anchors: int,
    selection_window_length: int,
    window_length: int,
    fasta_path: str | Path,
    truth_path: str | Path,
    two_bit_to_fa: str = "twoBitToFa",
) -> dict[str, object]:
    """Download pinned projections and genomes, then extract longer windows."""
    with tempfile.TemporaryDirectory(
        prefix="linclust_expanded_projection_fixture_"
    ) as directory:
        temporary = Path(directory)
        projection_paths: list[Path] = []
        genome_paths: list[Path] = []
        source_receipts: list[dict[str, object]] = []
        for source in sources:
            assert source.genome_uri is not None
            assert source.genome_etag is not None
            assert source.genome_size_bytes is not None
            projection_path = temporary / f"{source.label}.parquet"
            projection_sha256, projection_size_bytes = download_staged_genome(
                receipt={
                    "destination_uri": source.uri,
                    "destination_etag": source.etag,
                    "destination_size_bytes": source.size_bytes,
                },
                destination_path=projection_path,
                s3_client=s3_client,
            )
            genome_path = temporary / f"{source.label}.2bit"
            genome_sha256, genome_size_bytes = download_staged_genome(
                receipt={
                    "destination_uri": source.genome_uri,
                    "destination_etag": source.genome_etag,
                    "destination_size_bytes": source.genome_size_bytes,
                },
                destination_path=genome_path,
                s3_client=s3_client,
            )
            projection_paths.append(projection_path)
            genome_paths.append(genome_path)
            source_receipts.append(
                {
                    **asdict(source),
                    "projection_sha256": projection_sha256,
                    "observed_projection_size_bytes": projection_size_bytes,
                    "genome_sha256": genome_sha256,
                    "observed_genome_size_bytes": genome_size_bytes,
                }
            )
        receipt = build_center_expanded_projection_fixture(
            sources=sources,
            projection_paths=projection_paths,
            genome_paths=genome_paths,
            max_anchors=max_anchors,
            candidate_anchors=candidate_anchors,
            selection_window_length=selection_window_length,
            window_length=window_length,
            fasta_path=fasta_path,
            truth_path=truth_path,
            two_bit_to_fa=two_bit_to_fa,
        )
    receipt["sources"] = source_receipts
    return receipt


def parse_sources_json(value: str) -> list[ProjectionSource]:
    """Parse a JSON list of immutable source-object declarations."""
    rows = json.loads(value)
    assert isinstance(rows, list)
    return [ProjectionSource.from_dict(row) for row in rows]
