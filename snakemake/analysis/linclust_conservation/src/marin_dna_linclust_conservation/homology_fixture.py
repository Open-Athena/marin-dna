"""Build a bounded positive-control fixture from projected orthologous windows."""

from __future__ import annotations

import csv
import json
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

    @classmethod
    def from_dict(cls, row: dict[str, object]) -> ProjectionSource:
        return cls(
            label=str(row["label"]),
            species=str(row["species"]),
            assembly=str(row["assembly"]),
            uri=str(row["uri"]),
            etag=str(row["etag"]),
            size_bytes=int(row["size_bytes"]),
        )


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


def parse_sources_json(value: str) -> list[ProjectionSource]:
    """Parse a JSON list of immutable source-object declarations."""
    rows = json.loads(value)
    assert isinstance(rows, list)
    return [ProjectionSource.from_dict(row) for row in rows]
