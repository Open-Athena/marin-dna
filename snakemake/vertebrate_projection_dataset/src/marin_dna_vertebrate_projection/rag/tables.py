"""Bounded-batch Parquet adapters for RAG requests and auditable documents."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from marin_dna_vertebrate_projection.rag.documents import (
    HUMAN,
    Locus,
    assemble_document,
    select_validation,
    validate_sequence,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(
    path: str | Path, columns: list[str] | None = None
) -> Iterator[dict[str, Any]]:
    """Read at most one Parquet batch at a time; TSV inputs support fixtures."""
    path = Path(path)
    if path.suffix == ".parquet":
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=4096, columns=columns
        ):
            yield from batch.to_pylist()
    else:
        with path.open() as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                yield (
                    row
                    if columns is None
                    else {column: row[column] for column in columns}
                )


def write_rows(
    path: str | Path, rows: Iterable[dict[str, Any]], schema: pa.Schema
) -> int:
    """Write deterministic Parquet batches, including schema-valid empty outputs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    batch = []
    with pq.ParquetWriter(path, schema, compression="zstd") as writer:
        for row in rows:
            batch.append(row)
            if len(batch) == 4096:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                count += len(batch)
                batch = []
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
            count += len(batch)
    return count


def coordinate_int(value: Any) -> int:
    if type(value) is int:
        return value
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
        return int(value)
    raise ValueError("genomic coordinates require exact integers")


def read_catalog(path: str | Path, spec: Mapping[str, Any]) -> dict[Locus, str]:
    """Select exact source memberships with an explicit coordinate-name adapter.

    Specs map ``id``, ``chrom``, ``start``, ``end``, and ``region`` to source
    columns and supply ``region_value``. ``chrom_names`` explicitly declares
    UCSC names or the source's bare/Ensembl names; numeric positions are already
    0-based half-open in both accepted anchor sources.
    """
    if sha256_file(path) != spec["sha256"]:
        raise ValueError("source anchor SHA-256 mismatch")
    fields = spec["columns"]
    naming = spec["chrom_names"]
    if naming not in {"ucsc", "ensembl"}:
        raise ValueError("anchor chromosome-name convention must be explicit")
    result: dict[Locus, str] = {}
    ids: set[str] = set()
    for row in read_rows(path, list(fields.values())):
        if row[fields["region"]] != spec["region_value"]:
            continue
        chrom = str(row[fields["chrom"]])
        if naming == "ensembl":
            if chrom.startswith("chr"):
                raise ValueError("bare chromosome adapter received a UCSC name")
            chrom = "chr" + chrom
        locus = Locus(
            chrom,
            coordinate_int(row[fields["start"]]),
            coordinate_int(row[fields["end"]]),
        )
        source_id = str(row[fields["id"]])
        if locus in result or source_id in ids or not source_id:
            raise ValueError("duplicate or empty source anchor identity")
        result[locus] = source_id
        ids.add(source_id)
    if len(result) != spec["expected_rows"]:
        raise ValueError(
            f"source catalog count mismatch: {len(result)} != {spec['expected_rows']}"
        )
    return result


class WindowStore:
    """Disk-backed sequence/provenance lookup; no all-species sequence group-by.

    One accepted sequence or explicit rejection must exist for every requested
    locus/species pair. Missing files and incomplete producers are errors, not
    biological missingness. The primary key detects overlapping cache inputs.
    """

    def __init__(self, path: str | Path) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA cache_size=-32768")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute(
            "CREATE TABLE windows(query_name TEXT, species TEXT, sequence TEXT, provenance TEXT NOT NULL, "
            "PRIMARY KEY(query_name,species)) WITHOUT ROWID"
        )

    def close(self) -> None:
        self.connection.close()

    def ingest(
        self,
        path: str | Path,
        *,
        species: str,
        accepted: bool,
        expected: set[Locus],
    ) -> int:
        count = 0
        with self.connection:
            for row in read_rows(path):
                locus = Locus(
                    row["source_chrom"], row["source_start"], row["source_end"]
                )
                if locus not in expected:
                    raise ValueError(
                        "sequence producer contains an unrequested source locus"
                    )
                sequence = row.get("sequence") if accepted else None
                if accepted:
                    validate_sequence(sequence)
                    if row.get("alignment_name") != species:
                        raise ValueError(
                            "sequence species differs from its registered input"
                        )
                    if row["t_end"] - row["t_start"] != 255 or row["t_strand"] not in {
                        "+",
                        "-",
                    }:
                        raise ValueError("invalid projected sequence geometry")
                    if not 0 <= row["t_start"] < row["t_end"] <= row["t_src_size"]:
                        raise ValueError("projected sequence exceeds target bounds")
                    if species != HUMAN and (
                        row["pre_resize_t_end"] - row["pre_resize_t_start"] != 1
                        or row["pre_resize_t_start"] != row["t_start"] + 127
                    ):
                        raise ValueError("mapped center is not at sequence index 127")
                elif not row.get("rejection_reason"):
                    raise ValueError(
                        "unavailable species requires an explicit rejection reason"
                    )
                metadata = {
                    key: value for key, value in row.items() if key != "sequence"
                }
                try:
                    self.connection.execute(
                        "INSERT INTO windows VALUES (?,?,?,?)",
                        (
                            locus.query_name,
                            species,
                            sequence,
                            json.dumps(metadata, sort_keys=True),
                        ),
                    )
                except sqlite3.IntegrityError as error:
                    raise ValueError(
                        "duplicate locus/species outcome in sequence inputs"
                    ) from error
                count += 1
        return count

    def get(
        self, locus: Locus, species: set[str]
    ) -> tuple[dict[str, str], dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT species,sequence,provenance FROM windows WHERE query_name=? ORDER BY species",
            (locus.query_name,),
        ).fetchall()
        if {row[0] for row in rows} != species:
            raise ValueError("incomplete species outcome accounting for a source locus")
        available = {
            name: sequence for name, sequence, _ in rows if sequence is not None
        }
        if HUMAN not in available:
            raise ValueError("human reference sequence is unavailable")
        return available, {name: json.loads(provenance) for name, _, provenance in rows}


TRAIN_SCHEMA = pa.schema(
    [
        ("sequence", pa.string()),
        ("query_name", pa.string()),
        ("source_row_id", pa.string()),
        ("source_chrom", pa.string()),
        ("source_start", pa.int64()),
        ("source_end", pa.int64()),
        ("orientation", pa.string()),
        ("species_order", pa.list_(pa.string())),
        ("unpadded_tokens", pa.int64()),
        ("padding_tokens", pa.int64()),
    ]
)


def _training_documents(
    loci: Iterable[Locus],
    orientations: tuple[str, ...],
    store: WindowStore,
    species: set[str],
    source_ids: Mapping[Locus, str],
    seed: int,
) -> Iterator[dict[str, Any]]:
    for locus in loci:
        windows, _ = store.get(locus, species)
        for orientation in orientations:
            document = assemble_document(
                windows, locus=locus, seed=seed, orientation=orientation
            )
            yield {
                "sequence": document.sequence,
                "query_name": locus.query_name,
                "source_row_id": source_ids[locus],
                "source_chrom": locus.chrom,
                "source_start": locus.start,
                "source_end": locus.end,
                "orientation": orientation,
                "species_order": list(document.species_order),
                "unpadded_tokens": document.unpadded_tokens,
                "padding_tokens": document.padding_tokens,
            }


def write_training_datasets(
    catalogs: Mapping[str, Mapping[Locus, str]],
    store: WindowStore,
    *,
    species: set[str],
    directory: str | Path,
    validation_rows: int = 400,
    seed: int = 42,
) -> dict[str, Any]:
    """Write internal documents after the chr18 split; publication strips columns."""
    directory = Path(directory)
    plan = select_validation(catalogs, validation_rows=validation_rows, seed=seed)
    summary: dict[str, Any] = {
        "validation_chromosome": "chr18",
        "validation_seed": seed,
        "exact_cross_region_duplicate_memberships": plan.exact_cross_region_duplicates,
        "cross_region_duplicates": "retain original region membership outside chr18",
        "regions": {},
    }
    for region, source_ids in sorted(catalogs.items()):
        region_summary = {
            "source_anchors": len(source_ids),
            "unused_chr18_anchors": len(plan.excluded[region]),
        }
        for split, loci, orientations in (
            ("train", plan.train[region], ("forward", "rc")),
            ("validation", plan.validation[region], ("forward",)),
        ):
            count = write_rows(
                directory / region / f"{split}.parquet",
                _training_documents(
                    loci, orientations, store, species, source_ids, seed
                ),
                TRAIN_SCHEMA,
            )
            if count != len(loci) * len(orientations):
                raise ValueError("document count differs from split plan")
            region_summary[f"{split}_rows"] = count
            region_summary[f"{split}_allocated_tokens"] = count * 10240
        summary["regions"][region] = region_summary
    (directory / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
