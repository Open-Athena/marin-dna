"""Combined development harness with exact canonical row identities."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa

from marin_dna_vertebrate_projection.rag.documents import (
    Locus,
    allele_documents,
    stable_digest,
)
from marin_dna_vertebrate_projection.rag.tables import (
    WindowStore,
    read_rows,
    sha256_file,
    write_rows,
)

BENCHMARKS = {"mendelian_traits", "complex_traits", "sge"}
DEVELOPMENT_CHROMS = {f"chr{number}" for number in range(1, 23, 2)} | {"chrX"}


def read_benchmark(path: str | Path, spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Read a pinned official train split, converting its 1-based SNV boundary.

    The declared chromosome convention is checked rather than guessed. Original
    row positions and complete canonical metadata survive harness construction.
    No held-out rows may enter this development harness.
    """
    name = spec["name"]
    if name not in BENCHMARKS or spec["split"] != "train":
        raise ValueError(
            "only the three registered development benchmarks are authorized"
        )
    if sha256_file(path) != spec["sha256"]:
        raise ValueError("canonical benchmark SHA-256 mismatch")
    naming = spec["chrom_names"]
    if naming not in {"ucsc", "ensembl"}:
        raise ValueError("benchmark chromosome naming must be explicit")
    records = []
    for index, metadata in enumerate(read_rows(path)):
        chrom = str(metadata["chrom"])
        if naming == "ensembl":
            if chrom.startswith("chr"):
                raise ValueError(
                    "bare benchmark chromosome adapter received a UCSC name"
                )
            chrom = "chr" + chrom
        if chrom not in DEVELOPMENT_CHROMS:
            raise ValueError("held-out chromosome in the declared development input")
        pos = metadata["pos"]
        if type(pos) is not int or pos < 1:
            raise ValueError(
                "canonical VEP position must be a 1-based positive integer"
            )
        ref, alt = metadata["ref"], metadata["alt"]
        if (
            ref not in {"A", "C", "G", "T"}
            or alt not in {"A", "C", "G", "T"}
            or ref == alt
        ):
            raise ValueError("canonical RAG cohort must contain distinct REF/ALT SNVs")
        locus = Locus(chrom, pos - 1 - 127, pos - 1 + 128)
        records.append(
            {
                "benchmark": name,
                "source_row_index": index,
                "source_row_id": stable_digest(name, spec["revision"], "train", index),
                "locus": locus,
                "ref": ref,
                "alt": alt,
                "metadata": metadata,
            }
        )
    if len(records) != spec["expected_rows"]:
        raise ValueError("canonical source cohort row count changed")
    return records


HARNESS_SCHEMA = pa.schema(
    [
        ("benchmark", pa.string()),
        ("source_row_index", pa.int64()),
        ("source_row_id", pa.string()),
        ("query_name", pa.string()),
        ("source_chrom", pa.string()),
        ("source_start", pa.int64()),
        ("source_end", pa.int64()),
        ("split", pa.string()),
        ("ref", pa.string()),
        ("alt", pa.string()),
        ("sequence", pa.string()),
        ("species_order", pa.list_(pa.string())),
        ("group_id", pa.string()),
        ("accession_id", pa.string()),
        ("metadata_json", pa.string()),
    ]
)


def write_harness(
    benchmarks: Mapping[str, list[dict[str, Any]]],
    store: WindowStore,
    *,
    species: set[str],
    output: str | Path,
    seed: int = 42,
) -> dict[str, int]:
    """Store one reference document per canonical row; alleles/strands derive from it."""
    if set(benchmarks) != BENCHMARKS:
        raise ValueError(
            "combined harness requires exactly the three benchmark cohorts"
        )
    counts = {name: len(records) for name, records in benchmarks.items()}

    def rows() -> Iterator[dict[str, Any]]:
        for name, records in sorted(benchmarks.items()):
            identities = set()
            for record in records:
                if record["benchmark"] != name or record["source_row_id"] in identities:
                    raise ValueError(
                        "duplicate or incorrectly namespaced source identity"
                    )
                identities.add(record["source_row_id"])
                locus = record["locus"]
                windows, _ = store.get(locus, species)
                # This asserts REF and equal retrieval/permutation for all inputs.
                document = allele_documents(
                    windows,
                    locus=locus,
                    ref=record["ref"],
                    alt=record["alt"],
                    seed=seed,
                )["ref_forward"]
                metadata = record["metadata"]
                yield {
                    "benchmark": name,
                    "source_row_index": record["source_row_index"],
                    "source_row_id": record["source_row_id"],
                    "query_name": locus.query_name,
                    "source_chrom": locus.chrom,
                    "source_start": locus.start,
                    "source_end": locus.end,
                    "split": "train",
                    "ref": record["ref"],
                    "alt": record["alt"],
                    "sequence": document.sequence,
                    "species_order": list(document.species_order),
                    "group_id": stable_digest(
                        name, "match_group", str(metadata["match_group"])
                    )
                    if "match_group" in metadata
                    else None,
                    "accession_id": stable_digest(
                        name, "accession", str(metadata["mavedb_urn"])
                    )
                    if "mavedb_urn" in metadata
                    else None,
                    "metadata_json": json.dumps(
                        metadata, sort_keys=True, allow_nan=False
                    ),
                }

    written = write_rows(output, rows(), HARNESS_SCHEMA)
    if written != sum(counts.values()):
        raise ValueError("joint harness lost canonical benchmark rows")
    Path(output).with_suffix(".counts.json").write_text(
        json.dumps(counts, indent=2) + "\n"
    )
    return counts
