"""File boundaries for the additive RAG workflow.

Coordinates are deduplicated across all source cohorts before projection.
Source memberships, including unused chr18 rows, remain separate durable data.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pyarrow as pa

from marin_dna_vertebrate_projection.projection.chains import read_sizes
from marin_dna_vertebrate_projection.rag.documents import (
    HUMAN,
    REGIONS,
    Locus,
    select_validation,
)
from marin_dna_vertebrate_projection.rag.harness import read_benchmark, write_harness
from marin_dna_vertebrate_projection.rag.requests import ProjectionIdentity
from marin_dna_vertebrate_projection.rag.tables import (
    WindowStore,
    read_catalog,
    read_rows,
    sha256_file,
    write_rows,
    write_training_datasets,
)

CATALOG_SCHEMA = pa.schema(
    [
        ("query_name", pa.string()),
        ("source_chrom", pa.string()),
        ("source_start", pa.int64()),
        ("source_end", pa.int64()),
        ("region_label", pa.string()),
    ]
)
MEMBERSHIP_SCHEMA = pa.schema(
    [
        ("namespace", pa.string()),
        ("source_row_id", pa.string()),
        ("query_name", pa.string()),
        ("source_chrom", pa.string()),
        ("source_start", pa.int64()),
        ("source_end", pa.int64()),
        ("split", pa.string()),
    ]
)


def download_benchmark(spec: Mapping[str, Any], output: str | Path) -> None:
    """Fetch only the immutable, explicitly registered development file."""
    if spec["split"] != "train" or not spec["url"].endswith(
        f"/{spec['revision']}/train.parquet"
    ):
        raise ValueError("benchmark URL must pin the official development split")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".partial")
    urllib.request.urlretrieve(spec["url"], temporary)
    if sha256_file(temporary) != spec["sha256"]:
        raise ValueError("downloaded benchmark SHA-256 mismatch")
    temporary.replace(target)


def load_sources(
    catalog_paths: Mapping[str, str],
    benchmark_paths: Mapping[str, str],
    rag: Mapping[str, Any],
) -> tuple[dict[str, dict[Locus, str]], dict[str, list[dict[str, Any]]]]:
    if set(catalog_paths) != set(REGIONS) or set(catalog_paths) != set(rag["catalogs"]):
        raise ValueError("RAG training requires the five declared source regions")
    catalogs = {
        name: read_catalog(path, rag["catalogs"][name])
        for name, path in catalog_paths.items()
    }
    benchmarks = {
        name: read_benchmark(path, rag["benchmarks"][name])
        for name, path in benchmark_paths.items()
    }
    return catalogs, benchmarks


def compile_requests(
    catalog_paths: Mapping[str, str],
    benchmark_paths: Mapping[str, str],
    rag: Mapping[str, Any],
    chains: Mapping[str, Mapping[str, str]],
    genomes: Mapping[str, Mapping[str, str]],
    sizes_path: str,
    catalog_output: str,
    memberships_output: str,
    audit_output: str,
) -> None:
    catalogs, benchmarks = load_sources(catalog_paths, benchmark_paths, rag)
    split = select_validation(
        catalogs, validation_rows=rag["validation_rows"], seed=rag["seed"]
    )
    memberships = []
    loci: set[Locus] = set()
    sizes = read_sizes(sizes_path)
    for region, source_ids in sorted(catalogs.items()):
        for label, selected in (
            ("train", split.train[region]),
            ("validation", split.validation[region]),
            ("excluded_chr18", split.excluded[region]),
        ):
            for locus in selected:
                locus.validate_bounds(sizes)
                if label != "excluded_chr18":
                    loci.add(locus)
                memberships.append(
                    {
                        "namespace": region,
                        "source_row_id": source_ids[locus],
                        "query_name": locus.query_name,
                        "source_chrom": locus.chrom,
                        "source_start": locus.start,
                        "source_end": locus.end,
                        "split": label,
                    }
                )
    for benchmark, records in sorted(benchmarks.items()):
        for record in records:
            locus = record["locus"]
            locus.validate_bounds(sizes)
            loci.add(locus)
            memberships.append(
                {
                    "namespace": benchmark,
                    "source_row_id": record["source_row_id"],
                    "query_name": locus.query_name,
                    "source_chrom": locus.chrom,
                    "source_start": locus.start,
                    "source_end": locus.end,
                    "split": "development",
                }
            )
    write_rows(memberships_output, memberships, MEMBERSHIP_SCHEMA)
    write_rows(
        catalog_output,
        (
            {
                "query_name": locus.query_name,
                "source_chrom": locus.chrom,
                "source_start": locus.start,
                "source_end": locus.end,
                "region_label": "rag_union",
            }
            for locus in sorted(loci)
        ),
        CATALOG_SCHEMA,
    )
    identities = {
        name: ProjectionIdentity(
            species=name,
            assembly=row["assembly"],
            chain_sha256=row["chain_sha256"],
            source_sizes_sha256=row["source_sizes_sha256"],
            target_sizes_sha256=row["target_sizes_sha256"],
            genome_sha256=genomes[name]["sha256"],
        ).fingerprint
        for name, row in chains.items()
    }
    Path(audit_output).write_text(
        json.dumps(
            {
                "unique_requests": len(loci),
                "source_memberships": len(memberships),
                "benchmark_rows": {
                    name: len(rows) for name, rows in benchmarks.items()
                },
                "projection_fingerprints": identities,
                "reused_queries": 0,
                "reuse_decision": "No archived sequence cache has the identical -multiple chain contract and complete validated provenance; reuse pinned genomes/chains and project the exact union once.",
            },
            indent=2,
        )
        + "\n"
    )


def assemble_outputs(
    catalog_paths: Mapping[str, str],
    benchmark_paths: Mapping[str, str],
    rag: Mapping[str, Any],
    union_path: str,
    human_path: str,
    sequence_paths: Mapping[str, str],
    rejection_paths: Mapping[str, str],
    sequence_rejection_paths: Mapping[str, str],
    directory: str,
    harness_output: str,
) -> None:
    catalogs, benchmarks = load_sources(catalog_paths, benchmark_paths, rag)
    expected = {
        Locus(row["source_chrom"], row["source_start"], row["source_end"])
        for row in read_rows(union_path)
    }
    if not set(sequence_paths) == set(rejection_paths) == set(sequence_rejection_paths):
        raise ValueError("species outcome files must match exactly")
    Path(directory).mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".rag-windows-", dir=directory) as temporary:
        store = WindowStore(Path(temporary) / "windows.sqlite")
        try:
            count = store.ingest(
                human_path, species=HUMAN, accepted=True, expected=expected
            )
            if count != len(expected):
                raise ValueError("incomplete human sequence accounting")
            for name, path in sorted(sequence_paths.items()):
                count = store.ingest(
                    path, species=name, accepted=True, expected=expected
                )
                count += store.ingest(
                    rejection_paths[name],
                    species=name,
                    accepted=False,
                    expected=expected,
                )
                count += store.ingest(
                    sequence_rejection_paths[name],
                    species=name,
                    accepted=False,
                    expected=expected,
                )
                if count != len(expected):
                    raise ValueError(f"incomplete accepted/rejected accounting: {name}")
            species = {HUMAN, *sequence_paths}
            write_training_datasets(
                catalogs,
                store,
                species=species,
                directory=directory,
                validation_rows=rag["validation_rows"],
                seed=rag["seed"],
            )
            write_harness(
                benchmarks,
                store,
                species=species,
                output=harness_output,
                seed=rag["seed"],
            )
        finally:
            store.close()
