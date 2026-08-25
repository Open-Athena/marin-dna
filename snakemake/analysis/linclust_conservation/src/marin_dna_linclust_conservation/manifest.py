"""Resolve and validate an order-deduplicated RefSeq mammal manifest."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

ASSEMBLY_LEVEL_RANK: dict[str, int] = {
    "Complete Genome": 0,
    "Chromosome": 1,
    "Scaffold": 2,
    "Contig": 3,
}
ELIGIBLE_ASSEMBLY_LEVELS = frozenset({"Complete Genome", "Chromosome"})
HUMAN_TAX_ID = 9606
MOUSE_TAX_ID = 10090
FORCED_TAX_IDS = frozenset({HUMAN_TAX_ID, MOUSE_TAX_ID})

MANIFEST_COLUMNS: tuple[str, ...] = (
    "accession",
    "tax_id",
    "species",
    "order",
    "assembly_name",
    "assembly_level",
    "contig_n50",
    "total_length",
    "release_date",
    "annotation_name",
    "source_database",
    "datasets_version",
    "retrieved_at",
    "download_uri",
    "source_checksum_type",
    "source_checksum",
    "source_size_bytes",
    "sequence_sha256",
    "fallback_needed",
    "selected",
    "selection_reason",
)


def _read_json_lines(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            assert isinstance(record, dict), f"{path}:{line_number}: expected object"
            records.append(record)
    return records


def parse_taxonomy_orders(
    records: Iterable[Mapping[str, Any]],
) -> dict[int, str]:
    """Return taxon ID to NCBI order from taxonomy JSONL records."""
    orders: dict[int, str] = {}
    for record in records:
        taxonomy = record.get("taxonomy")
        assert isinstance(taxonomy, Mapping), "taxonomy record lacks taxonomy"
        tax_id = int(taxonomy["tax_id"])
        classification = taxonomy.get("classification")
        assert isinstance(classification, Mapping), (
            f"taxon {tax_id} lacks classification"
        )
        order = classification.get("order")
        assert isinstance(order, Mapping) and order.get("name"), (
            f"taxon {tax_id} lacks an NCBI order"
        )
        order_name = str(order["name"])
        if tax_id in orders:
            assert orders[tax_id] == order_name, f"conflicting order for taxon {tax_id}"
        orders[tax_id] = order_name
    return orders


def parse_genome_records(
    records: Iterable[Mapping[str, Any]],
    *,
    taxonomy_orders: Mapping[int, str],
    datasets_version: str,
    retrieved_at: str,
) -> pl.DataFrame:
    """Normalize NCBI genome JSONL records into deterministic candidate rows."""
    parsed_time = datetime.fromisoformat(retrieved_at)
    assert parsed_time.tzinfo is not None, "retrieved_at must include a timezone"
    assert datasets_version, "datasets_version must be non-empty"

    rows: list[dict[str, object]] = []
    for record in records:
        assembly_info = record.get("assembly_info")
        assembly_stats = record.get("assembly_stats")
        organism = record.get("organism")
        annotation_info = record.get("annotation_info")
        assert isinstance(assembly_info, Mapping), "genome record lacks assembly_info"
        assert isinstance(assembly_stats, Mapping), "genome record lacks assembly_stats"
        assert isinstance(organism, Mapping), "genome record lacks organism"
        assert isinstance(annotation_info, Mapping), "genome record is not annotated"

        accession = str(record["accession"])
        tax_id = int(organism["tax_id"])
        assert accession.startswith("GCF_"), f"{accession}: expected RefSeq accession"
        assert tax_id in taxonomy_orders, f"{accession}: missing taxonomy order"
        assert assembly_info.get("assembly_status") == "current", (
            f"{accession}: assembly is not current"
        )

        rows.append(
            {
                "accession": accession,
                "tax_id": tax_id,
                "species": str(organism["organism_name"]),
                "order": str(taxonomy_orders[tax_id]),
                "assembly_name": str(assembly_info["assembly_name"]),
                "assembly_level": str(assembly_info["assembly_level"]),
                "contig_n50": int(assembly_stats["contig_n50"]),
                "total_length": int(assembly_stats["total_sequence_length"]),
                "release_date": str(assembly_info["release_date"]),
                "annotation_name": str(annotation_info["name"]),
                "source_database": str(record["source_database"]),
                "datasets_version": datasets_version,
                "retrieved_at": parsed_time.astimezone(UTC)
                .isoformat()
                .replace("+00:00", "Z"),
                "download_uri": "",
                "source_checksum_type": "",
                "source_checksum": "",
                "source_size_bytes": None,
                "sequence_sha256": "",
            }
        )
    assert rows, "NCBI query returned no genome records"
    candidates = pl.DataFrame(rows)
    assert candidates["accession"].n_unique() == candidates.height
    return candidates


def parse_ncbi_jsonl(
    genome_path: str | Path,
    taxonomy_path: str | Path,
    *,
    datasets_version: str,
    retrieved_at: str,
) -> pl.DataFrame:
    """Read NCBI genome and taxonomy JSONL files into normalized candidates."""
    taxonomy_orders = parse_taxonomy_orders(_read_json_lines(taxonomy_path))
    return parse_genome_records(
        _read_json_lines(genome_path),
        taxonomy_orders=taxonomy_orders,
        datasets_version=datasets_version,
        retrieved_at=retrieved_at,
    )


def select_order_representatives(candidates: pl.DataFrame) -> pl.DataFrame:
    """Select one eligible assembly per NCBI order, forcing human and mouse."""
    required = {
        "accession",
        "tax_id",
        "species",
        "order",
        "assembly_level",
        "contig_n50",
        "total_length",
    }
    assert required.issubset(candidates.columns), sorted(
        required - set(candidates.columns)
    )
    assert candidates["accession"].n_unique() == candidates.height
    assert candidates["tax_id"].is_not_null().all()
    assert candidates["order"].is_not_null().all()
    assert (candidates["contig_n50"] > 0).all()
    assert (candidates["total_length"] > 0).all()

    unknown_levels = set(candidates["assembly_level"].unique()) - set(
        ASSEMBLY_LEVEL_RANK
    )
    assert not unknown_levels, f"unknown assembly levels: {sorted(unknown_levels)}"

    eligible = candidates.filter(
        pl.col("assembly_level").is_in(sorted(ELIGIBLE_ASSEMBLY_LEVELS))
    ).with_columns(
        pl.col("assembly_level")
        .replace_strict(ASSEMBLY_LEVEL_RANK, return_dtype=pl.Int8)
        .alias("_assembly_rank"),
        pl.col("tax_id").is_in(sorted(FORCED_TAX_IDS)).alias("_forced"),
    )

    forced = eligible.filter(pl.col("_forced"))
    missing_forced = FORCED_TAX_IDS - set(forced["tax_id"].to_list())
    assert not missing_forced, (
        f"forced taxa lack an eligible assembly: {missing_forced}"
    )
    for tax_id in FORCED_TAX_IDS:
        assert forced.filter(pl.col("tax_id") == tax_id).height == 1, (
            f"expected one current eligible assembly for forced taxon {tax_id}"
        )

    ranked = eligible.sort(
        by=[
            "order",
            "_forced",
            "_assembly_rank",
            "contig_n50",
            "total_length",
            "species",
            "accession",
        ],
        descending=[False, True, False, True, True, False, False],
    )
    selected_accessions = set(
        ranked.group_by("order", maintain_order=True).first()["accession"].to_list()
    )
    orders_without_eligible_assemblies = set(candidates["order"].to_list()) - set(
        eligible["order"].to_list()
    )
    result = candidates.with_columns(
        pl.col("order")
        .is_in(sorted(orders_without_eligible_assemblies))
        .alias("fallback_needed"),
        pl.col("accession").is_in(sorted(selected_accessions)).alias("selected"),
        pl.when(pl.col("accession").is_in(sorted(selected_accessions)))
        .then(
            pl.when(pl.col("tax_id").is_in(sorted(FORCED_TAX_IDS)))
            .then(pl.lit("forced human or mouse"))
            .otherwise(pl.lit("best eligible assembly for order"))
        )
        .when(pl.col("order").is_in(sorted(orders_without_eligible_assemblies)))
        .then(pl.lit("fallback required: no complete or chromosome assembly"))
        .when(pl.col("assembly_level").is_in(sorted(ELIGIBLE_ASSEMBLY_LEVELS)).not_())
        .then(pl.lit("ineligible assembly level"))
        .otherwise(pl.lit("lower-ranked eligible assembly for order"))
        .alias("selection_reason"),
    ).sort(
        ["order", "selected", "species", "accession"],
        descending=[False, True, False, False],
    )

    selected = result.filter(pl.col("selected"))
    assert selected["order"].n_unique() == selected.height
    assert FORCED_TAX_IDS.issubset(set(selected["tax_id"].to_list()))
    return result


def validate_pinned_manifest(manifest: pl.DataFrame) -> None:
    """Fail if a selected manifest lacks reproducible source metadata."""
    missing_columns = set(MANIFEST_COLUMNS) - set(manifest.columns)
    assert not missing_columns, f"manifest lacks columns: {sorted(missing_columns)}"
    selected = manifest.filter(pl.col("selected"))
    assert selected.height > 0
    assert selected["order"].n_unique() == selected.height
    assert selected["accession"].n_unique() == selected.height
    assert FORCED_TAX_IDS.issubset(set(selected["tax_id"].to_list()))
    assert set(selected["assembly_level"].unique()).issubset(ELIGIBLE_ASSEMBLY_LEVELS)
    for column in (
        "datasets_version",
        "retrieved_at",
        "download_uri",
        "source_checksum_type",
        "source_checksum",
    ):
        assert selected[column].is_not_null().all(), f"{column} contains nulls"
        assert selected[column].str.len_chars().min() > 0, f"{column} contains blanks"
    assert selected["source_size_bytes"].is_not_null().all()
    assert (selected["source_size_bytes"] > 0).all()
    assert selected["fallback_needed"].not_().all()
    sha_values = selected.filter(
        pl.col("sequence_sha256").fill_null("").str.len_chars() > 0
    )["sequence_sha256"]
    assert sha_values.len() in (0, selected.height), (
        "sequence_sha256 must be empty for all selected rows before staging or set for all"
    )
    if sha_values.len():
        assert sha_values.str.contains(r"^[0-9a-f]{64}$").all()
