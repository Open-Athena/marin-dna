from __future__ import annotations

import json

import polars as pl
import pytest
from marin_dna_linclust_conservation.manifest import (
    parse_genome_records,
    parse_taxonomy_orders,
    select_order_representatives,
)


def _genome(
    accession: str,
    tax_id: int,
    species: str,
    level: str,
    n50: int,
    length: int,
) -> dict[str, object]:
    return {
        "accession": accession,
        "annotation_info": {"name": f"annotation-{accession}"},
        "assembly_info": {
            "assembly_level": level,
            "assembly_name": f"assembly-{accession}",
            "assembly_status": "current",
            "release_date": "2026-01-01",
        },
        "assembly_stats": {
            "contig_n50": n50,
            "total_sequence_length": str(length),
        },
        "organism": {"organism_name": species, "tax_id": tax_id},
        "source_database": "SOURCE_DATABASE_REFSEQ",
    }


def test_parse_taxonomy_orders() -> None:
    records = [
        {
            "taxonomy": {
                "tax_id": 1,
                "classification": {"order": {"name": "Order one"}},
            }
        }
    ]
    assert parse_taxonomy_orders(records) == {1: "Order one"}


def test_selection_forces_human_mouse_then_ranks_level_and_n50() -> None:
    records = [
        _genome("GCF_human", 9606, "Homo sapiens", "Chromosome", 10, 100),
        _genome("GCF_primate", 2, "Other primate", "Complete Genome", 999, 999),
        _genome("GCF_mouse", 10090, "Mus musculus", "Chromosome", 10, 100),
        _genome("GCF_rodent", 4, "Other rodent", "Complete Genome", 999, 999),
        _genome("GCF_complete", 5, "Complete", "Complete Genome", 10, 100),
        _genome("GCF_chromosome", 6, "Chromosome", "Chromosome", 999, 999),
        _genome("GCF_n50_low", 7, "N50 low", "Chromosome", 20, 200),
        _genome("GCF_n50_high", 8, "N50 high", "Chromosome", 30, 100),
        _genome("GCF_scaffold", 9, "Scaffold", "Scaffold", 1_000, 1_000),
    ]
    orders = {
        9606: "Primates",
        2: "Primates",
        10090: "Rodentia",
        4: "Rodentia",
        5: "Shared order",
        6: "Shared order",
        7: "N50 order",
        8: "N50 order",
        9: "Fallback order",
    }
    candidates = parse_genome_records(
        records,
        taxonomy_orders=orders,
        datasets_version="18.36.0",
        retrieved_at="2026-08-25T00:00:00Z",
    )
    result = select_order_representatives(candidates)
    chosen = set(result.filter(pl.col("selected"))["accession"].to_list())
    assert chosen == {
        "GCF_human",
        "GCF_mouse",
        "GCF_complete",
        "GCF_n50_high",
    }
    scaffold = result.filter(pl.col("accession") == "GCF_scaffold")
    assert scaffold["selection_reason"].item() == (
        "fallback required: no complete or chromosome assembly"
    )
    assert scaffold["fallback_needed"].item()
    assert not result.filter(pl.col("selected"))["fallback_needed"].any()


def test_parse_rejects_unannotated_or_non_refseq() -> None:
    record = _genome("GCA_bad", 1, "Bad", "Chromosome", 10, 100)
    with pytest.raises(AssertionError, match="RefSeq"):
        parse_genome_records(
            [record],
            taxonomy_orders={1: "Order"},
            datasets_version="18.36.0",
            retrieved_at="2026-08-25T00:00:00Z",
        )
    json.dumps(record)
