"""Testable dataframe transforms for the genome-selection pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

TAXONOMIC_RANKS: tuple[str, ...] = (
    "species",
    "genus",
    "family",
    "order",
    "class",
    "phylum",
    "kingdom",
    "domain",
)

ASSEMBLY_LEVELS: tuple[str, ...] = (
    "Complete Genome",
    "Chromosome",
    "Scaffold",
    "Contig",
)


def extract_tax_ids(input_path: str | Path, output_path: str | Path) -> None:
    """Write unique taxonomy IDs from the NCBI genome table."""
    genomes = pd.read_csv(input_path, sep="\t", usecols=["Organism Taxonomic ID"])
    assert genomes["Organism Taxonomic ID"].notna().all()
    genomes.drop_duplicates().to_csv(output_path, index=False, header=False)


def add_taxonomy(
    genomes: pd.DataFrame,
    taxonomy_records: pd.DataFrame,
) -> pd.DataFrame:
    """Join NCBI taxonomy classifications and expand canonical rank columns."""
    assert "Organism Taxonomic ID" in genomes.columns
    assert "taxonomy" in taxonomy_records.columns

    taxonomy = taxonomy_records[["taxonomy"]].copy()
    taxonomy["Organism Taxonomic ID"] = taxonomy["taxonomy"].map(
        lambda value: value["tax_id"]
    )
    taxonomy["classification"] = taxonomy["taxonomy"].map(
        lambda value: value["classification"]
    )
    taxonomy = taxonomy.drop(columns=["taxonomy"])
    assert not taxonomy["Organism Taxonomic ID"].duplicated().any()

    result = genomes.merge(taxonomy, on="Organism Taxonomic ID", how="left")

    def get_rank(classification: Any, rank: str) -> str | None:
        if not isinstance(classification, dict):
            return None
        rank_value = classification.get(rank)
        if not isinstance(rank_value, dict):
            return None
        name = rank_value.get("name")
        return str(name) if name is not None else None

    for rank in TAXONOMIC_RANKS:
        result[rank] = result["classification"].map(
            lambda classification, rank=rank: get_rank(classification, rank)
        )
    return result.drop(columns=["classification"])


def add_taxonomy_to_genomes(
    genomes_path: str | Path,
    taxonomy_path: str | Path,
    output_path: str | Path,
) -> None:
    """Read, annotate, and write the genome table."""
    genomes = pd.read_csv(genomes_path, sep="\t")
    taxonomy = pd.read_json(taxonomy_path, lines=True)
    add_taxonomy(genomes, taxonomy).to_parquet(output_path, index=False)


def filter_genomes(
    genomes: pd.DataFrame,
    *,
    exclude_genomes: list[str],
    deduplicate_taxonomic_rank: str,
    min_assembly_level: str,
    max_genome_size: int,
    priority_genomes: list[str],
) -> pd.DataFrame:
    """Apply the configured quality filters and keep one assembly per taxon."""
    required = {
        "Assembly Accession",
        "Assembly Level",
        "Assembly Stats Total Sequence Length",
        "Organism Name",
        deduplicate_taxonomic_rank,
    }
    assert required.issubset(genomes.columns), sorted(required - set(genomes.columns))
    assert min_assembly_level in ASSEMBLY_LEVELS
    assert max_genome_size > 0

    result = genomes.loc[~genomes["Assembly Accession"].isin(exclude_genomes)].copy()
    result = result.dropna(subset=[deduplicate_taxonomic_rank])
    result["Assembly Level"] = pd.Categorical(
        result["Assembly Level"], ASSEMBLY_LEVELS, ordered=True
    )
    result = result.loc[result["Assembly Level"] <= min_assembly_level]
    result = result.loc[
        result["Assembly Stats Total Sequence Length"] < max_genome_size
    ]
    result["Priority"] = "1_Low"
    result.loc[result["Assembly Accession"].isin(priority_genomes), "Priority"] = (
        "0_High"
    )
    result = (
        result.sort_values(
            [
                "Priority",
                "Assembly Level",
                "Assembly Stats Total Sequence Length",
                "Organism Name",
            ]
        )
        .drop(columns=["Priority"])
        .drop_duplicates(deduplicate_taxonomic_rank)
    )
    assert not result[deduplicate_taxonomic_rank].duplicated().any()
    assert (result["Assembly Stats Total Sequence Length"] < max_genome_size).all()
    return result


def filter_genomes_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    exclude_genomes: list[str],
    deduplicate_taxonomic_rank: str,
    min_assembly_level: str,
    max_genome_size: int,
    priority_genomes: list[str],
) -> None:
    """Read, filter, and write the selected genome table."""
    genomes = pd.read_parquet(input_path)
    selected = filter_genomes(
        genomes,
        exclude_genomes=exclude_genomes,
        deduplicate_taxonomic_rank=deduplicate_taxonomic_rank,
        min_assembly_level=min_assembly_level,
        max_genome_size=max_genome_size,
        priority_genomes=priority_genomes,
    )
    selected.to_parquet(output_path, index=False)
