import pandas as pd

from marin_dna_genome_selection.selection import add_taxonomy, filter_genomes


def test_add_taxonomy_expands_ranks_and_preserves_missing() -> None:
    genomes = pd.DataFrame({"Organism Taxonomic ID": [1, 2]})
    taxonomy = pd.DataFrame(
        {
            "taxonomy": [
                {
                    "tax_id": 1,
                    "classification": {
                        "species": {"name": "Species one"},
                        "genus": {"name": "Genus one"},
                    },
                }
            ]
        }
    )

    result = add_taxonomy(genomes, taxonomy)

    assert result.loc[0, "species"] == "Species one"
    assert result.loc[0, "genus"] == "Genus one"
    assert pd.isna(result.loc[1, "species"])
    assert "classification" not in result.columns


def test_filter_genomes_prefers_priority_then_assembly_quality() -> None:
    genomes = pd.DataFrame(
        {
            "Assembly Accession": ["ordinary-complete", "priority-chromosome", "other"],
            "Assembly Level": ["Complete Genome", "Chromosome", "Complete Genome"],
            "Assembly Stats Total Sequence Length": [100, 200, 50],
            "Organism Name": ["A", "B", "C"],
            "species": ["same", "same", "other"],
        }
    )

    result = filter_genomes(
        genomes,
        exclude_genomes=[],
        deduplicate_taxonomic_rank="species",
        min_assembly_level="Chromosome",
        max_genome_size=1_000,
        priority_genomes=["priority-chromosome"],
    )

    assert result["Assembly Accession"].tolist() == [
        "priority-chromosome",
        "other",
    ]


def test_filter_genomes_applies_exclusion_level_and_size_limits() -> None:
    genomes = pd.DataFrame(
        {
            "Assembly Accession": ["excluded", "scaffold", "too-large", "kept"],
            "Assembly Level": [
                "Complete Genome",
                "Scaffold",
                "Chromosome",
                "Chromosome",
            ],
            "Assembly Stats Total Sequence Length": [10, 10, 1_000, 999],
            "Organism Name": ["A", "B", "C", "D"],
            "species": ["a", "b", "c", "d"],
        }
    )

    result = filter_genomes(
        genomes,
        exclude_genomes=["excluded"],
        deduplicate_taxonomic_rank="species",
        min_assembly_level="Chromosome",
        max_genome_size=1_000,
        priority_genomes=[],
    )

    assert result["Assembly Accession"].tolist() == ["kept"]
