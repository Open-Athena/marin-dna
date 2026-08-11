from pathlib import Path

from marin_dna_zoonomia_projection.cli.build_species_list import (
    _to_query,
    default_species_output,
)


def test_to_query_normalizes_known_alias_and_duplicate_suffix() -> None:
    assert _to_query("CanFam4") == "Canis lupus familiaris"
    assert _to_query("Mus_musculus_a") == "Mus musculus"


def test_default_species_output_is_in_pipeline_config() -> None:
    output = default_species_output("family")

    assert output.name == "species_zoonomia_447_family_dedup.tsv"
    assert output.parent.name == "config"
    assert (output.parent.parent / "pyproject.toml").is_file()
