from __future__ import annotations

import csv
from pathlib import Path

import pytest
from marin_dna_vertebrate_projection.genome_assets import (
    read_genome_assets,
    validate_genome_dictionary,
    validate_genome_source,
    validate_human_dictionary,
)
from marin_dna_vertebrate_projection.projection.chains import (
    file_sha256,
    read_chain_assets,
)

PROJECT = Path(__file__).parents[1]
FIXTURE = PROJECT / "tests/fixtures/chains"


def test_genome_manifest_covers_both_cohorts_and_human() -> None:
    chains = read_chain_assets(
        FIXTURE / "pipeline_assets.tsv", PROJECT / "config/species_selected.tsv"
    )
    genomes = read_genome_assets(FIXTURE / "genomes.tsv", chains)
    assert set(genomes) == {"hg38", "Papio_anubis", "galGal4"}


@pytest.mark.parametrize(
    "change", ["assembly", "dictionary", "missing", "duplicate", "digest"]
)
def test_genome_manifest_fails_closed(tmp_path: Path, change: str) -> None:
    chains = read_chain_assets(
        FIXTURE / "pipeline_assets.tsv", PROJECT / "config/species_selected.tsv"
    )
    with (FIXTURE / "genomes.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    columns = list(rows[0])
    if change == "assembly":
        rows[1]["assembly"] = "wrong-version"
    elif change == "dictionary":
        rows[1]["chrom_sizes_sha256"] = "a" * 64
    elif change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(rows[0])
    else:
        rows[0]["sha256"] = "invalid"
    path = tmp_path / "genomes.tsv"
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        read_genome_assets(path, chains)


def test_genome_bytes_and_complete_dictionary_are_verified(tmp_path: Path) -> None:
    chains = read_chain_assets(
        FIXTURE / "pipeline_assets.tsv", PROJECT / "config/species_selected.tsv"
    )
    genome = read_genome_assets(FIXTURE / "genomes.tsv", chains)["hg38"]
    validate_genome_source(
        FIXTURE / "human.fa", FIXTURE / "source.sizes", genome, tmp_path / "source.json"
    )
    with pytest.raises(ValueError, match="sequence SHA"):
        validate_genome_source(
            FIXTURE / "target.fa",
            FIXTURE / "source.sizes",
            genome,
            tmp_path / "source.json",
        )
    sizes = tmp_path / "actual.sizes"
    sizes.write_text("chr1\t1000\n")
    validate_genome_dictionary(
        sizes, FIXTURE / "source.sizes", tmp_path / "checked.json"
    )
    for wrong in ("1\t1000\n", "chr1\t1001\n", "chr1\t1000\nchrExtra\t100\n"):
        sizes.write_text(wrong)
        with pytest.raises(ValueError, match="pinned chromosome dictionary"):
            validate_genome_dictionary(
                sizes, FIXTURE / "source.sizes", tmp_path / "checked.json"
            )


def test_human_chain_dictionary_may_add_but_not_change_aliases(tmp_path: Path) -> None:
    sizes = tmp_path / "chain.sizes"
    sizes.write_text("chr1\t1000\nCM000663.2\t1000\n")
    validate_human_dictionary(sizes, FIXTURE / "source.sizes")
    for wrong in ("chr1\t1001\n", "CM000663.2\t1000\n"):
        sizes.write_text(wrong)
        with pytest.raises(ValueError, match="human genome"):
            validate_human_dictionary(sizes, FIXTURE / "source.sizes")


def test_fasta_conversion_rejects_lossy_iupac(tmp_path: Path) -> None:
    chains = read_chain_assets(
        FIXTURE / "pipeline_assets.tsv", PROJECT / "config/species_selected.tsv"
    )
    genome = read_genome_assets(FIXTURE / "genomes.tsv", chains)["hg38"]
    fasta = tmp_path / "ambiguous.fa"
    fasta.write_text(">chr1\nACGTRY\n")
    genome["sha256"] = file_sha256(fasta)
    with pytest.raises(ValueError, match="2bit cannot preserve"):
        validate_genome_source(
            fasta, FIXTURE / "source.sizes", genome, tmp_path / "validated.json"
        )
