"""Pinned, assembly-matched sequence inputs for chain projection.

Manifests are metadata only; reading one never downloads or opens a genome.
Actual files are verified before conversion, dictionary inspection, or extraction.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from marin_dna_vertebrate_projection.projection.chains import file_sha256, read_sizes

COLUMNS = {
    "name",
    "assembly",
    "sequence",
    "format",
    "sha256",
    "chrom_sizes",
    "chrom_sizes_sha256",
    "origin",
}


def read_genome_assets(
    path: str | Path, chains: dict[str, dict[str, str]]
) -> dict[str, dict[str, str]]:
    """Require human and every projected target, with exact assembly/dictionary pins."""
    assets: dict[str, dict[str, str]] = {}
    with Path(path).open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not COLUMNS <= set(reader.fieldnames or []):
            raise ValueError("genome manifest is missing required columns")
        for row in reader:
            if any(not row.get(key) for key in COLUMNS):
                raise ValueError("genome manifest contains empty fields")
            name = row["name"]
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in assets:
                raise ValueError(f"invalid or duplicate genome name: {name}")
            if row["format"] not in {"fasta", "twobit"}:
                raise ValueError(f"unsupported genome format: {name}")
            for key in ("sha256", "chrom_sizes_sha256"):
                if not re.fullmatch(r"[0-9a-f]{64}", row[key]):
                    raise ValueError(f"invalid genome {key}: {name}")
            assets[name] = {key: row[key] for key in sorted(COLUMNS)}
    if set(assets) != {"hg38", *chains}:
        raise ValueError(
            "genome manifest must contain human and exactly the chain targets"
        )
    if assets["hg38"]["assembly"] != "hg38":
        raise ValueError("human sequence must use the pinned hg38 assembly")
    for name, chain in chains.items():
        genome = assets[name]
        if genome["assembly"] != chain["assembly"]:
            raise ValueError(f"chain/genome assembly mismatch: {name}")
        if (genome["chrom_sizes"], genome["chrom_sizes_sha256"]) != (
            chain["target_sizes"],
            chain["target_sizes_sha256"],
        ):
            raise ValueError(f"chain/genome chromosome dictionary mismatch: {name}")
    return assets


def validate_genome_source(
    sequence: str | Path,
    sizes: str | Path,
    expected: dict[str, str],
    output: str | Path,
) -> None:
    """Verify source bytes and dictionary before preparing a sequence archive."""
    if file_sha256(sequence) != expected["sha256"]:
        raise ValueError("genome sequence SHA-256 mismatch")
    if file_sha256(sizes) != expected["chrom_sizes_sha256"]:
        raise ValueError("genome chromosome dictionary SHA-256 mismatch")
    if expected["format"] == "fasta":
        with Path(sequence).open() as handle:
            for line in handle:
                if not line.startswith(">") and set(line.strip()) - set("ACGTNacgtn"):
                    raise ValueError("FASTA contains bases that 2bit cannot preserve")
    read_sizes(sizes)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(expected, indent=2) + "\n")


def validate_genome_dictionary(
    observed: str | Path, expected: str | Path, output: str | Path
) -> None:
    """Require exact contig names and lengths for the complete sequence archive."""
    actual, pinned = read_sizes(observed), read_sizes(expected)
    if actual != pinned:
        raise ValueError(
            "sequence archive differs from the pinned chromosome dictionary"
        )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps({"checked_contigs": len(pinned)}) + "\n")


def validate_human_dictionary(
    chain_sizes: str | Path, genome_sizes: str | Path
) -> None:
    """Allow verified extra aliases in chains, but require the complete human dictionary."""
    source, genome = read_sizes(chain_sizes), read_sizes(genome_sizes)
    if any(source.get(name) != size for name, size in genome.items()):
        raise ValueError("chain source dictionary disagrees with the human genome")
