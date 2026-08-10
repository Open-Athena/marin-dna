# Standard library
import os
import subprocess

# Core data science libraries
import numpy as np
import pandas as pd
import polars as pl
import py2bit
import pyBigWig
import matplotlib.pyplot as plt
from tqdm import tqdm

# MarinDNA module imports
from marin_dna.data.intervals import GenomicSet
from marin_dna.data.utils import (
    ENHANCER_CRE_CLASSES,
    HUMAN_GENOME,
    STANDARD_CHROMS,
    add_rc,
    get_3_prime_utr,
    get_5_prime_utr,
    get_array_split_pairs,
    get_cds,
    get_downstream_of_CDS,
    get_exons_for_masking,
    get_mrna_exons,
    get_ncrna_exons,
    get_promoters,
    get_promoters_from_exons,
    get_upstream_of_CDS,
    load_annotation,
    load_fasta,
    read_bed_to_pandas,
    write_pandas_to_bed,
)

tqdm.pandas()


def _git_commit_sha() -> str:
    """Return the current git commit SHA, or ``"main"`` if git is unavailable.

    Resolution order: ``--config commit_sha=...`` / ``$COMMIT_SHA`` override
    (lets a sky launcher forward the host SHA to a worktree-synced cluster
    where ``git rev-parse`` would fail), then ``git rev-parse HEAD`` from the
    workflow dir, then the literal ``"main"`` fallback so the permalink still
    resolves. Embedded in the per-repo HF dataset cards (see
    ``marin_dna.pipelines.training_dataset.hf_readme``); resolved once at
    Snakefile-load time so all cards in a run share one SHA.
    """
    override = config.get("commit_sha") or os.environ.get("COMMIT_SHA")
    if override and len(override) == 40:
        return override
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workflow.basedir,
            capture_output=True,
            text=True,
            check=False,
        )
        sha = result.stdout.strip()
        if result.returncode == 0 and len(sha) == 40:
            return sha
    except FileNotFoundError:
        pass
    return "main"


GIT_COMMIT_SHA = _git_commit_sha()


def load_genome_info(path: str) -> pd.DataFrame:
    genomes = pd.read_parquet(path).set_index("Assembly Accession")
    genomes["Assembly Name"] = genomes["Assembly Name"].str.replace(" ", "_")
    genomes["genome_path"] = (
        "tmp/"
        + genomes.index
        + "/ncbi_dataset/data/"
        + genomes.index
        + "/"
        + genomes.index
        + "_"
        + genomes["Assembly Name"]
        + "_genomic.fna"
    )
    genomes["annotation_path"] = (
        "tmp/"
        + genomes.index
        + "/ncbi_dataset/data/"
        + genomes.index
        + "/"
        + "genomic.gtf"
    )
    return genomes


def load_genome_sets(
    genomes: pd.DataFrame, genome_sets_config: list[dict]
) -> dict[str, list[str]]:
    """
    Load genome sets based on taxonomic filtering or an explicit accession list.

    Each entry in ``genome_sets_config`` must have a ``name``, plus either:
      - ``rank_key`` + ``rank_value`` to select all genomes with
        ``genomes[rank_key] == rank_value`` (e.g., all Mammalia), or
      - ``accessions``: an explicit list of Assembly Accessions
        (e.g., the 20 genomes covered by segmentation prediction).

    Returns:
        Dictionary mapping subset names to lists of genome Assembly Accessions.
    """
    genome_sets = {}
    for genome_set in genome_sets_config:
        name = genome_set["name"]
        if "accessions" in genome_set:
            accessions = list(genome_set["accessions"])
            unknown = [a for a in accessions if a not in genomes.index]
            assert (
                not unknown
            ), f"genome_set {name!r} references unknown accessions: {unknown}"
            genome_sets[name] = accessions
        else:
            rank_key = genome_set["rank_key"]
            rank_value = genome_set["rank_value"]
            filtered = genomes[genomes[rank_key] == rank_value]
            genome_sets[name] = filtered.index.tolist()

    return genome_sets
