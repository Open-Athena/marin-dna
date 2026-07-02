"""Genome-list selection helpers for the training-dataset pipeline.

``select_one_per_rank`` mirrors the ``filter_genomes`` rule in the
``genome_selection`` Snakefile, but with the deduplication rank as a parameter,
so the same quality-ranked "one genome per taxonomic rank" logic can produce a
one-per-order list (or any rank) from an already-built genome universe without
re-running the NCBI download pipeline.

``read_accessions_tsv`` reads the committed species-list TSVs those runs
produce, so a curated genome set can be wired into ``dataset_creation`` via an
``accessions_tsv:`` entry instead of an inline accession list.
"""

import pandas as pd

# Most complete to least complete; expresses the dedup preference. Matches the
# ``ASSEMBLY_LEVELS`` list in the genome_selection Snakefile.
ASSEMBLY_LEVELS: list[str] = [
    "Complete Genome",
    "Chromosome",
    "Scaffold",
    "Contig",
]

ACCESSION_COL = "Assembly Accession"
LEVEL_COL = "Assembly Level"
SIZE_COL = "Assembly Stats Total Sequence Length"
NAME_COL = "Organism Name"


def select_one_per_rank(
    genomes: pd.DataFrame,
    rank: str,
    *,
    priority: list[str] | tuple[str, ...] = (),
    exclude: list[str] | tuple[str, ...] = (),
    max_genome_size: int | None = None,
    min_assembly_level: str = "Contig",
) -> pd.DataFrame:
    """Keep one best-quality genome per value of ``rank``.

    Reproduces the ``filter_genomes`` ranking (priority genomes first, then
    most-complete assembly level, then smallest genome, then name) with the
    deduplication rank parameterised. All filters are idempotent, so running
    this on an already family-deduplicated universe re-deduplicates it to a
    coarser rank (e.g. ``order``) while keeping the same genome winning each
    group.

    Args:
        genomes: One row per assembly, with the NCBI-datasets columns
            ``Assembly Accession``, ``Assembly Level``,
            ``Assembly Stats Total Sequence Length``, ``Organism Name`` and a
            column named ``rank`` (e.g. ``order``).
        rank: Taxonomic-rank column to deduplicate on (one row kept per value).
        priority: Accessions forced to the top of the ranking (kept if present
            in their group).
        exclude: Accessions dropped entirely.
        max_genome_size: If set, drop assemblies with at least this many bases.
        min_assembly_level: Least-complete assembly level to keep.

    Returns:
        A copy of ``genomes`` with one row per ``rank`` value, sorted by the
        ranking keys and re-indexed. ``Assembly Level`` is an ordered
        categorical in the result.
    """
    assert rank in genomes.columns, f"rank column {rank!r} not in genomes"
    priority = list(priority)
    exclude = set(exclude)

    df = genomes[~genomes[ACCESSION_COL].isin(exclude)].copy()
    df = df.dropna(subset=[rank])
    df[LEVEL_COL] = pd.Categorical(df[LEVEL_COL], ASSEMBLY_LEVELS, ordered=True)
    df = df[df[LEVEL_COL] <= min_assembly_level]
    if max_genome_size is not None:
        df = df[df[SIZE_COL] < max_genome_size]

    df["_priority"] = "1_Low"
    df.loc[df[ACCESSION_COL].isin(priority), "_priority"] = "0_High"
    df = (
        df.sort_values(["_priority", LEVEL_COL, SIZE_COL, NAME_COL])
        .drop(columns=["_priority"])
        .drop_duplicates(rank)
        .reset_index(drop=True)
    )
    return df


def read_accessions_tsv(path: str) -> list[str]:
    """Return the ``Assembly Accession`` column of a committed species-list TSV.

    Used by ``dataset_creation``'s genome-set loader to resolve an
    ``accessions_tsv:`` entry. Asserts the column is present and accessions are
    unique.
    """
    df = pd.read_csv(path, sep="\t")
    assert ACCESSION_COL in df.columns, (
        f"{path} missing {ACCESSION_COL!r} column; has {list(df.columns)}"
    )
    accessions = df[ACCESSION_COL].tolist()
    assert len(accessions) == len(set(accessions)), f"duplicate accessions in {path}"
    return accessions
