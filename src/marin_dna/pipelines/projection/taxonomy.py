"""Family-deduplicate the Zoonomia 447 leaf set.

Pure functions over already-fetched data: Newick parsing, name
normalization, and the per-family ranking + dedup policy. Network calls
(NCBI Datasets v2, Zoonomia supplementary xlsx) live in the reproducer
script ``snakemake/zoonomia_projection_dataset/scripts/build_species_list.py``,
not here, so this module stays pure-Python testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# Higher rank = better assembly. Missing keys map to 0 in lookups.
ASSEMBLY_LEVEL_RANK: dict[str, int] = {
    "Complete Genome": 3,
    "Chromosome": 2,
    "Scaffold": 1,
    "Contig": 0,
}

# Higher rank = preferred quality_source. ST2-true beats proxy beats unknown.
# Why this matters: an ST2-true entry tells us which assembly is *actually*
# inside the HAL; a proxy's "higher" N50 may be for NCBI's current best,
# which may not be what Cactus aligned. Picking a proxy over an ST2-true
# entry would silently swap out the HAL assembly.
QUALITY_SOURCE_RANK: dict[str, int] = {
    "zoonomia_supp_st2": 2,
    "ncbi_taxon_proxy": 1,
    "unknown": 0,
}

ANCESTOR_LABEL_RE = re.compile(r"^(?:fullTreeAnc|PrimatesAnc|Anc)\d+$")


@dataclass(frozen=True)
class LeafMeta:
    """Per-leaf record fed to :func:`dedup_by_family`."""

    leaf: str
    family: str | None
    order: str | None  # NCBI order; carried through for the output TSV
    accession: str | None
    assembly_level: str | None
    contig_n50: int | None
    quality_source: str  # one of QUALITY_SOURCE_RANK keys


def parse_newick_leaves(text: str) -> list[str]:
    """Return leaf names from a Newick tree, dropping ancestor labels.

    Cactus convention: ancestor labels match :data:`ANCESTOR_LABEL_RE`
    (e.g. ``fullTreeAnc116``, ``PrimatesAnc7``). Leaves are names that
    appear directly after ``(`` or ``,`` (i.e. not after ``)``).
    """
    leaves: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in "(," and i + 1 < n and text[i + 1] not in "(),:;":
            j = i + 1
            while j < n and text[j] not in ":,();":
                j += 1
            name = text[i + 1 : j].strip()
            if name and not name.startswith("'"):
                leaves.append(name)
            i = j
        else:
            i += 1
    return [name for name in leaves if not ANCESTOR_LABEL_RE.match(name)]


def normalize_zoonomia_leaf(name: str) -> str:
    """Strip ``_a`` / ``_b`` duplicate-disambiguator suffixes.

    The 447-mammalian Newick adds ``_a`` / ``_b`` to four primate species
    that appeared twice during the alignment merge (per the README's
    "naming-error fix"). Stripping the suffix recovers the biological
    species — useful for NCBI taxonomy lookups and matching the binomials
    in Zoonomia ST2.

    Use this **only** for external lookups. The HAL itself stores leaves
    under their raw ``_a`` / ``_b`` names — pass the un-normalized leaf to
    ``halStats`` / ``halLiftover`` or those will fail with
    ``Genome <name> not found``.
    """
    if name.endswith("_a") or name.endswith("_b"):
        return name[:-2]
    return name


def _rank_key(meta: LeafMeta) -> tuple[int, int, int, str]:
    """Sort key: lower-is-better, so higher quality leafs come first."""
    return (
        -QUALITY_SOURCE_RANK.get(meta.quality_source, 0),
        -ASSEMBLY_LEVEL_RANK.get(meta.assembly_level or "", 0),
        -(meta.contig_n50 or 0),
        meta.leaf,
    )


def dedup_by_family(
    rows: list[LeafMeta],
    *,
    force_include: frozenset[str] = frozenset(
        {"Homo_sapiens", "Mus_musculus", "Bos_taurus"}
    ),
) -> list[LeafMeta]:
    """Pick one leaf per family by the dedup policy.

    Policy:

    1. ``force_include`` species win their family unconditionally.
    2. Other families pick by sort order :func:`_rank_key`.

    Rows with ``family is None`` are dropped — they cannot participate in
    family-level dedup. Force-include species missing from ``rows``
    entirely are silently absent (caller asserts).

    Returns the chosen LeafMeta records, sorted by family.
    """
    by_family: dict[str, list[LeafMeta]] = {}
    for r in rows:
        if r.family is None:
            continue
        by_family.setdefault(r.family, []).append(r)

    winners: list[LeafMeta] = []
    for family in sorted(by_family.keys()):
        candidates = by_family[family]
        forced = [r for r in candidates if r.leaf in force_include]
        if forced:
            chosen = sorted(forced, key=_rank_key)[0]
        else:
            chosen = sorted(candidates, key=_rank_key)[0]
        winners.append(chosen)
    return winners
