"""Rank-deduplicate the Zoonomia 447 leaf set (one leaf per family or order).

Pure functions over already-fetched data: Newick parsing, name
normalization, and the per-rank ranking + dedup policy. Network calls
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
    """Per-leaf record fed to :func:`dedup_by_rank`."""

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
    if name.endswith(("_a", "_b")):
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


# Taxonomic ranks we can deduplicate to. These are the only two fields
# populated on LeafMeta (the build script fetches family + order from NCBI);
# grouping on anything else would silently collapse every leaf into one
# all-None group, so we guard against it.
DEDUP_RANKS: tuple[str, ...] = ("family", "order")


def dedup_by_rank(
    rows: list[LeafMeta],
    rank: str,
    *,
    force_include: frozenset[str] = frozenset(
        {"Homo_sapiens", "Mus_musculus", "Bos_taurus"}
    ),
) -> list[LeafMeta]:
    """Pick one leaf per taxonomic ``rank`` group by the dedup policy.

    ``rank`` selects the grouping attribute on :class:`LeafMeta` and must be
    one of :data:`DEDUP_RANKS` (``"family"`` or ``"order"``) — the two
    taxonomy fields the build script populates. ``"order"`` yields a sparser,
    more deeply-diverged set than ``"family"``.

    Policy:

    1. ``force_include`` species win their group unconditionally.
    2. Other groups pick by sort order :func:`_rank_key`.

    Rows whose ``rank`` value is ``None`` are dropped — they cannot
    participate in dedup at that rank. Force-include species missing from
    ``rows`` entirely are silently absent (caller asserts).

    Returns the chosen LeafMeta records, sorted by the group key.

    Subset invariant: every ``rank="order"`` winner is also a
    ``rank="family"`` winner over the same rows, so
    order-winners ⊆ family-winners. An order with no force-include is won by
    its top-ranked leaf, which is also top-ranked in its own family (⊆ the
    order). An order containing force-includes is won by its top-ranked
    force-include, which is likewise the top-ranked force-include in its own
    family — so it wins both ranks. (Any *other* force-include in that order
    loses it but still wins its own family, which is fine: only the winner
    need be a family winner.) Relied on by the dataset follow-up.
    """
    assert rank in DEDUP_RANKS, f"rank must be one of {DEDUP_RANKS}; got {rank!r}"
    by_group: dict[str, list[LeafMeta]] = {}
    for r in rows:
        key = getattr(r, rank)
        if key is None:
            continue
        by_group.setdefault(key, []).append(r)

    winners: list[LeafMeta] = []
    for key in sorted(by_group.keys()):
        candidates = by_group[key]
        forced = [r for r in candidates if r.leaf in force_include]
        if forced:
            chosen = min(forced, key=_rank_key)
        else:
            chosen = min(candidates, key=_rank_key)
        winners.append(chosen)
    return winners
