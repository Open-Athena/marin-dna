"""Exact request identities and fail-closed reuse of unsplit chain outputs."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from marin_dna_vertebrate_projection.rag.documents import Locus, stable_digest


@dataclass(frozen=True)
class ProjectionIdentity:
    """Immutable inputs that determine projection and extracted sequence bytes.

    Archive digests identify genome revisions. The complete contract identifies
    mapping multiplicity, center placement, sequence case, and orientation.
    Historical single-best, HAL, or MultiZ results are different identities.
    """

    species: str
    assembly: str
    chain_sha256: str
    source_sizes_sha256: str
    target_sizes_sha256: str
    genome_sha256: str
    contract: str = "ucsc-liftover-minmatch0.95-multiple-center1-window255-human-orientation-case-v1"

    def __post_init__(self) -> None:
        if not self.species or not self.assembly or not self.contract:
            raise ValueError("projection species, assembly, and contract are required")
        for digest in (
            self.chain_sha256,
            self.source_sizes_sha256,
            self.target_sizes_sha256,
            self.genome_sha256,
        ):
            if not re.fullmatch("[0-9a-f]{64}", digest):
                raise ValueError(
                    "projection identity requires complete SHA-256 digests"
                )

    @property
    def fingerprint(self) -> str:
        return stable_digest(
            self.species,
            self.assembly,
            self.chain_sha256,
            self.source_sizes_sha256,
            self.target_sizes_sha256,
            self.genome_sha256,
            self.contract,
        )


@dataclass(frozen=True)
class SourceRecord:
    """A catalog or benchmark row and its exact projection request."""

    namespace: str
    source_row_id: str
    locus: Locus

    def __post_init__(self) -> None:
        if not self.namespace or not self.source_row_id:
            raise ValueError("source identity requires namespace and row ID")

    @property
    def row_id(self) -> str:
        return stable_digest(self.namespace, self.source_row_id)


@dataclass(frozen=True)
class RequestPlan:
    requests: tuple[Locus, ...]
    source_rows: tuple[SourceRecord, ...]
    missing: dict[str, tuple[Locus, ...]]
    reused: dict[str, tuple[Locus, ...]]


def plan_requests(
    source_rows: Iterable[SourceRecord],
    identities: Mapping[str, ProjectionIdentity],
    *,
    chrom_sizes: Mapping[str, int],
    cached: Mapping[str, Mapping[Locus, str]] | None = None,
) -> RequestPlan:
    """Deduplicate requests across all catalogs and benchmarks before species batches.

    ``cached`` records completed accepted OR rejected queries, keyed by exact
    coordinates and accompanied by their producing projection fingerprint.
    Only verified matching fingerprints can suppress a chain query. Cache
    provenance/file checksums must be validated by the file adapter first.
    """
    rows = tuple(source_rows)
    if len({row.row_id for row in rows}) != len(rows):
        raise ValueError("duplicate namespaced source row identity")
    if not rows or not identities:
        raise ValueError("projection plan requires source rows and target identities")
    if any(species != identity.species for species, identity in identities.items()):
        raise ValueError("target identity does not match species key")
    loci = tuple(sorted({row.locus for row in rows}))
    for locus in loci:
        locus.validate_bounds(chrom_sizes)
    cache = cached or {}
    if set(cache) - set(identities):
        raise ValueError("cache includes an unregistered target species")
    missing, reused = {}, {}
    for species, identity in sorted(identities.items()):
        species_cache = cache.get(species, {})
        reused[species] = tuple(
            locus for locus in loci if species_cache.get(locus) == identity.fingerprint
        )
        reused_set = set(reused[species])
        missing[species] = tuple(locus for locus in loci if locus not in reused_set)
    return RequestPlan(loci, rows, missing, reused)
