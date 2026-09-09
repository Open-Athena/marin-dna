"""Locus identity, document geometry, and locus-isolated validation splits.

Coordinates are 0-based and half-open. These functions consume unsplit,
human-oriented windows, never the legacy per-species training splits.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

WINDOW_BP = 255
CENTER_INDEX = 127
MAX_SPECIES = 40
MODEL_TOKENS = 10_240
SEPARATOR = "[SEQ]"
HUMAN = "hg38"
REGIONS = ("cds", "tss_utr5", "utr3", "ncrna", "enhancer")
_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def stable_digest(*parts: str | int) -> str:
    """Hash an unambiguous identity without Python's process-randomized hash."""
    return hashlib.sha256(
        json.dumps(parts, ensure_ascii=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True, order=True)
class Locus:
    """One exact hg38 query window, independent of catalog or benchmark."""

    chrom: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if not self.chrom.startswith("chr") or any(c.isspace() for c in self.chrom):
            raise ValueError("locus requires an explicit UCSC chromosome name")
        if type(self.start) is not int or type(self.end) is not int:
            raise ValueError("locus coordinates must be integers")
        if self.start < 0 or self.end - self.start != WINDOW_BP:
            raise ValueError("locus must be a nonnegative 255-bp half-open window")

    @property
    def query_name(self) -> str:
        """Coordinate identity shared by training and exact VEP requests."""
        return "rag_" + stable_digest("hg38", self.chrom, self.start, self.end)

    def validate_bounds(self, chrom_sizes: Mapping[str, int]) -> None:
        if self.chrom not in chrom_sizes or self.end > chrom_sizes[self.chrom]:
            raise ValueError(f"locus outside pinned genome: {self}")


def reverse_complement(sequence: str) -> str:
    """Reverse-complement DNA while retaining genuine N bases and case."""
    if set(sequence) - set("ACGTNacgtn"):
        raise ValueError("DNA contains characters outside A/C/G/T/N")
    return sequence.translate(_COMPLEMENT)[::-1]


def validate_sequence(sequence: str) -> None:
    if len(sequence) != WINDOW_BP or set(sequence) - set("ACGTNacgtn"):
        raise ValueError("each available species must have exactly 255 A/C/G/T/N bases")


@dataclass(frozen=True)
class Document:
    sequence: str
    species_order: tuple[str, ...]
    orientation: str

    @property
    def unpadded_tokens(self) -> int:
        return 256 * len(self.species_order)

    @property
    def padding_tokens(self) -> int:
        return MODEL_TOKENS - self.unpadded_tokens


def assemble_document(
    windows: Mapping[str, str],
    *,
    locus: Locus,
    seed: int = 42,
    orientation: str = "forward",
    evaluation: bool = False,
) -> Document:
    """Order each published row reproducibly using no alleles or labels.

    The hash ranking is a reproducible random permutation independent of input
    iteration order. Training orientation copies receive their own fixed order.
    Evaluation orientations share one order and put human last. Missing species
    must be omitted from ``windows``; an observed all-N window remains available.
    """
    if orientation not in {"forward", "rc"}:
        raise ValueError("orientation must be forward or rc")
    if HUMAN not in windows or not 1 <= len(windows) <= MAX_SPECIES:
        raise ValueError("documents require human and at most 40 species")
    for species, sequence in windows.items():
        if not isinstance(species, str) or not species:
            raise ValueError("species identities must be nonempty strings")
        validate_sequence(sequence)
    domain = "evaluation" if evaluation else f"training-{orientation}"
    order = sorted(
        (species for species in windows if not (evaluation and species == HUMAN)),
        key=lambda species: (
            stable_digest(seed, domain, locus.query_name, species),
            species,
        ),
    )
    if evaluation:
        order.append(HUMAN)
    segments = [windows[species] for species in order]
    if orientation == "rc":
        segments = [reverse_complement(segment) for segment in segments]
    return Document(SEPARATOR.join(segments), tuple(order), orientation)


def allele_documents(
    windows: Mapping[str, str],
    *,
    locus: Locus,
    ref: str,
    alt: str,
    seed: int = 42,
) -> dict[str, Document]:
    """Build paired SNV inputs with the same retrieval, order, and padding."""
    if (
        ref not in "ACGT"
        or alt not in "ACGT"
        or len(ref) != 1
        or len(alt) != 1
        or ref == alt
    ):
        raise ValueError("RAG VEP requires distinct single-base uppercase REF and ALT")
    if HUMAN not in windows:
        raise ValueError("human reference window is required")
    human = windows[HUMAN]
    validate_sequence(human)
    if human[CENTER_INDEX].upper() != ref:
        raise ValueError(f"REF mismatch at {locus.chrom}:{locus.start + CENTER_INDEX}")
    # Normalize both designated alleles to the same case before tokenization.
    outputs = {}
    for allele_name, allele in (("ref", ref), ("alt", alt)):
        changed = {
            **windows,
            HUMAN: human[:CENTER_INDEX] + allele + human[CENTER_INDEX + 1 :],
        }
        for orientation in ("forward", "rc"):
            outputs[f"{allele_name}_{orientation}"] = assemble_document(
                changed,
                locus=locus,
                seed=seed,
                orientation=orientation,
                evaluation=True,
            )
    return outputs


@dataclass(frozen=True)
class SplitPlan:
    """Original-orientation memberships; RC is applied only after this split."""

    train: dict[str, tuple[Locus, ...]]
    validation: dict[str, tuple[Locus, ...]]
    excluded: dict[str, tuple[Locus, ...]]
    exact_cross_region_duplicates: int


def select_validation(
    catalogs: Mapping[str, Iterable[Locus]],
    *,
    validation_rows: int = 400,
    seed: int = 42,
) -> SplitPlan:
    """Hold out all chr18 loci and sample fixed original-orientation validation rows.

    Source-region memberships are preserved. Every chr18 window is excluded
    from training, whether or not it is among that region's sampled validation
    rows. Use all available chr18 loci when fewer than the requested budget
    exist, including an explicitly empty validation set. This also separates all orientation copies and
    cross-region overlaps without a coordinate-overlap exclusion pass.
    """
    if validation_rows < 1 or not catalogs:
        raise ValueError("validation budget and catalogs must be nonempty")
    validation: dict[str, tuple[Locus, ...]] = {}
    train: dict[str, tuple[Locus, ...]] = {}
    excluded: dict[str, tuple[Locus, ...]] = {}
    seen: set[Locus] = set()
    duplicates = 0
    for region, records in sorted(catalogs.items()):
        rows = tuple(records)
        unique = set(rows)
        if len(rows) != len(unique):
            raise ValueError(f"duplicate locus within {region} source catalog")
        duplicates += len(unique & seen)
        seen.update(unique)
        candidates = [locus for locus in rows if locus.chrom == "chr18"]
        validation[region] = tuple(
            sorted(
                candidates,
                key=lambda locus: (
                    stable_digest(seed, "validation", region, locus.query_name),
                    locus,
                ),
            )[:validation_rows]
        )
        selected = set(validation[region])
        train[region] = tuple(sorted(locus for locus in rows if locus.chrom != "chr18"))
        excluded[region] = tuple(
            sorted(locus for locus in candidates if locus not in selected)
        )
        if not train[region]:
            raise ValueError(f"chr18 holdout leaves no training rows in {region}")
    return SplitPlan(train, validation, excluded, duplicates)
