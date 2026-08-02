"""Frozen constants and pure helpers for feature-1662 saturation mutagenesis."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

RUN_ID = "dna-exp438-feature1662-saturation-r2"
DESIGN_RUN_ID = f"{RUN_ID}-design"
SELECTION_HASH_NAMESPACE = "exp438-feature1662-saturation-r1"
FEATURE_ID = 1_662
BLOCK_INDEX = 18
WINDOW_BP = 255
FOCAL_INDEX = 127
SATURATION_RADIUS = 15
CONTEXTS_PER_CODON_POSITION = 120
POSITIONS = tuple(range(-SATURATION_RADIUS, SATURATION_RADIUS + 1))
ORIENTATIONS = ("forward", "reverse_complement")
NUCLEOTIDES = tuple("ACGT")

COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A"}

# Standard nuclear genetic code, kept local so panel semantics do not depend on
# a mutable annotation or translation package.
GENETIC_CODE = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}

assert WINDOW_BP == 2 * FOCAL_INDEX + 1
assert len(POSITIONS) == 31
assert set(GENETIC_CODE) == {
    first + second + third
    for first in NUCLEOTIDES
    for second in NUCLEOTIDES
    for third in NUCLEOTIDES
}


def complement(base: str) -> str:
    base = base.upper()
    assert base in COMPLEMENT
    return COMPLEMENT[base]


def translate_codon(codon: str) -> str:
    codon = codon.upper()
    assert len(codon) == 3 and set(codon) <= set(NUCLEOTIDES)
    return GENETIC_CODE[codon]


def parse_codon_change(value: str) -> tuple[str, str, int]:
    """Return uppercase ref/alt codons and the unique 0-based changed position."""

    parts = value.upper().split("/")
    assert len(parts) == 2
    ref_codon, alt_codon = parts
    assert len(ref_codon) == len(alt_codon) == 3
    assert set(ref_codon + alt_codon) <= set(NUCLEOTIDES)
    changed = [index for index in range(3) if ref_codon[index] != alt_codon[index]]
    assert len(changed) == 1
    return ref_codon, alt_codon, changed[0]


def selection_hash(row: dict[str, Any]) -> str:
    key = "|".join(
        (
            SELECTION_HASH_NAMESPACE,
            str(row["panel_row"]),
            str(row["chrom"]),
            str(row["pos"]),
            str(row["ref"]),
            str(row["alt"]),
            str(row["transcript_id"]),
        )
    )
    return hashlib.sha256(key.encode()).hexdigest()


def codon_genomic_offsets(focal_codon_position: int, strand: int) -> tuple[int, ...]:
    """Map transcript codon positions 1..3 to genomic offsets from the focal SNV."""

    assert focal_codon_position in {1, 2, 3} and strand in {-1, 1}
    focal_index = focal_codon_position - 1
    return tuple(strand * (index - focal_index) for index in range(3))


def mutate_transcript_codon(
    ref_codon: str,
    *,
    focal_codon_position: int,
    strand: int,
    genomic_offset: int,
    genomic_alt: str,
) -> tuple[int, str, str, str] | None:
    """Annotate a genomic edit when it falls inside the focal transcript codon."""

    offsets = codon_genomic_offsets(focal_codon_position, strand)
    if genomic_offset not in offsets:
        return None
    codon_index = offsets.index(genomic_offset)
    transcript_alt = genomic_alt.upper() if strand == 1 else complement(genomic_alt)
    alt_codon = ref_codon[:codon_index] + transcript_alt + ref_codon[codon_index + 1 :]
    ref_aa, alt_aa = translate_codon(ref_codon), translate_codon(alt_codon)
    if alt_aa == ref_aa:
        consequence = "synonymous"
    elif alt_aa == "*":
        consequence = "stop_gained"
    else:
        consequence = "missense"
    return codon_index + 1, alt_codon, alt_aa, consequence


def bh_adjust(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    assert values.ndim == 1 and values.size > 0 and np.isfinite(values).all()
    assert np.all((0 <= values) & (values <= 1))
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    scaled = ranked * ranked.size / np.arange(1, ranked.size + 1)
    monotone = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted = np.empty_like(values)
    adjusted[order] = np.clip(monotone, 0, 1)
    return adjusted
