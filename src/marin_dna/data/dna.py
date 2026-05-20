"""Canonical DNA constants and case-preserving sequence helpers.

The single source of truth for ``NUCLEOTIDES`` / ``COMPLEMENT`` /
``reverse_complement`` across marin_dna. Module-level only — no torch /
polars / biopython deps — so it can be imported anywhere cheaply.
"""

NUCLEOTIDES = list("ACGT")
COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A"}

# Case-preserving translate table; ``N``/``n`` map to themselves. Unknown
# characters (IUPAC ambiguity codes, etc.) pass through unchanged — same
# behavior as biopython's ``Seq.reverse_complement`` for our purposes.
_DNA_RC_TRANSLATE = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def reverse_complement(seq: str) -> str:
    """Reverse-complement an arbitrary DNA string, preserving case.

    ~2× faster than ``str(Bio.Seq.Seq(seq).reverse_complement())`` for
    sequences up to ~1 kb; biopython catches up at longer windows.
    """
    return seq.translate(_DNA_RC_TRANSLATE)[::-1]


def complement_base(base: str) -> str:
    """Complement a single A/C/G/T base; pass anything else through unchanged.

    Non-ACGT inputs (N, IUPAC codes, lowercase) round-trip via the unchanged
    branch — downstream DNA tokenizers collapse them to a single unknown
    token regardless, so the exact value returned here is moot.
    """
    return COMPLEMENT.get(base, base)
