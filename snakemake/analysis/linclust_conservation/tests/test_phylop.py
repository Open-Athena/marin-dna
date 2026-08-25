import math

import pytest

from marin_dna_linclust_conservation.phylop import (
    REFSEQ_GRCH38_TO_UCSC,
    chromosome_split,
    phyloP_fraction,
    validate_human_mapping,
)


def test_missing_phylop_values_count_as_zero_with_fixed_denominator() -> None:
    values = [2.2162] * 64 + [2.216199] * 64 + [None] * 64 + [math.nan] * 63
    assert phyloP_fraction(values) == 64 / 255


def test_phyloP_requires_exact_window_length() -> None:
    with pytest.raises(AssertionError, match="255"):
        phyloP_fraction([3.0] * 254)


def test_odd_even_autosome_split() -> None:
    assert chromosome_split("chr21") == "tuning"
    assert chromosome_split("22") == "held_out"
    with pytest.raises(AssertionError, match="autosome"):
        chromosome_split("chrX")


def test_human_mapping_requires_length_and_sample_sequence_equality() -> None:
    refseq_lengths = {name: 1_000 for name in REFSEQ_GRCH38_TO_UCSC}
    ucsc_lengths = {name: 1_000 for name in REFSEQ_GRCH38_TO_UCSC.values()}
    first = next(iter(REFSEQ_GRCH38_TO_UCSC))
    validate_human_mapping(
        refseq_lengths=refseq_lengths,
        ucsc_lengths=ucsc_lengths,
        sampled_sequences=[(first, 10, 14, "AcgT", "ACGT")],
    )
    ucsc_lengths[REFSEQ_GRCH38_TO_UCSC[first]] = 999
    with pytest.raises(AssertionError, match="length mismatch"):
        validate_human_mapping(
            refseq_lengths=refseq_lengths,
            ucsc_lengths=ucsc_lengths,
            sampled_sequences=[(first, 10, 14, "ACGT", "ACGT")],
        )
