"""Biological and split contracts for the new RAG recipe."""

import pytest
from marin_dna_vertebrate_projection.rag.documents import (
    HUMAN,
    SEPARATOR,
    Locus,
    allele_documents,
    assemble_document,
    reverse_complement,
    select_validation,
)


def test_missing_species_are_omitted_and_genuine_n_is_preserved():
    locus = Locus("chr1", 0, 255)
    human_only = assemble_document({HUMAN: "a" * 255}, locus=locus)
    assert human_only.sequence == "a" * 255
    assert human_only.unpadded_tokens == 256
    assert human_only.padding_tokens == 9984
    available_n = assemble_document({HUMAN: "a" * 255, "bird": "N" * 255}, locus=locus)
    assert len(available_n.sequence.split(SEPARATOR)) == 2
    assert "N" * 255 in available_n.sequence
    assert available_n.unpadded_tokens == 512


def test_geometry_at_complete_cohort_and_permutation_independent_of_input_order():
    windows = {HUMAN: "A" * 255, **{f"species{i}": "C" * 255 for i in range(39)}}
    locus = Locus("chrX", 1000, 1255)
    result = assemble_document(windows, locus=locus)
    reordered = assemble_document(dict(reversed(list(windows.items()))), locus=locus)
    assert result == reordered
    assert result.padding_tokens == 0
    assert result.unpadded_tokens == 10_240
    assert result.sequence.count(SEPARATOR) == 39
    assert len(set(result.species_order)) == 40
    assert HUMAN in result.species_order
    orders = {
        assemble_document(windows, locus=Locus("chr1", i, i + 255)).species_order
        for i in range(8)
    }
    assert len(orders) == 8
    assert any(order[-1] != HUMAN for order in orders)


def test_rc_preserves_species_segments_and_syntax():
    windows = {HUMAN: "AcgTN" * 51, "bird": "tGcAn" * 51}
    locus = Locus("chr1", 200, 455)
    forward = assemble_document(windows, locus=locus, evaluation=True)
    rc = assemble_document(windows, locus=locus, evaluation=True, orientation="rc")
    assert forward.species_order == rc.species_order == ("bird", HUMAN)
    assert rc.sequence.split(SEPARATOR) == [
        reverse_complement(s) for s in forward.sequence.split(SEPARATOR)
    ]
    assert reverse_complement(reverse_complement(windows[HUMAN])) == windows[HUMAN]


def test_ref_alt_have_identical_context_order_and_padding_in_both_orientations():
    windows = {HUMAN: "a" * 255, "bird": "C" * 255, "mammal": "g" * 255}
    result = allele_documents(windows, locus=Locus("chr3", 10, 265), ref="A", alt="G")
    assert len({d.species_order for d in result.values()}) == 1
    assert len({d.padding_tokens for d in result.values()}) == 1
    for orientation, expected in (("forward", ("A", "G")), ("rc", ("T", "C"))):
        ref = result[f"ref_{orientation}"].sequence.split(SEPARATOR)
        alt = result[f"alt_{orientation}"].sequence.split(SEPARATOR)
        assert ref[:-1] == alt[:-1]
        assert [(a, b) for a, b in zip(ref[-1], alt[-1], strict=True) if a != b] == [
            expected
        ]
        assert (ref[-1][127], alt[-1][127]) == expected


def test_ref_check_fails_before_emitting_documents():
    with pytest.raises(ValueError, match="REF mismatch"):
        allele_documents(
            {HUMAN: "A" * 255}, locus=Locus("chr1", 0, 255), ref="C", alt="G"
        )


@pytest.mark.parametrize("ref,alt", [("", "A"), ("A", "A"), ("A", "AG"), ("N", "G")])
def test_unsupported_variants_fail(ref, alt):
    with pytest.raises(ValueError, match="single-base"):
        allele_documents(
            {HUMAN: "A" * 255}, locus=Locus("chr1", 0, 255), ref=ref, alt=alt
        )


def test_chr18_is_entirely_held_out_and_membership_is_preserved_elsewhere():
    catalogs = {
        "cds": [
            Locus(chrom, i, i + 255)
            for chrom in ("chr1", "chr18")
            for i in range(0, 8000, 128)
        ],
        "ncrna": [
            Locus(chrom, i, i + 255)
            for chrom in ("chr1", "chr18")
            for i in range(0, 8000, 200)
        ],
    }
    result = select_validation(catalogs, validation_rows=4)
    assert all(len(rows) == 4 for rows in result.validation.values())
    assert result.exact_cross_region_duplicates > 0
    assert all(result.excluded.values())
    for region, region_rows in catalogs.items():
        assert set(result.train[region]) | set(result.validation[region]) | set(
            result.excluded[region]
        ) == set(region_rows)
        assert set(result.train[region]) == {
            locus for locus in region_rows if locus.chrom != "chr18"
        }
        assert all(
            locus.chrom == "chr18"
            for locus in result.validation[region] + result.excluded[region]
        )
        assert not set(result.validation[region]) & set(result.excluded[region])
    reverse_inputs = {
        key: reversed(value) for key, value in reversed(list(catalogs.items()))
    }
    assert result == select_validation(reverse_inputs, validation_rows=4)


def test_small_chr18_catalog_uses_all_available_without_sampling_other_chromosomes():
    rows = [Locus("chr1", i, i + 255) for i in range(100)] + [Locus("chr18", 0, 255)]
    result = select_validation({"cds": rows}, validation_rows=2)
    assert result.validation["cds"] == (Locus("chr18", 0, 255),)
    assert len(result.train["cds"]) == 100
    empty = select_validation({"cds": rows[:-1]}, validation_rows=2)
    assert empty.validation["cds"] == ()
    assert len(empty.train["cds"]) == 100


def test_every_chr18_window_stays_out_of_training_when_validation_uses_only_a_sample():
    rows = [Locus("chr18", i, i + 255) for i in range(0, 2550, 255)] + [
        Locus("chr1", 0, 255)
    ]
    result = select_validation({"cds": rows}, validation_rows=2)
    assert result.train["cds"] == (Locus("chr1", 0, 255),)
    assert len(result.excluded["cds"]) == 8


def test_validation_duplicate_source_rows_fail():
    locus = Locus("chr1", 0, 255)
    with pytest.raises(ValueError, match="duplicate locus"):
        select_validation({"cds": [locus, locus]}, validation_rows=1)


def test_invalid_windows_and_out_of_bounds_fail():
    with pytest.raises(ValueError, match="half-open"):
        Locus("chr1", -1, 254)
    with pytest.raises(ValueError, match="UCSC"):
        Locus("1", 0, 255)
    locus = Locus("chr1", 0, 255)
    with pytest.raises(ValueError, match="outside pinned genome"):
        locus.validate_bounds({"chr1": 254})
    for windows in ({}, {"bird": "A" * 255}, {HUMAN: "A" * 254}, {HUMAN: "R" * 255}):
        with pytest.raises(ValueError):
            assemble_document(windows, locus=locus)
