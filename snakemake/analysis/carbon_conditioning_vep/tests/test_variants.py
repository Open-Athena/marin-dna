import pandas as pd
import pytest
from marin_dna_carbon_conditioning_vep.variants import (
    centered_window_bounds,
    materialize_variant_windows,
    replace_centered_allele,
    validate_mendelian_dataset,
    validate_reference_contigs,
)


def _matched_fixture() -> pd.DataFrame:
    rows = []
    for group, chrom in ((10, "1"), (20, "X")):
        for member in range(10):
            rows.append(
                {
                    "chrom": chrom,
                    "pos": 5 + member,
                    "ref": "A",
                    "alt": "C",
                    "label": member == 0,
                    "subset": "missense_variant",
                    "match_group": group,
                }
            )
    return pd.DataFrame(rows)


def test_validate_mendelian_group_and_split_contract() -> None:
    validated = validate_mendelian_dataset(
        _matched_fixture(),
        expected_rows=20,
        expected_positives=2,
        expected_groups=2,
        expected_group_size=10,
        expected_chroms={"1", "X"},
    )
    assert validated["variant_id"].is_unique
    assert validated.groupby("match_group")["label"].sum().eq(1).all()


def test_one_based_coordinate_conversion_and_centered_replacement() -> None:
    start, end, variant_index = centered_window_bounds(5, 8)
    assert (start, end, variant_index) == (0, 8, 4)
    ref = "AAAACAAA"
    alt = replace_centered_allele(ref, variant_index, "C", "G")
    assert alt == "AAAAGAAA"
    assert sum(a != b for a, b in zip(ref, alt, strict=True)) == 1


def test_boundary_and_reference_failures_remove_complete_groups() -> None:
    dataset = pd.DataFrame(
        [
            {
                "variant_id": "1:5:T>G",
                "chrom": "1",
                "pos": 5,
                "ref": "T",
                "alt": "G",
                "match_group": 1,
            },
            {
                "variant_id": "1:6:A>C",
                "chrom": "1",
                "pos": 6,
                "ref": "A",
                "alt": "C",
                "match_group": 1,
            },
            {
                "variant_id": "1:7:A>C",
                "chrom": "1",
                "pos": 7,
                "ref": "A",
                "alt": "C",
                "match_group": 2,
            },
            {
                "variant_id": "1:8:A>C",
                "chrom": "1",
                "pos": 8,
                "ref": "A",
                "alt": "C",
                "match_group": 2,
            },
        ]
    )
    sequence = "AAAACAAAAAAAAAAAAAAA"

    def genome(chrom: str, start: int, end: int, strand: str) -> str:
        assert chrom == "1" and strand == "+"
        return sequence[start:end]

    windows, failures = materialize_variant_windows(
        dataset,
        genome,
        {"1": len(sequence)},
        window_size=8,
    )
    assert set(windows["match_group"]) == {2}
    assert set(failures["match_group"]) == {1}
    assert failures.iloc[0]["reason"] == "reference_allele"


def test_boundary_failure_and_contig_validation() -> None:
    assert centered_window_bounds(1, 8) == (-4, 4, 4)
    with pytest.raises(AssertionError, match="length mismatches"):
        validate_reference_contigs({"1": 9}, {"1": 10})
