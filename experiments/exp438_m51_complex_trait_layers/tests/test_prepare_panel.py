from __future__ import annotations

from prepare_panel import EXPECTED_SUBSET_COUNTS


def test_expected_subset_contract_has_fixed_prevalence() -> None:
    assert set(EXPECTED_SUBSET_COUNTS) == {
        "3_prime_UTR_variant",
        "5_prime_UTR_variant",
        "distal",
        "missense_variant",
        "non_coding_transcript_exon_variant",
        "splicing",
        "synonymous_variant",
        "tss_proximal",
    }
    assert all(
        rows == 10 * positives for rows, positives in EXPECTED_SUBSET_COUNTS.values()
    )
