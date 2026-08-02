from __future__ import annotations

import polars as pl
import pytest

from prepare_feature1662_saturation import eligibility_metadata, select_contexts
from saturation_common import CONTEXTS_PER_CODON_POSITION


def test_eligibility_metadata_verifies_reverse_strand_alleles() -> None:
    row = {
        "assembly_name": "GRCh38",
        "transcript_consequence_count": 1,
        "biotype": "protein_coding",
        "strand": -1,
        "transcript_id": "ENST1",
        "codons": "GAA/GAG",
        "ref": "T",
        "alt": "C",
    }
    observed = eligibility_metadata(row)
    assert observed == {
        "eligibility_reason": None,
        "ref_codon": "GAA",
        "official_alt_codon": "GAG",
        "focal_codon_position": 3,
        "ref_amino_acid": "E",
        "official_alt_amino_acid": "E",
    }


def test_select_contexts_is_balanced_and_rejects_label_column() -> None:
    rows = []
    for position in (1, 2, 3):
        for index in range(130):
            rows.append(
                {
                    "panel_row": 1_000 * position + index,
                    "chrom": "1",
                    "pos": 10_000 * position + index,
                    "ref": "A",
                    "alt": "C",
                    "transcript_id": f"ENST{position:02d}{index:04d}",
                    "focal_codon_position": position,
                }
            )
    eligible = pl.DataFrame(rows)
    selected = select_contexts(eligible)
    assert selected.height == 3 * CONTEXTS_PER_CODON_POSITION
    assert (
        selected.group_by("focal_codon_position")
        .len()
        .sort("focal_codon_position")["len"]
        .to_list()
        == [CONTEXTS_PER_CODON_POSITION] * 3
    )
    with pytest.raises(AssertionError):
        select_contexts(eligible.with_columns(pl.lit(False).alias("label")))
