from __future__ import annotations

import numpy as np

from analyze_candidate_mechanism import bh_adjust, codon_change_position
from annotate_missense_vep import flatten_result, variant_input


def test_bh_adjust_known_values() -> None:
    observed = bh_adjust(np.array([0.01, 0.04, 0.03, 0.20]))
    np.testing.assert_allclose(observed, [0.04, 0.0533333333, 0.0533333333, 0.20])


def test_codon_change_position() -> None:
    assert codon_change_position("cCg/cTg") == 2
    assert codon_change_position("Aaa/Gaa") == 1
    assert codon_change_position("aaC/aaT") == 3
    assert codon_change_position(None) is None
    assert codon_change_position("AAA/CCC") is None


def test_variant_input_uses_panel_row_as_stable_id() -> None:
    row = {"chrom": "7", "pos": 75_981_558, "panel_row": 10022, "ref": "C", "alt": "T"}
    assert variant_input(row) == "7 75981558 10022 C T . . ."


def test_flatten_result_picks_transcript_and_clinical_fields() -> None:
    result = {
        "id": "10022",
        "assembly_name": "GRCh38",
        "most_severe_consequence": "missense_variant",
        "transcript_consequences": [
            {
                "gene_id": "ENSG1",
                "gene_symbol": "GENE",
                "transcript_id": "ENST1",
                "canonical": 1,
                "consequence_terms": ["missense_variant"],
                "codons": "cCg/cTg",
                "amino_acids": "P/L",
                "sift_score": 0.01,
                "polyphen_score": 0.155,
                "blosum62": -3,
            }
        ],
        "colocated_variants": [
            {"id": "rs1", "clin_sig": ["likely_benign"]},
            {"id": "COSV1"},
        ],
    }
    flattened = flatten_result(result)
    assert flattened["panel_row"] == 10022
    assert flattened["gene_symbol"] == "GENE"
    assert flattened["canonical"] is True
    assert flattened["clinical_significance"] == "likely_benign"
    assert flattened["known_variant_ids"] == "COSV1,rs1"
