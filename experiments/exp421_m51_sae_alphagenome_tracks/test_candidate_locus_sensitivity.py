from __future__ import annotations

import polars as pl

from candidate_locus_sensitivity import locus_exclusion_sensitivity


def test_locus_exclusion_removes_expected_rows() -> None:
    contexts = pl.DataFrame(
        {
            "panel_row": list(range(40)),
            "label": [1] * 20 + [0] * 20,
            "subset": ["5_prime_UTR_variant"] * 40,
            "exon_closest_gene_id": ["ENSG00000087086"] * 2 + ["OTHER"] * 38,
            "tss_closest_gene_id": ["OTHER"] * 38 + ["ENSG00000101981"] * 2,
        }
    )
    responses = pl.concat(
        [
            pl.DataFrame(
                {
                    "panel_row": list(range(40)),
                    "report_block": [19] * 40,
                    "feature_id": [11_928] * 40,
                    "orientation": [orientation] * 40,
                    "abs_delta": [2.0 + index / 100 for index in range(20)]
                    + [1.0 + index / 100 for index in range(20)],
                }
            )
            for orientation in ("forward", "reverse_complement")
        ]
    )
    result = locus_exclusion_sensitivity(responses, contexts)
    assert result.height == 16
    overall = result.filter(pl.col("scope") == "all_subsets")
    assert set(overall.filter(pl.col("excluded_loci") == "none")["n"]) == {40}
    assert set(overall.filter(pl.col("excluded_loci") == "FTL")["n"]) == {38}
    assert set(overall.filter(pl.col("excluded_loci") == "F9")["n"]) == {38}
    assert set(overall.filter(pl.col("excluded_loci") == "FTL+F9")["n"]) == {36}
    assert (result["rank_biserial"] > 0).all()
