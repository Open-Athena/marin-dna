from __future__ import annotations

import math

import polars as pl

from summarize_interpretation import (
    LOCI,
    annotate_loci,
    hotspot_sensitivity,
    label_stratified_score_correlations,
    locus_response_summary,
)


def test_annotate_loci_converts_external_one_based_positions() -> None:
    ftl = LOCI[0]
    frame = pl.DataFrame(
        {
            "chrom": [ftl.chrom, ftl.chrom, ftl.chrom],
            "pos": [ftl.start0 + 1, ftl.end0, ftl.end0 + 1],
        }
    )
    result = annotate_loci(frame)
    assert result["pos0"].to_list() == [
        ftl.start0,
        ftl.end0 - 1,
        ftl.end0,
    ]
    assert result["locus"].to_list() == ["FTL", "FTL", None]


def synthetic_responses() -> pl.DataFrame:
    loci_by_row = {
        0: LOCI[0],
        5: LOCI[1],
        30: LOCI[2],
        35: LOCI[3],
    }
    rows = []
    for orientation_index, orientation in enumerate(("forward", "reverse_complement")):
        scale = 1.0 + 0.1 * orientation_index
        for panel_row in range(60):
            label = int(panel_row % 5 == 0)
            subset = (
                "5_prime_UTR_variant"
                if panel_row < 30
                else "non_coding_transcript_exon_variant"
            )
            locus = loci_by_row.get(panel_row)
            chrom = locus.chrom if locus is not None else "1"
            pos = locus.start0 + 1 if locus is not None else 1_000 + panel_row
            delta = scale * (panel_row / 100 + 2.0 * label)
            if locus is not None:
                delta += 5.0
            rows.append(
                {
                    "arm": "block19-25m",
                    "budget": 25_000_200,
                    "orientation": orientation,
                    "feature_id": 9086,
                    "panel_row": panel_row,
                    "label": label,
                    "subset": subset,
                    "match_group": panel_row // 2,
                    "chrom": chrom,
                    "pos": pos,
                    "delta": delta,
                    "abs_delta": abs(delta),
                    "minus_llr_avg": 2.0 * delta + label,
                    "probe_score": 3.0 * delta - label,
                }
            )
    return annotate_loci(pl.DataFrame(rows))


def test_hotspot_sensitivity_reports_prevalence_and_lift() -> None:
    result = hotspot_sensitivity(synthetic_responses())
    assert result.height == 6 * 2 * 2
    original = result.filter(
        (pl.col("target") == "overall")
        & (pl.col("orientation") == "forward")
        & (pl.col("response") == "abs_delta")
    ).row(0, named=True)
    excluded = result.filter(
        (pl.col("target") == "overall_without_FTL_FTH1_TERC_RMRP")
        & (pl.col("orientation") == "forward")
        & (pl.col("response") == "abs_delta")
    ).row(0, named=True)
    assert original["n"] == 60
    assert original["n_positive"] == 12
    assert math.isclose(original["prevalence"], 0.2)
    assert excluded["n"] == 56
    assert excluded["n_positive"] == 8
    assert excluded["best_auprc_lift"] >= 1
    assert result.filter(
        (pl.col("welch_q") < 0)
        | (pl.col("welch_q") > 1)
        | (pl.col("mann_whitney_q") < 0)
        | (pl.col("mann_whitney_q") > 1)
    ).is_empty()


def test_label_stratified_correlations_preserve_within_label_signal() -> None:
    result = label_stratified_score_correlations(synthetic_responses())
    assert result.height == 2 * 3 * 2 * 2
    benign = result.filter(
        (pl.col("orientation") == "forward")
        & (pl.col("label_stratum") == "benign")
        & (pl.col("response") == "delta")
        & (pl.col("outcome") == "minus_llr_avg")
    ).row(0, named=True)
    assert math.isclose(benign["pearson_r"], 1)
    assert math.isclose(benign["spearman_rho"], 1)
    assert 0 <= benign["pearson_q"] <= 1
    assert 0 <= benign["spearman_q"] <= 1


def test_locus_response_summary_uses_only_named_loci() -> None:
    result = locus_response_summary(synthetic_responses())
    assert set(result["locus"].unique().to_list()) == {
        "FTL",
        "FTH1",
        "TERC",
        "RMRP",
    }
    assert result.height == 4 * 2
    assert result.filter(pl.col("label") != 1).is_empty()
    assert result.filter(pl.col("orientation_abs_delta_mass_fraction") <= 0).is_empty()
