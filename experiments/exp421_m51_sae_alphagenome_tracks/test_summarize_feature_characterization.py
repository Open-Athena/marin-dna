from __future__ import annotations

import polars as pl

from summarize_feature_characterization import (
    DOMAIN_ORDER,
    FEATURE_LABELS,
    gc_profile,
    label_summary,
)


def test_label_summary_selects_one_overall_direction_and_all_domains() -> None:
    rows: list[dict[str, object]] = []
    for (block, feature_id), _ in FEATURE_LABELS.items():
        for orientation in ("forward", "reverse_complement"):
            for level in ("benign", "pathogenic"):
                rows.append(
                    {
                        "report_block": block,
                        "feature_id": feature_id,
                        "orientation": orientation,
                        "dimension": "label",
                        "level": level,
                        "n_level": 1,
                        "n_other": 9,
                        "mean_level": 2.0,
                        "mean_other": 1.0,
                        "rank_biserial": 0.2 if level == "pathogenic" else -0.2,
                        "mann_whitney_p": 0.01,
                        "mann_whitney_q": 0.02,
                        "mean_difference": 1.0,
                        "welch_p": 0.01,
                        "welch_q": 0.02,
                    }
                )
            for level in (
                "5_prime_UTR_variant",
                "missense_variant",
                "non_coding_transcript_exon_variant",
                "tss_proximal",
                "splicing",
                "distal",
                "3_prime_UTR_variant",
                "synonymous_variant",
            ):
                rows.append(
                    {
                        "report_block": block,
                        "feature_id": feature_id,
                        "orientation": orientation,
                        "dimension": "label_within_subset",
                        "level": level,
                        "n_level": 1,
                        "n_other": 9,
                        "mean_level": 2.0,
                        "mean_other": 1.0,
                        "rank_biserial": 0.2,
                        "mann_whitney_p": 0.01,
                        "mann_whitney_q": 0.02,
                        "mean_difference": 1.0,
                        "welch_p": 0.01,
                        "welch_q": 0.02,
                    }
                )
    result = label_summary(pl.DataFrame(rows))
    assert result.height == len(FEATURE_LABELS) * 2 * len(DOMAIN_ORDER)
    assert set(result["domain"]) == set(DOMAIN_ORDER)
    assert result.filter(pl.col("domain") == "Overall")["rank_biserial"].to_list() == [
        0.2
    ] * (len(FEATURE_LABELS) * 2)


def test_gc_profile_sums_c_and_g_and_centers_variant() -> None:
    rows = []
    for (block, feature_id), _ in FEATURE_LABELS.items():
        for orientation in ("forward", "reverse_complement"):
            for position in range(-31, 32):
                for base, difference in (("C", 0.1), ("G", 0.2), ("A", -0.1)):
                    rows.append(
                        {
                            "report_block": block,
                            "feature_id": feature_id,
                            "orientation": orientation,
                            "position": position,
                            "base": base,
                            "frequency_difference": difference,
                            "top_n": 100,
                            "background_n": 1000,
                        }
                    )
    result = gc_profile(pl.DataFrame(rows))
    assert result.height == len(FEATURE_LABELS) * 2 * 63
    assert set(result["relative_position"]) == set(range(-31, 32))
    assert all(
        abs(value - 0.3) < 1e-12
        for value in result["gc_frequency_difference"].to_list()
    )
