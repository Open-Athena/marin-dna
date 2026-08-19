from __future__ import annotations

import polars as pl
import pytest
from marin_dna_vertebrate_projection.issue_473.pilot import (
    build_scored_anchor_catalog,
    sample_projection_pilot_anchors,
)


def _anchors() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    index = 0
    for region in ["cds", "ccre_non_promoter"]:
        for chrom in ["chr1", "chr2"]:
            for score in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
                rows.append(
                    {
                        "query_name": f"anchor-{index:03d}",
                        "source_chrom": chrom,
                        "source_start": index * 255,
                        "source_end": (index + 1) * 255,
                        "region_label": region,
                        "proportion_conserved": score,
                    }
                )
                index += 1
    return pl.DataFrame(rows)


def test_scored_catalog_retains_anchor_score_and_coordinate_contract() -> None:
    labels = pl.DataFrame(
        {
            "name": ["a", "b"],
            "chrom": ["1", "X"],
            "start": [0, 255],
            "end": [255, 510],
            "label": ["cds", "ccre_non_promoter"],
        }
    )
    scored = pl.DataFrame(
        {
            "name": ["b", "a", "excluded-before-labeling"],
            "proportion_conserved": [0.3, 0.8, 0.1],
        }
    )

    catalog = build_scored_anchor_catalog(labels, scored, min_proportion_conserved=0.2)

    assert catalog.select(
        "query_name", "source_chrom", "proportion_conserved"
    ).rows() == [("a", "chr1", 0.8), ("b", "chrX", 0.3)]
    assert (catalog["source_end"] - catalog["source_start"] == 255).all()


def test_pilot_sample_covers_every_observed_stratum_before_water_filling() -> None:
    result = sample_projection_pilot_anchors(
        _anchors(),
        regions=("cds", "ccre_non_promoter"),
        max_per_region=8,
        conservation_quantiles=2,
        seed=17,
    )

    assert result.anchors.height == 16
    assert result.selection_manifest.height == 16
    assert result.stratum_counts.height == 8
    assert set(result.stratum_counts["selected_anchors"].to_list()) == {2}
    assert set(result.anchors["conservation_quantile"].to_list()) == {1, 2}
    assert result.stratum_counts.group_by("region_label").agg(
        pl.col("selected_anchors").sum()
    ).sort("region_label")["selected_anchors"].to_list() == [8, 8]


def test_pilot_selection_is_reproducible_under_row_reordering() -> None:
    forward = sample_projection_pilot_anchors(
        _anchors(), regions=("cds",), max_per_region=10, seed=473
    )
    reverse = sample_projection_pilot_anchors(
        _anchors().reverse(), regions=("cds",), max_per_region=10, seed=473
    )
    assert forward.selection_manifest.equals(reverse.selection_manifest)


def test_pilot_keeps_all_rows_below_regional_cap() -> None:
    result = sample_projection_pilot_anchors(
        _anchors().filter(pl.col("region_label") == "cds"),
        regions=("cds",),
        max_per_region=20,
        conservation_quantiles=3,
    )
    assert result.anchors.height == 12
    assert result.stratum_counts["selected_anchors"].sum() == 12


def test_pilot_rejects_cap_too_small_to_represent_observed_strata() -> None:
    with pytest.raises(AssertionError, match="cannot represent"):
        sample_projection_pilot_anchors(
            _anchors(),
            regions=("cds",),
            max_per_region=3,
            conservation_quantiles=2,
        )


def test_pilot_rejects_missing_requested_region() -> None:
    with pytest.raises(AssertionError, match="missing requested regions"):
        sample_projection_pilot_anchors(
            _anchors(), regions=("utr3",), max_per_region=10
        )
