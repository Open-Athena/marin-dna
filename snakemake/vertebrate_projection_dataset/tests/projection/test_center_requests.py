from __future__ import annotations

import polars as pl
from marin_dna_vertebrate_projection.projection.requests import (
    LANDMARK_WIDTH,
    PROJECTION_POLICY,
    build_projection_requests,
)


def _anchors() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "query_name": ["a1", "a2"],
            "source_chrom": ["chr1", "chr2"],
            "source_start": [100, 1_000],
            "source_end": [355, 1_255],
            "region_label": ["cds", "ccre_non_promoter"],
        }
    )


def test_projection_requests_use_the_exact_center_base() -> None:
    requests = build_projection_requests(_anchors())

    assert requests["projection_policy"].unique().to_list() == [PROJECTION_POLICY]
    assert requests["landmark_width"].unique().to_list() == [LANDMARK_WIDTH]
    assert requests.select("projection_start", "projection_end").rows() == [
        (227, 228),
        (1_127, 1_128),
    ]
