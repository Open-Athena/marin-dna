"""Center-1-bp human-landmark request construction.

Source anchors remain the fixed 255 bp human intervals used for dataset
identity and train/validation splitting. ``projection_start`` and
``projection_end`` are the 0-based, half-open subinterval sent to an alignment
backend. Keeping those coordinate roles separate prevents a center-seeded
request from silently changing the source-anchor contract.
"""

import polars as pl

ANCHOR_COLUMNS = (
    "query_name",
    "source_chrom",
    "source_start",
    "source_end",
    "region_label",
)

PROJECTION_REQUEST_COLUMNS = (
    *ANCHOR_COLUMNS,
    "projection_policy",
    "landmark_width",
    "projection_start",
    "projection_end",
)


PROJECTION_POLICY = "center_1"
LANDMARK_WIDTH = 1


def build_projection_requests(anchors: pl.DataFrame) -> pl.DataFrame:
    """Attach the exact source interval to submit to HAL or MAF.

    The input and returned source coordinates are the original human anchors.
    For an anchor ``[s, s + 255)``, ``center_1`` emits the projection interval
    ``[s + 127, s + 128)``.
    """
    missing = set(ANCHOR_COLUMNS) - set(anchors.columns)
    assert not missing, f"anchors missing columns: {sorted(missing)}"
    assert anchors["query_name"].n_unique() == anchors.height
    assert (anchors["source_start"] >= 0).all()
    source_width = pl.col("source_end") - pl.col("source_start")
    widths = anchors.select(source_width.alias("width"))["width"]
    assert (widths > 0).all()

    assert (widths == 255).all(), "source anchors must be exactly 255 bp"
    center = pl.col("source_start") + source_width // 2
    projection_start = center
    projection_end = center + 1

    result = (
        anchors.select(ANCHOR_COLUMNS)
        .with_columns(
            pl.lit(PROJECTION_POLICY).alias("projection_policy"),
            pl.lit(LANDMARK_WIDTH, dtype=pl.Int64).alias("landmark_width"),
            projection_start.cast(pl.Int64).alias("projection_start"),
            projection_end.cast(pl.Int64).alias("projection_end"),
        )
        .select(PROJECTION_REQUEST_COLUMNS)
        .sort("source_chrom", "source_start", "query_name")
    )
    assert (result["projection_start"] >= result["source_start"]).all()
    assert (result["projection_end"] <= result["source_end"]).all()
    assert (
        result["projection_end"] - result["projection_start"]
        == result["landmark_width"]
    ).all()
    return result
