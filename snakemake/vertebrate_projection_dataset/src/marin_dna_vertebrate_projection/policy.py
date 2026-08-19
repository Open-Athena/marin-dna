"""Projection-policy definitions and human-landmark request construction.

Source anchors remain the fixed 255 bp human intervals used for dataset
identity and train/validation splitting. ``projection_start`` and
``projection_end`` are the 0-based, half-open subinterval sent to an alignment
backend. Keeping those coordinate roles separate prevents a center-seeded
request from silently changing the source-anchor contract.
"""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class ProjectionPolicy:
    """One source-landmark and target-span acceptance policy."""

    name: str
    landmark_width: int
    pre_resize_min_length: int
    pre_resize_max_length: int
    full_window: bool = False

    def __post_init__(self) -> None:
        assert self.name
        assert self.landmark_width > 0 and self.landmark_width % 2 == 1
        assert 0 < self.pre_resize_min_length <= self.pre_resize_max_length


FULL_WINDOW_POLICY = ProjectionPolicy(
    name="full_window",
    landmark_width=255,
    pre_resize_min_length=128,
    pre_resize_max_length=512,
    full_window=True,
)


def centered_landmark_policy(width: int) -> ProjectionPolicy:
    """Return the preregistered centered odd-width landmark policy.

    The target-span gate is 0.5--2.0 times the source landmark width. The
    integer lower bound is rounded upward so every accepted span contains at
    least one target base.
    """
    assert width > 0 and width % 2 == 1, "landmark width must be a positive odd int"
    return ProjectionPolicy(
        name=f"center_{width}",
        landmark_width=width,
        pre_resize_min_length=(width + 1) // 2,
        pre_resize_max_length=2 * width,
    )


def build_projection_requests(
    anchors: pl.DataFrame,
    policy: ProjectionPolicy,
) -> pl.DataFrame:
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

    if policy.full_window:
        assert (widths == policy.landmark_width).all(), (
            "full_window source anchors must match the policy width"
        )
        projection_start = pl.col("source_start")
        projection_end = pl.col("source_end")
    else:
        assert (widths % 2 == 1).all(), (
            "centered landmarks require odd-width source anchors"
        )
        assert (widths >= policy.landmark_width).all(), (
            "landmark cannot be wider than its source anchor"
        )
        center = pl.col("source_start") + source_width // 2
        flank = policy.landmark_width // 2
        projection_start = center - flank
        projection_end = center + flank + 1

    result = (
        anchors.select(ANCHOR_COLUMNS)
        .with_columns(
            pl.lit(policy.name).alias("projection_policy"),
            pl.lit(policy.landmark_width, dtype=pl.Int64).alias("landmark_width"),
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
