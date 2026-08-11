"""Tests for ``marin_dna_zoonomia_projection.projection.resize``."""

from __future__ import annotations

import polars as pl
import pytest
from marin_dna_zoonomia_projection.projection.resize import (
    resize_dataframe,
    resize_to_length,
)


def test_already_target_length_no_change() -> None:
    s, e = resize_to_length(1000, 1255, target_len=255, chrom_size=100000)
    assert (s, e) == (1000, 1255)


def test_longer_input_trimmed_around_midpoint() -> None:
    # input 400 bp, target 255 → trim to 255 around mid=1200
    s, e = resize_to_length(1000, 1400, target_len=255, chrom_size=100000)
    # midpoint = (1000+1400)//2 = 1200; half = 127; new_start = 1073
    assert (s, e) == (1073, 1328)


def test_shorter_input_padded_around_midpoint() -> None:
    # input 100 bp, target 255 → pad to 255 around mid=1050
    s, e = resize_to_length(1000, 1100, target_len=255, chrom_size=100000)
    # midpoint = 1050; half = 127; new_start = 923; new_end = 1178
    assert (s, e) == (923, 1178)


def test_left_edge_shifts_inward() -> None:
    # Window centered at chr-position 50 with target 255 would extend to -77
    s, e = resize_to_length(40, 60, target_len=255, chrom_size=100000)
    assert s == 0
    assert e - s == 255


def test_right_edge_shifts_inward() -> None:
    s, e = resize_to_length(99950, 99970, target_len=255, chrom_size=100000)
    assert e == 100000
    assert e - s == 255


def test_chrom_size_equal_to_target_len_works() -> None:
    s, e = resize_to_length(0, 100, target_len=255, chrom_size=255)
    assert (s, e) == (0, 255)


def test_chrom_size_smaller_than_target_raises() -> None:
    with pytest.raises(ValueError, match="cannot fit"):
        resize_to_length(0, 100, target_len=255, chrom_size=200)


def test_empty_interval_raises() -> None:
    with pytest.raises(ValueError, match="empty interval"):
        resize_to_length(100, 100, target_len=255, chrom_size=10000)
    with pytest.raises(ValueError, match="empty interval"):
        resize_to_length(100, 90, target_len=255, chrom_size=10000)


def test_nonpositive_target_raises() -> None:
    with pytest.raises(ValueError, match="positive"):
        resize_to_length(100, 200, target_len=0, chrom_size=10000)
    with pytest.raises(ValueError, match="positive"):
        resize_to_length(100, 200, target_len=-5, chrom_size=10000)


@pytest.mark.parametrize("target_len", [1, 2, 100, 255, 256, 999])
def test_output_length_exactly_target(target_len: int) -> None:
    s, e = resize_to_length(500, 600, target_len=target_len, chrom_size=10000)
    assert e - s == target_len
    assert 0 <= s <= e <= 10000


# ----- resize_dataframe tests (vectorised; agrees with resize_to_length) -----


def _df(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "query_name": pl.Utf8,
            "t_start": pl.Int64,
            "t_end": pl.Int64,
            "t_src_size": pl.Int64,
        },
    )


def test_resize_dataframe_matches_resize_to_length_on_mixed_rows() -> None:
    """Vectorised version must produce the same coordinates as the per-row one."""
    rows = [
        # (query_name, t_start, t_end, t_src_size)
        # already 255 → no change
        {
            "query_name": "ok_already",
            "t_start": 1000,
            "t_end": 1255,
            "t_src_size": 100000,
        },
        # longer than target → trim around midpoint
        {
            "query_name": "trim",
            "t_start": 1000,
            "t_end": 1400,
            "t_src_size": 100000,
        },
        # shorter than target → pad around midpoint
        {
            "query_name": "pad",
            "t_start": 1000,
            "t_end": 1100,
            "t_src_size": 100000,
        },
        # near left edge → shift right
        {
            "query_name": "left_edge",
            "t_start": 40,
            "t_end": 60,
            "t_src_size": 100000,
        },
        # near right edge → shift left
        {
            "query_name": "right_edge",
            "t_start": 99950,
            "t_end": 99970,
            "t_src_size": 100000,
        },
    ]
    df = _df(rows)
    out = resize_dataframe(df, target_len=255)
    assert out.height == len(rows)
    for i, row in enumerate(rows):
        ref_s, ref_e = resize_to_length(
            row["t_start"], row["t_end"], 255, row["t_src_size"]
        )
        got = out.row(i, named=True)
        assert (got["t_start"], got["t_end"]) == (ref_s, ref_e), (
            f"mismatch on {row['query_name']}: got "
            f"({got['t_start']}, {got['t_end']}), want ({ref_s}, {ref_e})"
        )


def test_resize_dataframe_preserves_extra_columns() -> None:
    df = pl.DataFrame(
        [
            {
                "query_name": "q",
                "species": "Mus_musculus",
                "t_chrom": "chr1",
                "t_start": 1000,
                "t_end": 1100,
                "t_strand": "+",
                "t_src_size": 100000,
            }
        ],
        schema={
            "query_name": pl.Utf8,
            "species": pl.Utf8,
            "t_chrom": pl.Utf8,
            "t_start": pl.Int64,
            "t_end": pl.Int64,
            "t_strand": pl.Utf8,
            "t_src_size": pl.Int64,
        },
    )
    out = resize_dataframe(df, target_len=255)
    assert out.columns == df.columns  # column order preserved
    row = out.row(0, named=True)
    assert row["species"] == "Mus_musculus"
    assert row["t_chrom"] == "chr1"
    assert row["t_strand"] == "+"


def test_resize_dataframe_empty_frame_passes_through() -> None:
    schema = {
        "query_name": pl.Utf8,
        "t_start": pl.Int64,
        "t_end": pl.Int64,
        "t_src_size": pl.Int64,
    }
    df = pl.DataFrame(schema=schema)
    out = resize_dataframe(df, target_len=255)
    assert out.height == 0
    assert dict(out.schema) == schema


def test_resize_dataframe_raises_when_chrom_too_small() -> None:
    """`t_src_size < target_len` would silently produce a bad row otherwise.

    The function asserts the post-condition; caller must filter such
    rows upstream (project.smk does, on `t_src_size >= TARGET_LEN`).
    """
    df = _df([{"query_name": "tiny", "t_start": 0, "t_end": 100, "t_src_size": 200}])
    with pytest.raises(AssertionError):
        resize_dataframe(df, target_len=255)
