"""Resize a projected interval to a fixed length, centered on its midpoint.

Used post-projection to coerce variable-length orthologous regions back
to a fixed model-context size (255 bp). When centering would fall off
either end of the chromosome, the interval is shifted inward to preserve
the target length. Padded bases beyond the orthologous extent are not
orthologous to the human flanks — they're adjacent species genome bases —
but for a fixed-context gLM that is acceptable; the model sees a coherent
stretch of target genome anchored at the orthologous center.
"""

from __future__ import annotations

import polars as pl


def resize_to_length(
    start: int, end: int, target_len: int, chrom_size: int
) -> tuple[int, int]:
    """Resize ``[start, end)`` to length ``target_len`` around its midpoint.

    Args:
        start: 0-based half-open start coordinate (inclusive).
        end: 0-based half-open end coordinate (exclusive). Must be > start.
        target_len: desired output length (positive).
        chrom_size: chromosome size; output is clamped to ``[0, chrom_size]``.

    Returns:
        ``(new_start, new_end)`` such that
        ``new_end - new_start == target_len`` and
        ``0 <= new_start <= new_end <= chrom_size``.

    Raises:
        ValueError: if ``chrom_size < target_len`` (cannot fit),
            ``end <= start`` (empty input interval), or
            ``target_len <= 0``.
    """
    if target_len <= 0:
        raise ValueError(f"target_len must be positive: {target_len}")
    if end <= start:
        raise ValueError(f"empty interval: start={start} end={end}")
    if chrom_size < target_len:
        raise ValueError(
            f"chrom_size ({chrom_size}) < target_len ({target_len}); cannot fit"
        )

    midpoint = (start + end) // 2
    half = target_len // 2
    new_start = midpoint - half
    new_end = new_start + target_len

    if new_start < 0:
        shift = -new_start
        new_start += shift
        new_end += shift
    elif new_end > chrom_size:
        shift = new_end - chrom_size
        new_start -= shift
        new_end -= shift

    assert new_end - new_start == target_len
    assert 0 <= new_start <= new_end <= chrom_size
    return new_start, new_end


def resize_dataframe(df: pl.DataFrame, target_len: int) -> pl.DataFrame:
    """Vectorised :func:`resize_to_length` over a Polars DataFrame.

    Replaces ``t_start`` / ``t_end`` in-place (one Polars expression, no
    Python row iteration). Caller must pre-filter rows with
    ``t_src_size < target_len`` — they would otherwise produce
    ``new_end > t_src_size``, which the post-condition asserts against.

    Args:
        df: must contain ``t_start, t_end, t_src_size`` Int64 columns.
        target_len: desired output length (positive).

    Returns:
        ``df`` with ``t_start`` and ``t_end`` replaced; all other columns
        preserved in their original order.

    Asserts (per CLAUDE.md "loud failures over silent corruption"):
        - every output row has ``t_end - t_start == target_len``
        - ``0 <= t_start <= t_end <= t_src_size``
    """
    assert target_len > 0, f"target_len must be positive: {target_len}"
    if df.is_empty():
        return df

    half = target_len // 2
    midpoint = (pl.col("t_start") + pl.col("t_end")) // 2
    raw_start = midpoint - half
    # Clamp into [0, t_src_size - target_len] — equivalent to the
    # left/right-edge shift in the per-row implementation. Rows with
    # `t_src_size < target_len` violate the upper bound so are rejected
    # by the post-condition; caller filters them upstream.
    new_start = raw_start.clip(0, pl.col("t_src_size") - target_len)
    out = df.with_columns(
        t_start=new_start,
        t_end=new_start + target_len,
    )

    assert (out["t_end"] - out["t_start"] == target_len).all()
    assert (out["t_start"] >= 0).all()
    assert (out["t_end"] <= out["t_src_size"]).all()
    return out
