"""Convert per-bin model predictions into windowed enhancer interval sets."""

import polars as pl

from marin_dna.data.intervals import GenomicSet


def top_quantile_bins_to_windows(
    bin_logits: pl.DataFrame,
    top_quantile: float,
    target_size: int,
) -> GenomicSet:
    """Pick the top fraction of bins by logit and resize each to a fixed window.

    Per-genome quantile thresholding: the threshold is set so that approximately
    ``top_quantile`` of bins (across the whole input) score at or above it.
    Each surviving bin is then resized to ``target_size`` bp centered on its
    midpoint, and the result is returned as a ``GenomicSet`` (so windows whose
    centers are within ``target_size`` bp of each other auto-merge).

    The resize math matches ``GenomicSet.resize`` / ``_resize_df``: when
    ``target_size - bin_size`` is odd the right side gets the extra base.
    No boundary clamping is applied -- callers should intersect the result
    with the genome's ``defined`` region to clip windows that fall off the
    chromosome ends (mirrors v17/v18).

    Args:
        bin_logits: Polars DataFrame with columns ``chrom``, ``bin_start``,
            ``bin_end``, ``logit`` -- one row per bin (typically the output of
            the segmentation prediction pipeline).
        top_quantile: Fraction of bins to keep, in (0, 1]. ``0.01`` keeps the
            top 1%.
        target_size: Output window size in bp.

    Returns:
        GenomicSet of resized, possibly-merged windows.
    """
    assert 0.0 < top_quantile <= 1.0, (
        f"top_quantile must be in (0, 1], got {top_quantile}"
    )
    assert target_size > 0, f"target_size must be positive, got {target_size}"
    required = {"chrom", "bin_start", "bin_end", "logit"}
    missing = required - set(bin_logits.columns)
    assert not missing, f"bin_logits missing required columns: {missing}"
    assert bin_logits.height > 0, "bin_logits is empty"

    threshold = bin_logits["logit"].quantile(1.0 - top_quantile)
    selected = bin_logits.filter(pl.col("logit") >= threshold)

    diff_expr = target_size - (pl.col("bin_end") - pl.col("bin_start"))
    left_adj = diff_expr // 2
    right_adj = diff_expr - left_adj

    windows = selected.select(
        pl.col("chrom"),
        (pl.col("bin_start") - left_adj).alias("start"),
        (pl.col("bin_end") + right_adj).alias("end"),
    )
    return GenomicSet(windows)
