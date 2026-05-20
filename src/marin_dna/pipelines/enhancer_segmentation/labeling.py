"""Per-bin labeling of genomic windows against a set of positive intervals."""

import bioframe as bf
import numpy as np
import pandas as pd


def label_windows_by_bin_overlap(
    windows: pd.DataFrame,
    positives: pd.DataFrame,
    bin_size: int,
    num_bins: int,
    threshold: float = 0.5,
) -> np.ndarray:
    """Label each bin of each window by fractional overlap with ``positives``.

    A bin is labeled 1 iff its overlap with ``positives`` is at least
    ``ceil(threshold * bin_size)`` bp. At the default threshold=0.5,
    bin_size=128 the cutoff is 64 bp.

    Args:
        windows: DataFrame with columns ``chrom, start, end``. Each window must
            be exactly ``bin_size * num_bins`` bp long.
        positives: DataFrame with columns ``chrom, start, end`` (e.g. conserved
            enhancer intervals).
        bin_size: Bin width in bp (e.g. 128).
        num_bins: Number of bins per window (e.g. 128 for a 16384 bp window).
        threshold: Overlap fraction required for a bin to be labeled 1.
            Default 0.5 (i.e. >= 50% of the bin overlaps a positive).

    Returns:
        ``(len(windows), num_bins)`` ``uint8`` array of per-bin labels.
    """
    expected_size = bin_size * num_bins
    sizes = windows["end"] - windows["start"]
    if not (sizes == expected_size).all():
        raise ValueError(
            f"All windows must have size {expected_size} (bin_size * num_bins)"
        )

    n_windows = len(windows)
    if n_windows == 0:
        return np.zeros((0, num_bins), dtype=np.uint8)

    window_idx = np.repeat(np.arange(n_windows), num_bins)
    bin_offsets = np.tile(np.arange(num_bins), n_windows) * bin_size
    chroms = windows["chrom"].to_numpy()[window_idx]
    window_starts = windows["start"].to_numpy()[window_idx]
    bin_starts = window_starts + bin_offsets
    bin_ends = bin_starts + bin_size

    labels_flat = np.zeros(n_windows * num_bins, dtype=np.uint8)

    if len(positives) == 0:
        return labels_flat.reshape(n_windows, num_bins)

    min_overlap = int(np.ceil(threshold * bin_size))
    positives_df = positives[["chrom", "start", "end"]]

    # Process one chromosome at a time to cap the intermediate dataframe size:
    # a whole-genome explosion creates (n_windows * num_bins) rows, which at
    # mammalian scale is 20M+ and can OOM a modest box during bioframe.coverage.
    chrom_groups = pd.Series(chroms).groupby(chroms, sort=False).indices
    for chrom_val, row_idx in chrom_groups.items():
        chrom_positives = positives_df[positives_df["chrom"] == chrom_val]
        if len(chrom_positives) == 0:
            continue
        bins_df = pd.DataFrame(
            {
                "chrom": chrom_val,
                "start": bin_starts[row_idx],
                "end": bin_ends[row_idx],
            }
        )
        cov = bf.coverage(bins_df, chrom_positives, return_input=False)[
            "coverage"
        ].to_numpy()
        labels_flat[row_idx] = (cov >= min_overlap).astype(np.uint8)

    return labels_flat.reshape(n_windows, num_bins)
