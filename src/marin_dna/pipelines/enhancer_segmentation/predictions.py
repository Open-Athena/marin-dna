"""Build per-(window, bin) prediction parquets shared by `train` and `evaluate`."""

import numpy as np
import polars as pl


def build_bin_predictions(val_meta: pl.DataFrame, logits: np.ndarray) -> pl.DataFrame:
    """Return one row per (window, bin), pairing `val_meta` with `logits`.

    `val_meta` has one row per window with columns
    ``genome, chrom, start, end, strand, labels`` (labels = list of per-bin
    uint8). `logits` is a (n_windows, num_bins) array (or 1D of length
    n_windows * num_bins). The returned dataframe has columns
    ``genome, chrom, start, end, strand, bin_idx, bin_start, bin_end,
    label, logit`` and `n_windows * num_bins` rows, suitable for the AUPRC /
    precision-recall tooling.
    """
    num_bins = len(val_meta["labels"][0])
    expected = len(val_meta) * num_bins
    flat_logits = np.asarray(logits).reshape(-1)
    if flat_logits.size != expected:
        raise ValueError(
            f"logits size {flat_logits.size} != val rows * num_bins "
            f"({len(val_meta)} * {num_bins} = {expected})"
        )

    window_size = int(val_meta["end"][0] - val_meta["start"][0])
    bin_size = window_size // num_bins

    labels_flat = np.asarray(val_meta["labels"].to_list(), dtype=np.uint8).reshape(-1)
    bin_idx = np.tile(np.arange(num_bins, dtype=np.int32), len(val_meta))
    window_starts = np.repeat(val_meta["start"].to_numpy(), num_bins)
    return pl.DataFrame(
        {
            "genome": np.repeat(val_meta["genome"].to_numpy(), num_bins),
            "chrom": np.repeat(val_meta["chrom"].to_numpy(), num_bins),
            "start": window_starts,
            "end": np.repeat(val_meta["end"].to_numpy(), num_bins),
            "strand": np.repeat(val_meta["strand"].to_numpy(), num_bins),
            "bin_idx": bin_idx,
            "bin_start": window_starts + bin_idx * bin_size,
            "bin_end": window_starts + (bin_idx + 1) * bin_size,
            "label": labels_flat,
            "logit": flat_logits,
        }
    )
