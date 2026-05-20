import numpy as np
import polars as pl
import pytest

from marin_dna.data.bin_predictions import top_quantile_bins_to_windows


def _bins(chrom: str, n: int, bin_size: int = 128, logits=None) -> pl.DataFrame:
    """Build a contiguous run of bins on one chrom for tests."""
    starts = [i * bin_size for i in range(n)]
    ends = [s + bin_size for s in starts]
    if logits is None:
        logits = list(range(n))
    return pl.DataFrame(
        {
            "chrom": [chrom] * n,
            "bin_start": starts,
            "bin_end": ends,
            "logit": [float(x) for x in logits],
        }
    )


def test_isolated_top_bin_resized_to_target_size():
    """A single isolated high-logit bin -> one window of exactly target_size, centered on the bin."""
    # 100 noise bins in [0, 1) plus one bin with logit 1000. Top 0.5% selects only that bin.
    rng = np.random.default_rng(0)
    logits = list(rng.uniform(0, 1, size=100)) + [1000.0]
    df = _bins("chr1", 101, bin_size=128, logits=logits)
    out = top_quantile_bins_to_windows(df, top_quantile=0.005, target_size=255)
    out_df = out.to_pandas().reset_index(drop=True)
    assert len(out_df) == 1
    # Bin 100 = [12800, 12928); diff=127 -> left=63, right=64 -> [12737, 12992).
    assert out_df.iloc[0]["start"] == 12800 - 63
    assert out_df.iloc[0]["end"] == 12928 + 64
    assert out_df.iloc[0]["end"] - out_df.iloc[0]["start"] == 255


def test_adjacent_top_bins_merge():
    """Two adjacent above-threshold bins -> their resized 255bp windows merge into one."""
    rng = np.random.default_rng(0)
    # 50 noise bins in [0,1), then 2 peak bins at 1000. top_quantile=0.5/52 ~= 0.0096
    # gives quantile(0.9904) = 1000 (nearest-interpolation idx 51), so threshold >= 1000
    # selects only the 2 peak bins.
    logits = list(rng.uniform(0, 1, size=50)) + [1000.0, 1000.0]
    df = _bins("chr1", 52, bin_size=128, logits=logits)
    out = top_quantile_bins_to_windows(df, top_quantile=0.5 / 52, target_size=255)
    out_df = out.to_pandas().reset_index(drop=True)
    assert len(out_df) == 1
    # Bin 50 [6400, 6528) -> [6337, 6592). Bin 51 [6528, 6656) -> [6465, 6720).
    # Their 255bp windows overlap (6465 < 6592) -> merged: [6337, 6720).
    assert out_df.iloc[0]["start"] == 50 * 128 - 63
    assert out_df.iloc[0]["end"] == 51 * 128 + 128 + 64


def test_distant_top_bins_dont_merge():
    """Two above-threshold bins far apart -> two separate 255bp windows."""
    rng = np.random.default_rng(0)
    logits = list(rng.uniform(0, 1, size=100))
    logits[10] = 1000.0
    logits[80] = 1000.0
    df = _bins("chr1", 100, bin_size=128, logits=logits)
    # top_quantile=1/100 -> quantile(0.99) at nearest idx 98 -> value 1000 -> selects 2 bins.
    out = top_quantile_bins_to_windows(df, top_quantile=1 / 100, target_size=255)
    out_df = out.to_pandas().reset_index(drop=True)
    assert len(out_df) == 2
    assert (out_df["end"] - out_df["start"]).tolist() == [255, 255]
    assert out_df.iloc[0]["start"] == 10 * 128 - 63
    assert out_df.iloc[1]["start"] == 80 * 128 - 63


def test_resize_math_matches_genomicset_resize():
    """Window length = target_size; centering matches GenomicSet.resize / _resize_df.

    For target_size=255 and bin_size=128: diff=127, left_adj=63, right_adj=64.
    """
    rng = np.random.default_rng(0)
    logits = list(rng.uniform(0, 1, size=100)) + [1000.0]
    df = _bins("chr1", 101, bin_size=128, logits=logits)
    out = top_quantile_bins_to_windows(df, top_quantile=0.005, target_size=255)
    out_df = out.to_pandas().reset_index(drop=True)
    bin_start = 100 * 128
    bin_end = bin_start + 128
    assert out_df.iloc[0]["start"] == bin_start - 63
    assert out_df.iloc[0]["end"] == bin_end + 64


def test_quantile_is_global_across_chromosomes():
    """Per-genome quantile thresholding -> selection respects global logit distribution."""
    df = pl.concat(
        [
            _bins("chr1", 50, logits=[0.0] * 50),  # all low
            _bins("chr2", 50, logits=[10.0] * 50),  # all high
        ]
    )
    out = top_quantile_bins_to_windows(df, top_quantile=0.5, target_size=255)
    out_df = out.to_pandas().reset_index(drop=True)
    # Half-quantile threshold lies between 0 and 10; only chr2 bins survive.
    # 50 contiguous chr2 bins merge into one interval.
    assert len(out_df) == 1
    assert out_df.iloc[0]["chrom"] == "chr2"


def test_invalid_quantile_rejected():
    df = _bins("chr1", 10)
    with pytest.raises(AssertionError):
        top_quantile_bins_to_windows(df, top_quantile=0.0, target_size=255)
    with pytest.raises(AssertionError):
        top_quantile_bins_to_windows(df, top_quantile=1.5, target_size=255)


def test_invalid_target_size_rejected():
    df = _bins("chr1", 10)
    with pytest.raises(AssertionError):
        top_quantile_bins_to_windows(df, top_quantile=0.1, target_size=0)


def test_missing_columns_rejected():
    df = pl.DataFrame({"chrom": ["chr1"], "bin_start": [0], "bin_end": [128]})
    with pytest.raises(AssertionError, match="missing required columns"):
        top_quantile_bins_to_windows(df, top_quantile=0.1, target_size=255)


def test_full_quantile_selects_everything():
    """top_quantile=1.0 -> threshold is the minimum, every bin selected."""
    rng = np.random.default_rng(0)
    df = _bins("chr1", 50, logits=list(rng.uniform(0, 1, size=50)))
    out = top_quantile_bins_to_windows(df, top_quantile=1.0, target_size=255)
    out_df = out.to_pandas().reset_index(drop=True)
    # 50 contiguous bins, all selected, all 255bp resized -> merged into one interval.
    assert len(out_df) == 1
    assert out_df.iloc[0]["start"] == -63
    assert out_df.iloc[0]["end"] == 49 * 128 + 128 + 64
