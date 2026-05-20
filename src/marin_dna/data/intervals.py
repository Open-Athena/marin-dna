import bioframe as bf
import numpy as np
import pandas as pd
import polars as pl

INTERVAL_COORDS = ["chrom", "start", "end"]


def _resize_df(data: pd.DataFrame, target_size: int) -> pd.DataFrame:
    """Resize intervals to *target_size* bp, centred on midpoints."""
    res = data.copy()
    size = res["end"] - res["start"]
    diff = target_size - size
    left_adj = diff // 2
    right_adj = diff - left_adj
    res["start"] = res["start"] - left_adj
    res["end"] = res["end"] + right_adj
    return res


class GenomicSet:
    """A set of genomic intervals that are always non-overlapping.

    This class represents a collection of genomic intervals (chromosome, start, end)
    with the guarantee that intervals are merged to ensure no overlaps exist within
    the set. The intervals are automatically sorted by chromosome, start, and end
    coordinates. The class supports set-like operations including union (|), intersection (&),
    and subtraction (-).

    Coordinates follow Python semantics:
    - 0-based indexing
    - start is inclusive
    - end is exclusive

    For example, chr1:0-50 represents positions 0 through 49 (50 positions total).

    Note: All intervals are assumed to be unstranded. Strand information is not
    stored or considered in any operations.

    Args:
        data: A pandas or polars DataFrame with columns ['chrom', 'start', 'end'].
            Any overlapping intervals in the input will be merged automatically,
            and the result will be sorted by chromosome, start, and end coordinates.
    """

    def __init__(self, data: pd.DataFrame | pl.DataFrame) -> None:
        if isinstance(data, pl.DataFrame):
            data = data.to_pandas()
        if len(data) == 0:
            self._data = pd.DataFrame(columns=INTERVAL_COORDS).astype(
                {"chrom": str, "start": int, "end": int}
            )
        else:
            _data = data[INTERVAL_COORDS]
            assert bf.is_bedframe(_data, raise_errors=True)
            self._data = bf.merge(_data)[INTERVAL_COORDS].sort_values(INTERVAL_COORDS)

    def __repr__(self) -> str:
        return f"GenomicSet\n{self._data}"

    def __or__(self, other: "GenomicSet") -> "GenomicSet":
        """Union of two GenomicSets.

        Returns a new GenomicSet containing all intervals from both sets,
        with overlapping intervals merged.

        Args:
            other: Another GenomicSet to union with.

        Returns:
            A new GenomicSet containing the union of intervals.
        """
        return GenomicSet(pd.concat([self._data, other._data], ignore_index=True))

    def __and__(self, other: "GenomicSet") -> "GenomicSet":
        """Intersection of two GenomicSets.

        Returns a new GenomicSet containing only the overlapping regions
        between the two sets.

        Args:
            other: Another GenomicSet to intersect with.

        Returns:
            A new GenomicSet containing the intersecting intervals.
        """
        return GenomicSet(
            bf.overlap(self._data, other._data, how="inner", return_overlap=True)[
                ["chrom", "overlap_start", "overlap_end"]
            ].rename(columns=dict(overlap_start="start", overlap_end="end"))
        )

    def __sub__(self, other: "GenomicSet") -> "GenomicSet":
        """Subtraction of two GenomicSets.

        Returns a new GenomicSet containing intervals from this set that
        do not overlap with any intervals in the other set.

        Args:
            other: Another GenomicSet to subtract from this set.

        Returns:
            A new GenomicSet containing the remaining intervals.
        """
        if len(self._data) == 0:
            return GenomicSet(self._data.copy())
        if len(other._data) == 0:
            return GenomicSet(self._data.copy())
        return GenomicSet(bf.subtract(self._data, other._data))

    def filter_not_overlapping(self, other: "GenomicSet") -> "GenomicSet":
        """Remove intervals that overlap any interval in the other set.

        Unlike subtraction (which splits intervals at overlap boundaries),
        this removes entire intervals from self that have any overlap with other.

        Args:
            other: A GenomicSet whose intervals define regions to exclude.

        Returns:
            A new GenomicSet containing only intervals from self that do not
            overlap any interval in other.
        """
        if len(self._data) == 0 or len(other._data) == 0:
            return GenomicSet(self._data.copy())
        result = bf.count_overlaps(self._data, other._data)
        return GenomicSet(result.loc[result["count"] == 0, INTERVAL_COORDS])

    def __eq__(self, other: object) -> bool:
        """Equality comparison based on underlying DataFrame equality.

        Args:
            other: Another GenomicSet to compare with.

        Returns:
            True if both GenomicSets have the same intervals, False otherwise.
        """
        if not isinstance(other, GenomicSet):
            return False
        return bool(self._data.equals(other._data))

    def to_pandas(self) -> pd.DataFrame:
        """Convert the GenomicSet to a pandas DataFrame.

        Returns:
            A pandas DataFrame with columns ['chrom', 'start', 'end']
            containing the non-overlapping intervals, sorted by chromosome,
            start, and end coordinates.
        """
        return self._data

    def to_polars(self) -> pl.DataFrame:
        """Convert the GenomicSet to a polars DataFrame.

        Returns:
            A polars DataFrame with columns ['chrom', 'start', 'end']
            containing the non-overlapping intervals, sorted by chromosome,
            start, and end coordinates.
        """
        return pl.from_pandas(self._data)

    def n_intervals(self) -> int:
        """Return the number of intervals in the GenomicSet.

        Returns:
            The number of non-overlapping intervals in the set.
        """
        return len(self._data)

    def total_size(self) -> int:
        """Return the total genomic basepairs covered by all intervals.

        Returns:
            The sum of all interval sizes (end - start) in base pairs.
            Since intervals are non-overlapping, this represents the
            actual genomic coverage.
        """
        return int((self._data["end"] - self._data["start"]).sum())

    def filter_size(
        self, min_size: int | None = None, max_size: int | None = None
    ) -> "GenomicSet":
        """Filter intervals by size.

        Keeps only intervals with size (end - start) within the specified range.

        Args:
            min_size: Minimum size in base pairs (inclusive). If None, no minimum.
            max_size: Maximum size in base pairs (inclusive). If None, no maximum.

        Returns:
            A new GenomicSet containing only intervals within the size range.
        """
        res = self._data.copy()
        res["size"] = res["end"] - res["start"]
        if min_size is not None:
            res = res[res["size"] >= min_size]
        if max_size is not None:
            res = res[res["size"] <= max_size]
        return GenomicSet(res.drop(columns=["size"]))

    def expand_min_size(self, min_size: int) -> "GenomicSet":
        """Expand intervals to at least the specified minimum size.

        Each interval is expanded by padding equally on both sides until it
        reaches at least `min_size`. Intervals that are already larger than
        `min_size` are left unchanged.

        Args:
            min_size: Minimum size (in base pairs) for each interval.

        Returns:
            A new GenomicSet with expanded intervals. Overlapping intervals
            resulting from expansion will be automatically merged.
        """
        res = self._data.copy()
        res["size"] = res["end"] - res["start"]
        res["pad"] = np.maximum(
            np.ceil((min_size - res["size"]) / 2).astype(int),
            0,
        )
        res["start"] = res["start"] - res["pad"]
        res["end"] = res["end"] + res["pad"]
        return GenomicSet(res.drop(columns=["size", "pad"]))

    def resize(self, target_size: int) -> "GenomicSet":
        """Resize all intervals to exactly `target_size` bp, centered on the midpoint.

        Intervals smaller than `target_size` are expanded; intervals larger are
        shrunk. When the size difference is odd, the center shifts 0.5 bp to
        the right.

        Args:
            target_size: Desired size (in base pairs) for every interval.

        Returns:
            A new GenomicSet with resized intervals. Overlapping intervals
            resulting from expansion will be automatically merged.
        """
        if target_size <= 0:
            raise ValueError(f"target_size must be positive, got {target_size}")
        return GenomicSet(_resize_df(self._data, target_size))

    def add_flank(self, flank: int) -> "GenomicSet":
        """Expand intervals by adding flanking regions on both sides.

        Each interval is expanded by adding `flank` base pairs to both the start
        (subtracting from start coordinate) and end (adding to end coordinate).

        Args:
            flank: Number of base pairs to add on each side of the interval.

        Returns:
            A new GenomicSet with expanded intervals. Overlapping intervals
            resulting from expansion will be automatically merged.
        """
        res = self._data.copy()
        res["start"] = res["start"] - flank
        res["end"] = res["end"] + flank
        return GenomicSet(res)

    def add_random_shift(self, max_shift: int, seed: int | None = None) -> "GenomicSet":
        """Add random shift to interval positions.

        Each interval is shifted by a random amount (in base pairs) within
        the range [-max_shift, max_shift] (inclusive). The same random shift is applied
        to both start and end positions, preserving the interval size.

        Args:
            max_shift: Maximum absolute shift value in base pairs.
            seed: Random seed for reproducible shifts. If None, shifts will be
                non-reproducible (random each time).

        Returns:
            A new GenomicSet with shifted intervals. Overlapping intervals
            resulting from shifts will be automatically merged.
        """
        rng = np.random.default_rng(seed)
        shift = rng.integers(-max_shift, max_shift, len(self._data), endpoint=True)
        res = self._data.copy()
        res["start"] = res["start"] + shift
        res["end"] = res["end"] + shift
        return GenomicSet(res)

    @classmethod
    def read_bed(cls, path: str) -> "GenomicSet":
        """Read intervals from a BED file.

        Args:
            path: Path to the BED file (can be gzipped).

        Returns:
            A new GenomicSet with intervals from the file.
        """
        return cls(
            pd.read_csv(
                path, sep="\t", header=None, names=INTERVAL_COORDS, dtype={"chrom": str}
            )
        )

    @classmethod
    def read_parquet(cls, path: str) -> "GenomicSet":
        """Read intervals from a parquet file.

        Args:
            path: Path to the parquet file.

        Returns:
            A new GenomicSet with intervals from the file.
        """
        return cls(pd.read_parquet(path))

    def write_bed(self, path: str) -> None:
        """Write intervals to a BED file.

        Args:
            path: Path to the output BED file.
        """
        self._data.to_csv(path, sep="\t", header=False, index=False)

    def write_parquet(self, path: str) -> None:
        """Write intervals to a parquet file.

        Args:
            path: Path to the output parquet file.
        """
        self._data.to_parquet(path, index=False)


class GenomicList:
    """An ordered list of genomic intervals that preserves each element's identity.

    Unlike GenomicSet, intervals are never merged -- two overlapping intervals
    remain as separate rows.  This is the right abstraction when each interval
    represents an independent genomic element (e.g. enhancers resized to a
    fixed window) and we need per-element filtering.

    Coordinates follow Python semantics (0-based, half-open).

    Args:
        data: A pandas or polars DataFrame with at least columns
            ['chrom', 'start', 'end'].  Only these columns are kept.
    """

    def __init__(self, data: pd.DataFrame | pl.DataFrame) -> None:
        if isinstance(data, pl.DataFrame):
            data = data.to_pandas()
        self._data = data[INTERVAL_COORDS].reset_index(drop=True)

    def __repr__(self) -> str:
        return f"GenomicList ({len(self._data)} intervals)\n{self._data}"

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GenomicList):
            return False
        return bool(self._data.equals(other._data))

    # -- transforms --

    def resize(self, target_size: int) -> "GenomicList":
        """Resize every interval to exactly *target_size* bp, centred on
        its midpoint.  Each element is handled independently."""
        if target_size <= 0:
            raise ValueError(f"target_size must be positive, got {target_size}")
        return GenomicList(_resize_df(self._data, target_size))

    # -- filters --

    def filter_size(
        self, min_size: int = 0, max_size: int = np.iinfo(np.int64).max
    ) -> "GenomicList":
        """Keep only intervals whose size falls in [min_size, max_size]."""
        size = self._data["end"] - self._data["start"]
        return GenomicList(self._data.loc[size.between(min_size, max_size)])

    def filter_not_overlapping(self, regions: GenomicSet) -> "GenomicList":
        """Drop intervals that overlap any region in *regions*."""
        if len(self._data) == 0 or len(regions._data) == 0:
            return GenomicList(self._data.copy())
        counts = bf.count_overlaps(self._data, regions._data)
        return GenomicList(self._data.loc[counts["count"] == 0])

    def filter_within(self, regions: GenomicSet) -> "GenomicList":
        """Keep only intervals fully contained within *regions*.

        An interval is kept when its coverage by *regions* equals its
        full length (i.e. no part falls outside *regions*)."""
        if len(self._data) == 0:
            return GenomicList(self._data.copy())
        if len(regions._data) == 0:
            return GenomicList(self._data.iloc[:0].copy())
        cov = bf.coverage(self._data, regions._data, return_input=False)
        size = self._data["end"] - self._data["start"]
        return GenomicList(self._data.loc[cov["coverage"] >= size])

    # -- conversions --

    def to_pandas(self) -> pd.DataFrame:
        return self._data

    def to_polars(self) -> pl.DataFrame:
        return pl.from_pandas(self._data)

    # -- I/O --

    def write_bed(self, path: str) -> None:
        self._data.to_csv(path, sep="\t", header=False, index=False)

    @classmethod
    def from_parquet(cls, path: str) -> "GenomicList":
        return cls(pd.read_parquet(path))
