"""Polars-native sets of 0-based, half-open genomic intervals."""

from pathlib import Path

import numpy as np
import polars as pl
import polars_bio as pb

INTERVAL_COORDS = ["chrom", "start", "end"]
INTERVAL_SCHEMA = pl.Schema(
    {
        "chrom": pl.String,
        "start": pl.Int64,
        "end": pl.Int64,
    }
)


def _empty_intervals() -> pl.DataFrame:
    return pl.DataFrame(schema=INTERVAL_SCHEMA)


def _mark_zero_based(data: pl.DataFrame) -> pl.DataFrame:
    """Attach the coordinate metadata required by polars-bio."""
    coordinate_metadata = object.__getattribute__(data, "config_meta")
    coordinate_metadata.set(coordinate_system_zero_based=True)
    return data


def _normalize_intervals(data: pl.DataFrame) -> pl.DataFrame:
    """Validate and normalize a Polars interval frame."""
    if not isinstance(data, pl.DataFrame):
        raise TypeError(
            f"GenomicSet requires a polars.DataFrame, got {type(data).__name__}"
        )

    missing = set(INTERVAL_COORDS) - set(data.columns)
    if missing:
        raise ValueError(f"interval DataFrame is missing columns: {sorted(missing)}")

    data = data.select(INTERVAL_COORDS)
    if data.is_empty():
        return _mark_zero_based(data.cast(INTERVAL_SCHEMA))

    chrom_dtype = data.schema["chrom"]
    if chrom_dtype not in (pl.String, pl.Categorical):
        raise TypeError(
            f"chrom must have String or Categorical dtype, got {chrom_dtype}"
        )
    for column in ("start", "end"):
        dtype = data.schema[column]
        if not dtype.is_integer():
            raise TypeError(f"{column} must have an integer dtype, got {dtype}")

    if any(data.null_count().row(0)):
        raise ValueError("interval coordinates must not contain nulls")
    if data.select((pl.col("start") >= pl.col("end")).any()).item():
        raise ValueError("intervals must satisfy start < end")

    return _mark_zero_based(data.cast(INTERVAL_SCHEMA))


def _require_frame(result: object, operation: str) -> pl.DataFrame:
    if not isinstance(result, pl.DataFrame):
        raise TypeError(f"polars-bio {operation} returned {type(result).__name__}")
    return result


def _merge_intervals(data: pl.DataFrame) -> pl.DataFrame:
    """Merge overlapping and adjacent intervals with polars-bio."""
    if data.is_empty():
        return data
    merged = _require_frame(
        pb.merge(
            _mark_zero_based(data),
            min_dist=1,
            output_type="polars.DataFrame",
        ),
        "merge",
    )
    return _mark_zero_based(merged.select(INTERVAL_COORDS).sort(INTERVAL_COORDS))


def _resize_frame(data: pl.DataFrame, target_size: int) -> pl.DataFrame:
    size = pl.col("end") - pl.col("start")
    difference = pl.lit(target_size) - size
    left_adjustment = difference // 2
    right_adjustment = difference - left_adjustment
    return data.with_columns(
        (pl.col("start") - left_adjustment).alias("start"),
        (pl.col("end") + right_adjustment).alias("end"),
    )


class GenomicSet:
    """A merged set of unstranded genomic intervals.

    The underlying data is always a Polars DataFrame with chrom, start,
    and end columns. Coordinates are 0-based and half-open. Intervals must
    satisfy start < end. Overlapping and adjacent input rows are merged and
    sorted during construction.

    Negative coordinates are allowed for intermediate transformations such as
    flanking and resizing. Intersect with chromosome bounds before sequence
    extraction.

    Args:
        data: Polars DataFrame containing chrom, start, and end.
            Extra columns are ignored.
    """

    def __init__(self, data: pl.DataFrame) -> None:
        self._data = _merge_intervals(_normalize_intervals(data))

    def __repr__(self) -> str:
        return f"GenomicSet\n{self._data}"

    def __or__(self, other: "GenomicSet") -> "GenomicSet":
        """Return the union of two interval sets."""
        return GenomicSet(pl.concat([self._data, other._data]))

    def __and__(self, other: "GenomicSet") -> "GenomicSet":
        """Return strict half-open intersections between two interval sets."""
        if self._data.is_empty() or other._data.is_empty():
            return GenomicSet(_empty_intervals())

        overlaps = _require_frame(
            pb.overlap(
                _mark_zero_based(self._data),
                _mark_zero_based(other._data),
                output_type="polars.DataFrame",
            ),
            "overlap",
        )
        intersections = overlaps.select(
            pl.col("chrom_1").alias("chrom"),
            pl.max_horizontal("start_1", "start_2").alias("start"),
            pl.min_horizontal("end_1", "end_2").alias("end"),
        )
        return GenomicSet(intersections)

    def __sub__(self, other: "GenomicSet") -> "GenomicSet":
        """Subtract the second interval set from the first."""
        if self._data.is_empty() or other._data.is_empty():
            return GenomicSet(self._data.clone())

        result = _require_frame(
            pb.subtract(
                _mark_zero_based(self._data),
                _mark_zero_based(other._data),
                output_type="polars.DataFrame",
            ),
            "subtract",
        )
        return GenomicSet(result.select(INTERVAL_COORDS))

    def filter_not_overlapping(self, other: "GenomicSet") -> "GenomicSet":
        """Remove every interval that overlaps the other set."""
        if self._data.is_empty() or other._data.is_empty():
            return GenomicSet(self._data.clone())

        counts = _require_frame(
            pb.count_overlaps(
                _mark_zero_based(self._data),
                _mark_zero_based(other._data),
                output_type="polars.DataFrame",
            ),
            "count_overlaps",
        )
        return GenomicSet(counts.filter(pl.col("count") == 0).select(INTERVAL_COORDS))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GenomicSet):
            return False
        return self._data.equals(other._data)

    def to_polars(self) -> pl.DataFrame:
        """Return a clone of the normalized interval DataFrame."""
        return self._data.clone()

    def n_intervals(self) -> int:
        """Return the number of merged intervals."""
        return self._data.height

    def total_size(self) -> int:
        """Return the number of covered bases."""
        return int(
            self._data.select((pl.col("end") - pl.col("start")).sum()).item() or 0
        )

    def filter_size(
        self, min_size: int | None = None, max_size: int | None = None
    ) -> "GenomicSet":
        """Keep intervals whose size is within the inclusive bounds."""
        predicate = pl.lit(True)
        size = pl.col("end") - pl.col("start")
        if min_size is not None:
            predicate &= size >= min_size
        if max_size is not None:
            predicate &= size <= max_size
        return GenomicSet(self._data.filter(predicate))

    def expand_min_size(self, min_size: int) -> "GenomicSet":
        """Symmetrically expand intervals to at least min_size bases."""
        if min_size <= 0:
            raise ValueError(f"min_size must be positive, got {min_size}")
        size = pl.col("end") - pl.col("start")
        padding = ((pl.lit(min_size) - size).clip(lower_bound=0) + 1) // 2
        return GenomicSet(
            self._data.with_columns(
                (pl.col("start") - padding).alias("start"),
                (pl.col("end") + padding).alias("end"),
            )
        )

    def resize(self, target_size: int) -> "GenomicSet":
        """Resize intervals around their midpoints."""
        if target_size <= 0:
            raise ValueError(f"target_size must be positive, got {target_size}")
        return GenomicSet(_resize_frame(self._data, target_size))

    def add_flank(self, flank: int) -> "GenomicSet":
        """Add flank bases to both sides of each interval."""
        return GenomicSet(
            self._data.with_columns(
                (pl.col("start") - flank).alias("start"),
                (pl.col("end") + flank).alias("end"),
            )
        )

    def add_random_shift(self, max_shift: int, seed: int | None = None) -> "GenomicSet":
        """Shift each interval by a reproducible random offset."""
        if max_shift < 0:
            raise ValueError(f"max_shift must be non-negative, got {max_shift}")
        rng = np.random.default_rng(seed)
        shifts = pl.Series(
            "shift",
            rng.integers(
                -max_shift,
                max_shift,
                self._data.height,
                endpoint=True,
            ),
            dtype=pl.Int64,
        )
        return GenomicSet(
            self._data.with_columns(
                (pl.col("start") + shifts).alias("start"),
                (pl.col("end") + shifts).alias("end"),
            )
        )

    @classmethod
    def read_bed(cls, path: str | Path) -> "GenomicSet":
        """Read a three-column BED file, including gzip-compressed input."""
        return cls(
            pl.read_csv(
                path,
                separator="\t",
                has_header=False,
                new_columns=INTERVAL_COORDS,
                schema_overrides=INTERVAL_SCHEMA,
            )
        )

    @classmethod
    def read_parquet(cls, path: str | Path) -> "GenomicSet":
        """Read interval coordinates from Parquet."""
        return cls(pl.read_parquet(path))

    def write_bed(self, path: str | Path) -> None:
        """Write a three-column BED file and gzip paths ending in .gz."""
        if str(path).endswith(".gz"):
            self._data.write_csv(
                path,
                separator="\t",
                include_header=False,
                compression="gzip",
            )
            return
        self._data.write_csv(path, separator="\t", include_header=False)

    def write_parquet(self, path: str | Path) -> None:
        """Write normalized interval coordinates to Parquet."""
        self._data.write_parquet(path)
