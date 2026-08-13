import polars as pl
import pytest
from polars.testing import assert_frame_equal

import marin_dna.data.intervals as interval_module
from marin_dna.data.intervals import INTERVAL_SCHEMA, GenomicSet


def frame(
    chrom: list[str],
    start: list[int],
    end: list[int],
) -> pl.DataFrame:
    return pl.DataFrame(
        {"chrom": chrom, "start": start, "end": end},
        schema=INTERVAL_SCHEMA,
    )


def test_public_contract_is_polars_only():
    data = frame(["1"], [0], [10])
    regions = GenomicSet(data)

    assert isinstance(regions.to_polars(), pl.DataFrame)
    assert not hasattr(regions, "to_pandas")
    assert not hasattr(interval_module, "GenomicList")
    with pytest.raises(TypeError, match="polars.DataFrame"):
        GenomicSet({"chrom": ["1"], "start": [0], "end": [10]})  # type: ignore[arg-type]


def test_constructor_normalizes_schema_sorts_and_merges():
    data = pl.DataFrame(
        {
            "chrom": ["2", "1", "1", "1"],
            "start": pl.Series([20, 10, 0, 5], dtype=pl.Int32),
            "end": pl.Series([30, 15, 5, 10], dtype=pl.Int32),
            "name": ["d", "c", "a", "b"],
        }
    )

    result = GenomicSet(data).to_polars()

    assert result.schema == INTERVAL_SCHEMA
    assert result.to_dicts() == [
        {"chrom": "1", "start": 0, "end": 15},
        {"chrom": "2", "start": 20, "end": 30},
    ]


def test_constructor_accepts_categorical_chromosomes():
    data = frame(["1"], [0], [10]).with_columns(pl.col("chrom").cast(pl.Categorical))

    assert GenomicSet(data) == GenomicSet(frame(["1"], [0], [10]))


@pytest.mark.parametrize(
    ("data", "error", "message"),
    [
        (
            pl.DataFrame({"chrom": ["1"], "start": [0]}),
            ValueError,
            "missing columns",
        ),
        (
            pl.DataFrame({"chrom": [1], "start": [0], "end": [10]}),
            TypeError,
            "chrom must",
        ),
        (
            pl.DataFrame({"chrom": ["1"], "start": [0.0], "end": [10]}),
            TypeError,
            "start must",
        ),
        (
            pl.DataFrame(
                {"chrom": ["1"], "start": [None], "end": [10]},
                schema={"chrom": pl.String, "start": pl.Int64, "end": pl.Int64},
            ),
            ValueError,
            "must not contain nulls",
        ),
        (
            frame(["1"], [10], [10]),
            ValueError,
            "start < end",
        ),
        (
            frame(["1"], [11], [10]),
            ValueError,
            "start < end",
        ),
    ],
)
def test_constructor_rejects_invalid_intervals(data, error, message):
    with pytest.raises(error, match=message):
        GenomicSet(data)


def test_empty_set_has_stable_schema_and_identities():
    empty = GenomicSet(frame([], [], []))
    nonempty = GenomicSet(frame(["1"], [0], [10]))

    assert empty.to_polars().schema == INTERVAL_SCHEMA
    assert empty.n_intervals() == 0
    assert empty.total_size() == 0
    assert empty | nonempty == nonempty
    assert empty & nonempty == empty
    assert nonempty - empty == nonempty
    assert empty - nonempty == empty


def test_union_merges_overlaps_and_adjacency():
    left = GenomicSet(frame(["1", "2"], [0, 0], [10, 5]))
    right = GenomicSet(frame(["1", "2"], [10, 4], [20, 8]))

    assert (left | right).to_polars().to_dicts() == [
        {"chrom": "1", "start": 0, "end": 20},
        {"chrom": "2", "start": 0, "end": 8},
    ]


def test_intersection_uses_strict_half_open_coordinates():
    left = GenomicSet(frame(["1", "1"], [-2, 20], [8, 30]))
    right = GenomicSet(frame(["1", "1"], [3, 8], [7, 20]))

    assert (left & right).to_polars().to_dicts() == [
        {"chrom": "1", "start": 3, "end": 7}
    ]
    assert (
        GenomicSet(frame(["1"], [0], [10])) & GenomicSet(frame(["1"], [10], [20]))
    ).n_intervals() == 0


def test_subtraction_fragments_intervals_and_preserves_negative_coordinates():
    left = GenomicSet(frame(["1", "1"], [-2, 20], [8, 30]))
    right = GenomicSet(frame(["1", "1"], [3, 22], [5, 25]))

    assert (left - right).to_polars().to_dicts() == [
        {"chrom": "1", "start": -2, "end": 3},
        {"chrom": "1", "start": 5, "end": 8},
        {"chrom": "1", "start": 20, "end": 22},
        {"chrom": "1", "start": 25, "end": 30},
    ]


def test_filter_not_overlapping_removes_whole_merged_intervals():
    regions = GenomicSet(frame(["1", "1"], [0, 20], [10, 30]))
    mask = GenomicSet(frame(["1"], [5], [6]))

    assert regions.filter_not_overlapping(mask) == GenomicSet(frame(["1"], [20], [30]))


def test_equality_repr_and_to_polars_clone():
    regions = GenomicSet(frame(["1"], [0], [10]))
    exported = regions.to_polars().with_columns(pl.lit(100).alias("end"))

    assert regions == GenomicSet(frame(["1"], [0], [10]))
    assert regions != "not a genomic set"
    assert regions.to_polars().item(0, "end") == 10
    assert exported.item(0, "end") == 100
    assert repr(regions).startswith("GenomicSet\n")


def test_size_queries_and_filtering_are_inclusive():
    regions = GenomicSet(frame(["1", "1", "1"], [0, 20, 50], [10, 40, 80]))

    assert regions.n_intervals() == 3
    assert regions.total_size() == 60
    assert regions.filter_size(min_size=10, max_size=20) == GenomicSet(
        frame(["1", "1"], [0, 20], [10, 40])
    )
    assert regions.filter_size(min_size=31).n_intervals() == 0


def test_expand_min_size_and_resize():
    regions = GenomicSet(frame(["1", "1"], [10, 35], [30, 50]))

    assert regions.expand_min_size(30) == GenomicSet(frame(["1"], [5], [58]))

    mixed = GenomicSet(frame(["1", "2"], [100, 500], [200, 850]))
    assert mixed.resize(255) == GenomicSet(frame(["1", "2"], [23, 548], [278, 803]))

    with pytest.raises(ValueError, match="min_size must be positive"):
        regions.expand_min_size(0)
    with pytest.raises(ValueError, match="target_size must be positive"):
        regions.resize(0)


def test_flank_and_random_shift():
    regions = GenomicSet(frame(["1", "2"], [5, 100], [15, 120]))

    assert regions.add_flank(10) == GenomicSet(frame(["1", "2"], [-5, 90], [25, 130]))
    assert regions.add_random_shift(5, seed=17) == regions.add_random_shift(5, seed=17)
    assert regions.add_random_shift(0, seed=17) == regions
    assert regions.add_random_shift(5, seed=17).total_size() == regions.total_size()
    with pytest.raises(ValueError, match="max_shift must be non-negative"):
        regions.add_random_shift(-1)


@pytest.mark.parametrize("suffix", [".bed", ".bed.gz"])
def test_bed_round_trip(tmp_path, suffix):
    path = tmp_path / f"regions{suffix}"
    regions = GenomicSet(frame(["1", "2"], [-5, 20], [10, 30]))

    regions.write_bed(path)

    assert GenomicSet.read_bed(path) == regions


def test_parquet_round_trip(tmp_path):
    path = tmp_path / "regions.parquet"
    regions = GenomicSet(frame(["1", "2"], [0, 20], [10, 30]))

    regions.write_parquet(path)
    loaded = GenomicSet.read_parquet(path)

    assert loaded == regions
    assert_frame_equal(loaded.to_polars(), regions.to_polars())
