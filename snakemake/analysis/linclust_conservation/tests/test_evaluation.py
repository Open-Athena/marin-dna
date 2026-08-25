from marin_dna_linclust_conservation.evaluation import (
    Interval,
    collapse_intervals,
    ranked_footprint,
    unique_bases,
)


def test_collapse_and_count_0_based_half_open_intervals() -> None:
    intervals = [
        Interval("chr1", 0, 255),
        Interval("chr1", 128, 383),
        Interval("chr1", 500, 600),
        Interval("chr2", 0, 10),
    ]
    assert collapse_intervals(intervals) == [
        Interval("chr1", 0, 383),
        Interval("chr1", 500, 600),
        Interval("chr2", 0, 10),
    ]
    assert unique_bases(intervals) == 493


def test_ranked_footprint_does_not_double_count_overlapping_windows() -> None:
    curve = ranked_footprint(
        [
            (Interval("chr1", 0, 255), 0.9),
            (Interval("chr1", 128, 383), 0.8),
            (Interval("chr2", 0, 255), 0.7),
        ]
    )
    assert curve == [(1, 255, 0.9), (2, 383, 0.8), (3, 638, 0.7)]
