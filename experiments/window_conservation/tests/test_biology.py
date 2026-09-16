import numpy as np

from window_conservation.biology import coverage, fractions, merge


def test_union_and_boundary_overlap() -> None:
    intervals = merge([(2, 7), (5, 11), (15, 21), (30, 31)], 31)
    expected = np.zeros(31, dtype=bool)
    expected[2:11] = True
    expected[15:21] = True
    expected[30] = True
    mask = np.array([True, False, True])
    selected = np.repeat(mask, 10)
    assert coverage(intervals, mask, 10) == int((selected & expected[:30]).sum())
    assert np.array_equal(
        fractions(intervals, 3, 10), expected[:30].reshape(3, 10).mean(axis=1)
    )
    assert merge([], 100).shape == (0, 2)
    assert coverage(merge([], 100), np.ones(10, dtype=bool), 10) == 0
