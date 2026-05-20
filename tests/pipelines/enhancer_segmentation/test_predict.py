"""Tests for segmentation bedgraph tiling."""

import pytest

from marin_dna.pipelines.enhancer_segmentation.predict import tile_region

WINDOW = 16384


def test_tile_single_window():
    tiles = tile_region(0, WINDOW, WINDOW)
    assert tiles == [(0, WINDOW)]


def test_tile_multiple_windows_are_contiguous():
    tiles = tile_region(0, 5 * WINDOW, WINDOW)
    assert len(tiles) == 5
    for i, (s, e) in enumerate(tiles):
        assert e - s == WINDOW
        if i > 0:
            assert s == tiles[i - 1][1]


def test_non_multiple_length_raises():
    with pytest.raises(ValueError):
        tile_region(0, WINDOW + 1, WINDOW)


def test_zero_length_raises():
    with pytest.raises(ValueError):
        tile_region(100, 100, WINDOW)
