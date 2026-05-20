"""Tests for whole-genome segmentation tiling."""

from marin_dna.pipelines.enhancer_segmentation.predict_genome import tile_chromosomes

WINDOW = 65536


def test_single_exact_window():
    tiles = tile_chromosomes({"chr1": WINDOW}, WINDOW)
    assert tiles == [("chr1", 0, WINDOW)]


def test_skips_contig_shorter_than_window():
    tiles = tile_chromosomes({"tiny": WINDOW - 1}, WINDOW)
    assert tiles == []


def test_drops_partial_tail_window():
    tiles = tile_chromosomes({"chr1": 3 * WINDOW + 100}, WINDOW)
    assert tiles == [
        ("chr1", 0, WINDOW),
        ("chr1", WINDOW, 2 * WINDOW),
        ("chr1", 2 * WINDOW, 3 * WINDOW),
    ]


def test_multiple_chromosomes():
    tiles = tile_chromosomes(
        {"chr1": 2 * WINDOW, "chr2": WINDOW, "tiny": 100},
        WINDOW,
    )
    assert tiles == [
        ("chr1", 0, WINDOW),
        ("chr1", WINDOW, 2 * WINDOW),
        ("chr2", 0, WINDOW),
    ]
