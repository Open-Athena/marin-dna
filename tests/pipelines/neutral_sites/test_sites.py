"""Tests for the GPN-Star neutral-site construction library."""

from __future__ import annotations

import gzip

import numpy as np
import pandas as pd
import pytest

from marin_dna.pipelines.neutral_sites.sites import (
    contiguous_runs,
    enumerate_positions,
    neutral_mask,
    parse_rmsk,
)


class TestContiguousRuns:
    def test_empty_and_all_false(self):
        assert contiguous_runs(np.array([], dtype=bool)) == []
        assert contiguous_runs(np.zeros(5, dtype=bool)) == []

    def test_all_true(self):
        assert contiguous_runs(np.ones(4, dtype=bool)) == [(0, 4)]

    def test_internal_run(self):
        # F T T F F  -> [1, 3)
        assert contiguous_runs(np.array([0, 1, 1, 0, 0], dtype=bool)) == [(1, 3)]

    def test_runs_at_both_edges(self):
        # T F T  -> [0,1) and [2,3)
        assert contiguous_runs(np.array([1, 0, 1], dtype=bool)) == [(0, 1), (2, 3)]

    def test_multiple_runs(self):
        mask = np.array([1, 1, 0, 1, 0, 0, 1, 1, 1], dtype=bool)
        assert contiguous_runs(mask) == [(0, 2), (3, 4), (6, 9)]

    def test_runs_partition_the_true_bases(self):
        rng = np.random.default_rng(0)
        mask = rng.random(1000) < 0.3
        runs = contiguous_runs(mask)
        recovered = np.zeros_like(mask)
        for s, e in runs:
            assert not recovered[s:e].any(), "runs overlap"
            recovered[s:e] = True
        assert np.array_equal(recovered, mask)


class TestNeutralMask:
    def test_threshold_and_phastcons(self):
        phylop = np.array([0.0, 0.05, 0.2, -0.05, -0.2])
        phastcons = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        # |phylop| < 0.1 AND phastcons == 0
        assert neutral_mask(phylop, phastcons, 0.1).tolist() == [
            True,
            True,
            False,
            True,
            False,
        ]

    def test_phastcons_nonzero_excluded(self):
        phylop = np.zeros(3)
        phastcons = np.array([0.0, 0.01, 1.0])
        assert neutral_mask(phylop, phastcons, 0.1).tolist() == [True, False, False]

    def test_nan_excluded(self):
        # NaN in either track (bigWig "no data") must fail the mask.
        phylop = np.array([0.0, np.nan, 0.0])
        phastcons = np.array([0.0, 0.0, np.nan])
        assert neutral_mask(phylop, phastcons, 0.1).tolist() == [True, False, False]


def _write_rmsk(path, rows: list[tuple[str, int, int, str, str]]) -> None:
    """Write a minimal gzipped UCSC rmsk.txt with the bin-column layout.

    ``rows`` are ``(genoName, genoStart, genoEnd, repName, repClass)``; all the
    other UCSC columns are filled with dummies at the right positions.
    """
    lines = []
    for genoName, start, end, repName, repClass in rows:
        cols = ["0"] * 17
        cols[0] = "585"  # bin
        cols[1] = "1000"  # swScore
        cols[5] = genoName
        cols[6] = str(start)
        cols[7] = str(end)
        cols[9] = "+"  # strand
        cols[10] = repName
        cols[11] = repClass
        cols[12] = "fam"  # repFamily
        lines.append("\t".join(cols))
    with gzip.open(path, "wt") as fh:
        fh.write("\n".join(lines) + "\n")


class TestParseRmsk:
    def test_excludes_simple_and_low_complexity(self, tmp_path):
        p = tmp_path / "rmsk.txt.gz"
        _write_rmsk(
            p,
            [
                ("chr1", 100, 200, "AluY", "SINE"),
                ("chr1", 300, 350, "(TA)n", "Simple_repeat"),
                ("chr2", 50, 80, "L2", "LINE"),
                ("chrX", 10, 20, "A-rich", "Low_complexity"),
            ],
        )
        df = parse_rmsk(str(p))
        # Simple_repeat and Low_complexity dropped; SINE/LINE kept.
        assert list(df["chrom"]) == ["chr1", "chr2"]
        assert list(df["start"]) == [100, 50]
        assert list(df["end"]) == [200, 80]
        assert list(df["name"]) == ["AluY", "L2"]
        # chr prefix retained (BED-land); coords are the raw 0-based half-open.
        assert df.columns.tolist() == ["chrom", "start", "end", "name"]

    def test_custom_exclude_set(self, tmp_path):
        p = tmp_path / "rmsk.txt.gz"
        _write_rmsk(
            p,
            [
                ("chr1", 100, 200, "AluY", "SINE"),
                ("chr1", 300, 350, "L1", "LINE"),
            ],
        )
        df = parse_rmsk(str(p), exclude_classes=frozenset({"SINE"}))
        assert list(df["name"]) == ["L1"]


class TestEnumeratePositions:
    def _genome(self) -> dict[str, str]:
        # 0-based:        0123456789
        return {"1": "ACGTNNACGT", "2": "acgtac"}

    def _get_seq(self):
        g = self._genome()
        return lambda chrom, start, end: g[chrom][start:end]

    def test_basic_1based_and_acgt_filter(self):
        # chr1[2,7) -> bases at 0-based 2..6 = G T N N A; N dropped.
        intervals = pd.DataFrame({"chrom": ["chr1"], "start": [2], "end": [7]})
        out = enumerate_positions(intervals, self._get_seq(), {"1", "2"})
        assert list(out["chrom"]) == ["1", "1", "1"]
        # 1-based pos: index 2->3 (G), 3->4 (T), 6->7 (A). 4,5 are N -> dropped.
        assert list(out["pos"]) == [3, 4, 7]
        assert list(out["ref"]) == ["G", "T", "A"]

    def test_uppercases_softmasked(self):
        intervals = pd.DataFrame({"chrom": ["chr2"], "start": [0], "end": [6]})
        out = enumerate_positions(intervals, self._get_seq(), {"1", "2"})
        assert list(out["ref"]) == ["A", "C", "G", "T", "A", "C"]
        assert list(out["pos"]) == [1, 2, 3, 4, 5, 6]

    def test_chrom_filter_skips_unlisted(self):
        intervals = pd.DataFrame(
            {"chrom": ["chr1", "chrM"], "start": [0, 0], "end": [4, 4]}
        )
        # genome has no "M"; if it weren't filtered the get_seq lookup would
        # KeyError. chroms={"1"} must skip chrM.
        out = enumerate_positions(intervals, self._get_seq(), {"1"})
        assert set(out["chrom"]) == {"1"}
        assert list(out["pos"]) == [1, 2, 3, 4]

    def test_length_mismatch_asserts(self):
        # A get_seq that under-returns must trip the defensive length assert.
        bad = lambda chrom, start, end: "AC"  # noqa: E731
        intervals = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [5]})
        with pytest.raises(AssertionError, match="expected"):
            enumerate_positions(intervals, bad, {"1"})
