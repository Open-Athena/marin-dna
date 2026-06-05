"""Tests for the GPN-Star neutral-site construction library."""

from __future__ import annotations

import gzip

import numpy as np
import pandas as pd
import pyBigWig
import pytest

from marin_dna.pipelines.neutral_sites.sites import (
    annotate_pentanucleotide,
    contiguous_runs,
    enumerate_positions,
    neutral_mask,
    parse_rmsk,
    scan_neutral_intervals,
    subsample_per_context,
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

    def test_dedups_overlapping_intervals(self):
        # Two chr1 intervals overlapping at 0-based 2-3; shared positions appear
        # once. g["1"]="ACGTNNACGT": [0,4)->pos 1-4 (ACGT), [2,6)->pos 3-4 (GT;
        # the N's at 4-5 drop). Union, deduped = pos 1-4.
        intervals = pd.DataFrame(
            {"chrom": ["chr1", "chr1"], "start": [0, 2], "end": [4, 6]}
        )
        out = enumerate_positions(intervals, self._get_seq(), {"1"})
        assert list(out["pos"]) == [1, 2, 3, 4]
        assert list(out["ref"]) == ["A", "C", "G", "T"]
        assert out.duplicated(["chrom", "pos"]).sum() == 0

    def test_length_mismatch_asserts(self):
        # A get_seq that under-returns must trip the defensive length assert.
        bad = lambda chrom, start, end: "AC"  # noqa: E731
        intervals = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [5]})
        with pytest.raises(AssertionError, match="expected"):
            enumerate_positions(intervals, bad, {"1"})


def _write_bw(path, chrom: str, length: int, values: list[float]) -> None:
    """Write a tiny per-base bigWig: ``values[i]`` at 0-based position ``i``."""
    bw = pyBigWig.open(str(path), "w")
    bw.addHeader([(chrom, length)])
    bw.addEntries(chrom, 0, values=[float(v) for v in values], span=1, step=1)
    bw.close()


class TestScanNeutralIntervals:
    def test_stitches_run_across_window_boundary(self, tmp_path):
        # chr1 (len 10): neutral (phyloP 0.0, phastCons 0) at 1..6, non-neutral
        # (phyloP 2.0) at 0,7,8,9. With window_size=5 the run [1,7) splits at the
        # 5-boundary into two ADJACENT intervals — exercises the per-window
        # `win_start + s` offset and that no base is lost/duplicated at the seam.
        pp, pc = tmp_path / "phylop.bw", tmp_path / "phast.bw"
        _write_bw(pp, "chr1", 10, [2, 0, 0, 0, 0, 0, 0, 2, 2, 2])
        _write_bw(pc, "chr1", 10, [0] * 10)
        df = scan_neutral_intervals(str(pp), str(pc), ["1"], 0.1, window_size=5)
        assert list(zip(df["chrom"], df["start"], df["end"])) == [
            ("chr1", 1, 5),
            ("chr1", 5, 7),
        ]

    def test_phastcons_nonzero_breaks_the_run(self, tmp_path):
        # A phastCons!=0 base in the middle of a low-phyloP stretch splits it.
        pp, pc = tmp_path / "phylop.bw", tmp_path / "phast.bw"
        _write_bw(pp, "chr1", 6, [0, 0, 0, 0, 0, 0])
        _write_bw(pc, "chr1", 6, [0, 0, 1, 0, 0, 0])  # conserved at idx 2
        df = scan_neutral_intervals(str(pp), str(pc), ["1"], 0.1, window_size=10)
        assert list(zip(df["chrom"], df["start"], df["end"])) == [
            ("chr1", 0, 2),
            ("chr1", 3, 6),
        ]

    def test_skips_contig_absent_from_a_track(self, tmp_path):
        # A requested chrom missing from the bigWigs is skipped, not crashed.
        pp, pc = tmp_path / "phylop.bw", tmp_path / "phast.bw"
        _write_bw(pp, "chr1", 4, [0, 0, 0, 0])
        _write_bw(pc, "chr1", 4, [0, 0, 0, 0])
        df = scan_neutral_intervals(str(pp), str(pc), ["1", "2"], 0.1, window_size=5)
        assert list(zip(df["chrom"], df["start"], df["end"])) == [("chr1", 0, 4)]


class TestAnnotatePentanucleotide:
    def _get_seq(self, genome: dict[str, str]):
        return lambda chrom, start, end: genome[chrom][start:end]

    def test_basic_5mer_and_centering(self):
        # "1": A C G T A C G T A C G T  (0-based 0..11)
        g = {"1": "ACGTACGTACGT"}
        sites = pd.DataFrame({"chrom": ["1", "1"], "pos": [5, 8], "ref": ["A", "T"]})
        out = annotate_pentanucleotide(sites, self._get_seq(g))
        # pos 5 -> center0 4 (A); 5-mer 0-based [2,7) = GTACG
        # pos 8 -> center0 7 (T); 5-mer [5,10) = CGTAC
        assert list(out["pentanuc"]) == ["GTACG", "CGTAC"]
        assert (out["pentanuc"].str[2] == out["ref"]).all()

    def test_uppercases_softmasked(self):
        g = {"1": "acgtacgtacgt"}
        sites = pd.DataFrame({"chrom": ["1"], "pos": [5], "ref": ["A"]})
        out = annotate_pentanucleotide(sites, self._get_seq(g))
        assert list(out["pentanuc"]) == ["GTACG"]

    def test_drops_non_acgt_flank(self):
        # "2": A C G N T A C G T A C G  — N at 0-based idx 3.
        g = {"2": "ACGNTACGTACG"}
        sites = pd.DataFrame({"chrom": ["2", "2"], "pos": [6, 9], "ref": ["A", "T"]})
        # pos 6 -> 5-mer [3,8) = NTACG (N flank) -> dropped; pos 9 -> CGTAC -> kept.
        out = annotate_pentanucleotide(sites, self._get_seq(g))
        assert list(out["pos"]) == [9]
        assert list(out["pentanuc"]) == ["CGTAC"]

    def test_center_ref_mismatch_asserts(self):
        g = {"1": "ACGTACGTACGT"}  # center of pos 5 is A, not C
        sites = pd.DataFrame({"chrom": ["1"], "pos": [5], "ref": ["C"]})
        with pytest.raises(AssertionError, match="center != ref"):
            annotate_pentanucleotide(sites, self._get_seq(g))

    def test_one_read_spans_distant_sites(self):
        # Two far-apart sites on one chrom: a single [min-3, max+2) read covers both.
        g = {"1": "ACGT" * 10}  # 40 bp; base at idx i is "ACGT"[i % 4]
        sites = pd.DataFrame({"chrom": ["1", "1"], "pos": [5, 35], "ref": ["A", "G"]})
        out = annotate_pentanucleotide(sites, self._get_seq(g))
        assert list(out["pentanuc"]) == ["GTACG", "ACGTA"]

    def test_chromosome_boundary_n_padding(self):
        # A near-the-start site whose 5-mer flank runs off the chromosome must be
        # dropped. Emulate the real Genome, which N-pads out-of-range reads so the
        # returned length is always end-start (exercises the lo < 0 path).
        seq = "ACGTACGT"  # 0-based 0..7

        def padded(chrom, start, end):
            return "".join(
                seq[i] if 0 <= i < len(seq) else "N" for i in range(start, end)
            )

        # pos 1 -> 5-mer 0-based [-2, 3) = "NNACG" (left N-pad) -> dropped;
        # pos 4 -> 5-mer [1, 6) = "CGTAC" -> kept. (single read lo=-2, hi=6)
        sites = pd.DataFrame({"chrom": ["1", "1"], "pos": [1, 4], "ref": ["A", "T"]})
        out = annotate_pentanucleotide(sites, padded)
        assert list(out["pos"]) == [4]
        assert list(out["pentanuc"]) == ["CGTAC"]


class TestSubsamplePerContext:
    def _sites(self) -> pd.DataFrame:
        # 5x AAAAA, 2x CCCCC, 1x GGGGG.
        return pd.DataFrame(
            {
                "chrom": ["1"] * 7 + ["2"],
                "pos": [10, 20, 30, 40, 50, 60, 70, 5],
                "ref": ["A"] * 5 + ["C"] * 2 + ["G"],
                "pentanuc": ["AAAAA"] * 5 + ["CCCCC"] * 2 + ["GGGGG"],
            }
        )

    def test_caps_and_keeps_small_groups_whole(self):
        out = subsample_per_context(self._sites(), n=3, seed=0)
        counts = out.groupby("pentanuc").size().to_dict()
        assert counts == {"AAAAA": 3, "CCCCC": 2, "GGGGG": 1}  # capped / kept whole
        # untouched groups keep all their original positions
        assert set(out.loc[out["pentanuc"] == "CCCCC", "pos"]) == {60, 70}

    def test_deterministic_same_seed(self):
        a = subsample_per_context(self._sites(), n=3, seed=7)
        b = subsample_per_context(self._sites(), n=3, seed=7)
        pd.testing.assert_frame_equal(a, b)

    def test_independent_of_input_row_order(self):
        ordered = subsample_per_context(self._sites(), n=3, seed=7)
        shuffled = subsample_per_context(
            self._sites().sample(frac=1, random_state=1).reset_index(drop=True),
            n=3,
            seed=7,
        )
        pd.testing.assert_frame_equal(ordered, shuffled)

    def test_preserves_columns_and_is_subset(self):
        src = self._sites()
        out = subsample_per_context(src, n=2, seed=0)
        assert list(out.columns) == list(src.columns)
        merged = out.merge(src, on=list(src.columns), how="left", indicator=True)
        assert (merged["_merge"] == "both").all()  # every kept row came from src

    def test_n_larger_than_all_groups_returns_all(self):
        src = self._sites()
        out = subsample_per_context(src, n=100, seed=0)
        assert len(out) == len(src)
