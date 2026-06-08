"""Tests for DART-Eval Task 3 cell-type-specific peak parsing + splitting."""

import polars as pl
import pytest

from marin_dna.pipelines.evals import hf_readme
from marin_dna.pipelines.evals.dart_task3 import (
    CELL_TYPES,
    PEAK_WIDTH,
    SPLIT_CHROMS,
    assert_full_dataset,
    parse_dart_task3,
    split_frames,
)


def _row(chrom: str, start: int, label: str) -> dict:
    """One row in the source (top_5000_deseq_peaks.tsv) format."""
    return {"Chr": chrom, "Start": start, "End": start + PEAK_WIDTH, "Cell Type": label}


# --------------------------------------------------------------------------- #
# parse_dart_task3
# --------------------------------------------------------------------------- #
class TestParseDartTask3:
    def _raw(self) -> pl.DataFrame:
        return pl.DataFrame(
            [
                _row("chr6", 5000, "K562"),
                _row("chr1", 1000, "GM12878"),
                _row("chr5", 2000, "H1ESC"),
            ]
        )

    def test_schema_and_strip_chr(self) -> None:
        out = parse_dart_task3(self._raw())
        assert out.columns == ["chrom", "start", "end", "label"]
        # `chr` stripped and sorted by (chrom, start): 1, 5, 6.
        assert out["chrom"].to_list() == ["1", "5", "6"]
        assert out["label"].to_list() == ["GM12878", "H1ESC", "K562"]

    def test_width_invariant(self) -> None:
        out = parse_dart_task3(self._raw())
        assert ((out["end"] - out["start"]) == PEAK_WIDTH).all()

    def test_coordinates_are_zero_based_half_open(self) -> None:
        # start passes through unchanged (0-based half-open in the source).
        out = parse_dart_task3(pl.DataFrame([_row("chr1", 1000, "K562")]))
        assert out["start"].to_list() == [1000]
        assert out["end"].to_list() == [1500]

    def test_missing_column_raises(self) -> None:
        with pytest.raises(AssertionError, match="missing expected columns"):
            parse_dart_task3(self._raw().drop("Start"))

    def test_bad_label_raises(self) -> None:
        bad = pl.DataFrame([_row("chr1", 1000, "HeLa")])
        with pytest.raises(AssertionError, match="unexpected cell-type labels"):
            parse_dart_task3(bad)

    def test_wrong_width_raises(self) -> None:
        bad = self._raw().with_columns(pl.col("End") - 1)
        with pytest.raises(AssertionError, match=f"not every window is {PEAK_WIDTH}"):
            parse_dart_task3(bad)


# --------------------------------------------------------------------------- #
# split_frames
# --------------------------------------------------------------------------- #
class TestSplitFrames:
    def _parsed(self) -> pl.DataFrame:
        # One window per split (chr1->train, chr6->validation, chr5->test) plus
        # an extra train chrom (Y) to check multi-chrom routing.
        return pl.DataFrame(
            {
                "chrom": ["1", "6", "5", "Y"],
                "start": [0, 0, 0, 0],
                "end": [PEAK_WIDTH] * 4,
                "label": ["GM12878", "K562", "H1ESC", "IMR90"],
            }
        )

    def test_partition_is_exact_and_routed(self) -> None:
        frames = split_frames(self._parsed())
        assert set(frames) == {"train", "validation", "test"}
        assert frames["train"]["chrom"].to_list() == ["1", "Y"]
        assert frames["validation"]["chrom"].to_list() == ["6"]
        assert frames["test"]["chrom"].to_list() == ["5"]
        assert sum(f.height for f in frames.values()) == 4

    def test_unknown_chrom_raises(self) -> None:
        bad = self._parsed().with_columns(
            pl.when(pl.col("chrom") == "Y")
            .then(pl.lit("MT"))
            .otherwise(pl.col("chrom"))
            .alias("chrom")
        )
        with pytest.raises(AssertionError, match="outside the canonical"):
            split_frames(bad)

    def test_canonical_split_partitions_all_chroms_disjointly(self) -> None:
        all_chroms = [str(i) for i in range(1, 23)] + ["X", "Y"]
        union = set().union(*SPLIT_CHROMS.values())
        assert union == set(all_chroms)
        # disjoint: lengths sum to 24 with no overlap.
        assert sum(len(v) for v in SPLIT_CHROMS.values()) == len(all_chroms)


# --------------------------------------------------------------------------- #
# assert_full_dataset
# --------------------------------------------------------------------------- #
class TestAssertFullDataset:
    def _frame(self, counts: dict[str, int]) -> pl.DataFrame:
        # Globally unique coordinates (distinct start range per cell type) so the
        # uniqueness check passes for the well-formed fixtures.
        rows = []
        for ci, (ct, n) in enumerate(counts.items()):
            rows.extend(_row("chr1", ci * 10_000_000 + i * 1000, ct) for i in range(n))
        return parse_dart_task3(pl.DataFrame(rows))

    def test_passes_with_all_five(self) -> None:
        df = self._frame({ct: 2 for ct in CELL_TYPES})
        assert_full_dataset(df, min_rows=5, max_rows=20)  # no raise

    def test_missing_celltype_raises(self) -> None:
        df = self._frame({ct: 2 for ct in CELL_TYPES[:4]})
        with pytest.raises(AssertionError, match="expected all 5 cell types"):
            assert_full_dataset(df, min_rows=1, max_rows=20)

    def test_over_cap_raises(self) -> None:
        counts = {ct: 1 for ct in CELL_TYPES}
        counts["K562"] = 5001
        df = self._frame(counts)
        with pytest.raises(AssertionError, match="top-5000 cap"):
            assert_full_dataset(df, min_rows=1, max_rows=100_000)

    def test_imbalanced_raises(self) -> None:
        # All under the cap and unique, but not equal across cell types.
        df = self._frame({"GM12878": 3, "H1ESC": 2, "HEPG2": 2, "IMR90": 2, "K562": 2})
        with pytest.raises(AssertionError, match="not balanced"):
            assert_full_dataset(df, min_rows=1, max_rows=100)

    def test_duplicate_coords_raise(self) -> None:
        # Two cell types sharing the exact same peak coordinate.
        df = parse_dart_task3(
            pl.DataFrame([_row("chr1", 1000, "GM12878"), _row("chr1", 1000, "K562")])
        )
        # add the other three so the cell-type and count checks pass first
        df = pl.concat([df, self._frame({ct: 1 for ct in ["H1ESC", "HEPG2", "IMR90"]})])
        with pytest.raises(AssertionError, match="duplicate peak coordinates"):
            assert_full_dataset(df, min_rows=1, max_rows=100)

    def test_row_count_out_of_range_raises(self) -> None:
        df = self._frame({ct: 2 for ct in CELL_TYPES})
        with pytest.raises(AssertionError, match="outside expected"):
            assert_full_dataset(df, min_rows=50, max_rows=100)


# --------------------------------------------------------------------------- #
# hf_readme.render_dart_task3
# --------------------------------------------------------------------------- #
class TestRenderCard:
    def _write(self, path, chrom: str, label: str, n: int) -> None:
        pl.DataFrame(
            {
                "chrom": [chrom] * n,
                "start": [1000 * i for i in range(n)],
                "end": [1000 * i + PEAK_WIDTH for i in range(n)],
                "label": [label] * n,
            }
        ).write_parquet(path)

    def test_render_smoke(self, tmp_path) -> None:
        train = tmp_path / "train.parquet"
        validation = tmp_path / "validation.parquet"
        test = tmp_path / "test.parquet"
        # 3 train + 2 validation + 1 test = 6 windows, all labeled K562.
        self._write(train, "1", "K562", 3)
        self._write(validation, "6", "K562", 2)
        self._write(test, "5", "K562", 1)

        md = hf_readme.render_dart_task3(
            sha="abc1234def",
            train_path=train,
            validation_path=validation,
            test_path=test,
        )
        assert "# evals_dart_task3" in md
        assert "biology" in md and "genomics" in md and "dna" in md  # frontmatter tags
        # Per-cell-type row: K562 with 3 / 2 / 1 / 6.
        assert "| `K562` | 3 | 2 | 1 | 6 |" in md
        # Split totals + canonical chrom lists.
        assert "| `train.parquet` | 3 |" in md
        assert "| `validation.parquet` | 2 | 6, 21 |" in md
        assert "**total** | **6**" in md
        # Provenance permalink pinned to the sha.
        assert "abc1234" in md
        assert "src/marin_dna/pipelines/evals/dart_task3.py" in md
