"""Tests for the standardized caQTL/dsQTL dataset build (issue #310).

Covers the TSV parser (`load_standardized_qtl`) and the genome-orientation
sign-flip (`orient_variants`) — the #310 correctness fix that keeps the carried
ChromBPNet/Enformer baseline scores aligned with `effect` through a ref/alt swap.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from marin_dna.pipelines.evals.chrombpnet_qtl import (
    STANDARDIZED_QTL,
    load_standardized_qtl,
    orient_variants,
)


def _write_tsv(df: pl.DataFrame, path: Path) -> str:
    df.write_csv(path, separator="\t")
    return str(path)


def _caqtl_raw() -> pl.DataFrame:
    # caQTL: bool obs.label, native hg38, `variantscore` infix. Row 2 is var.isused
    # = False (dropped); the control row (label False) has a null effect.
    return pl.DataFrame(
        {
            "var.isused": [True, True, False],
            "obs.label": [True, False, True],
            "var.chr": ["chr1", "chr2", "chr3"],
            "var.pos_hg38": [100, 200, 300],
            "var.allele1": ["a", "c", "g"],
            "var.allele2": ["t", "g", "a"],
            "obs.beta": [0.5, None, 9.9],
            "pred.chrombpnet.encsr637xsc.variantscore.ips": [1.0, 0.1, 9.0],
            "pred.chrombpnet.encsr637xsc.variantscore.logfc": [0.8, 0.1, 9.0],
            "pred.chrombpnet.encsr000emt.variantscore.ips": [0.9, 0.2, 9.0],
            "pred.chrombpnet.encsr000emt.variantscore.logfc": [0.6, 0.2, 9.0],
            "pred.enformer.encsr000emt.variantscore.local_logfc": [0.7, 0.0, 9.0],
        }
    )


def _dsqtl_raw() -> pl.DataFrame:
    # dsQTL: integer obs.label (1 = significant, -1 = control), hg19, `varscore` infix.
    return pl.DataFrame(
        {
            "var.isused": [1, 1, 0],
            "obs.label": [1, -1, 1],
            "var.chr": ["chr5", "chr6", "chr7"],
            "var.pos_hg19": [1000, 2000, 3000],
            "var.allele1": ["A", "C", "G"],
            "var.allele2": ["T", "G", "A"],
            "obs.estimate": [-0.3, 0.0, 9.9],
            "pred.chrombpnet.encsr637xsc.varscore.ips": [1.1, 0.1, 9.0],
            "pred.chrombpnet.encsr637xsc.varscore.logfc": [-0.9, 0.1, 9.0],
            "pred.enformer.encsr000emt.varscore.local_logfc": [-0.8, 0.0, 9.0],
        }
    )


class TestLoadStandardizedQtl:
    def test_caqtl_parse(self, tmp_path: Path) -> None:
        path = _write_tsv(_caqtl_raw(), tmp_path / "caqtl.tsv")
        out = load_standardized_qtl(path, STANDARDIZED_QTL["caqtl"])
        # var.isused filter drops the third row.
        assert out.height == 2
        assert out["label"].dtype == pl.Boolean
        assert out["label"].to_list() == [True, False]
        # chrom strips "chr"; ref/alt = allele1/allele2 uppercased.
        assert out["chrom"].to_list() == ["1", "2"]
        assert out["ref"].to_list() == ["A", "C"]
        assert out["alt"].to_list() == ["T", "G"]
        assert out["effect"].to_list()[0] == pytest.approx(0.5)
        # Canonical score columns mapped from the `variantscore` infix.
        for col in (
            "chrombpnet_atac_ips",
            "chrombpnet_atac_logfc",
            "chrombpnet_dnase_ips",
            "chrombpnet_dnase_logfc",
            "enformer_dnase_local_logfc",
        ):
            assert col in out.columns
        assert out["chrombpnet_atac_ips"].to_list() == [1.0, 0.1]

    def test_dsqtl_int_label(self, tmp_path: Path) -> None:
        path = _write_tsv(_dsqtl_raw(), tmp_path / "dsqtl.tsv")
        out = load_standardized_qtl(path, STANDARDIZED_QTL["dsqtl"])
        assert out.height == 2
        # -1 control must map to False (not truthy-positive).
        assert out["label"].to_list() == [True, False]
        assert out["effect"].to_list()[0] == pytest.approx(-0.3)
        # `varscore` infix; dsQTL carries no GM12878-DNase ChromBPNet columns here.
        assert out["chrombpnet_atac_logfc"].to_list() == [-0.9, 0.1]
        assert "chrombpnet_dnase_ips" not in out.columns

    def test_missing_required_score_col_raises(self, tmp_path: Path) -> None:
        raw = _caqtl_raw().drop("pred.chrombpnet.encsr637xsc.variantscore.ips")
        path = _write_tsv(raw, tmp_path / "caqtl.tsv")
        with pytest.raises(AssertionError, match="required baseline column"):
            load_standardized_qtl(path, STANDARDIZED_QTL["caqtl"])

    def test_bad_int_label_raises(self, tmp_path: Path) -> None:
        raw = _dsqtl_raw().with_columns(
            pl.Series("obs.label", [1, 2, 1])  # 2 is not in {-1, 0, 1}
        )
        path = _write_tsv(raw, tmp_path / "dsqtl.tsv")
        with pytest.raises(AssertionError, match="unexpected 'obs.label'"):
            load_standardized_qtl(path, STANDARDIZED_QTL["dsqtl"])

    def test_positive_with_nan_effect_raises(self, tmp_path: Path) -> None:
        raw = _caqtl_raw().with_columns(
            pl.Series("obs.beta", [None, None, 9.9], dtype=pl.Float64)
        )
        path = _write_tsv(raw, tmp_path / "caqtl.tsv")
        with pytest.raises(AssertionError, match="positives with NaN effect"):
            load_standardized_qtl(path, STANDARDIZED_QTL["caqtl"])


class _FakeGenome:
    """Callable genome stub: returns a preset reference base per (chrom, 1-based pos).

    `check_ref_alt` calls ``genome(chrom, pos - 1, pos)`` (0-based half-open single
    base), so we key on ``start + 1`` to recover the 1-based position.
    """

    def __init__(self, ref_by_pos: dict[tuple[str, int], str]) -> None:
        self._ref = ref_by_pos

    def __call__(self, chrom: str, start: int, end: int) -> str:
        assert end == start + 1
        return self._ref[(chrom, start + 1)]


class TestOrientVariants:
    def _frame(self) -> pl.DataFrame:
        # Three positives, one signed carried score column. Row layout:
        #  - pos 100: genome ref == ref  -> no swap
        #  - pos 200: genome ref == alt  -> swap (effect + score must flip)
        #  - pos 300: genome ref == neither -> dropped by check_ref_alt
        return pl.DataFrame(
            {
                "chrom": ["1", "1", "1"],
                "pos": [100, 200, 300],
                "ref": ["A", "C", "G"],
                "alt": ["T", "G", "A"],
                "label": [True, True, True],
                "effect": [0.5, 0.4, 0.3],
                "chrombpnet_atac_logfc": [0.8, 0.6, 0.2],
            }
        )

    def test_swap_flips_effect_and_score(self) -> None:
        V = self._frame()
        genome = _FakeGenome({("1", 100): "A", ("1", 200): "G", ("1", 300): "C"})
        out, n_swapped = orient_variants(V, genome)
        # Row at pos 300 (neither allele matches) is dropped; pos 200 swapped.
        assert n_swapped == 1
        assert sorted(out["pos"].to_list()) == [100, 200]
        out = out.sort("pos")
        # No-swap row unchanged.
        r0 = out.filter(pl.col("pos") == 100).to_dicts()[0]
        assert (r0["ref"], r0["alt"]) == ("A", "T")
        assert r0["effect"] == pytest.approx(0.5)
        assert r0["chrombpnet_atac_logfc"] == pytest.approx(0.8)
        # Swap row: ref/alt swapped, effect AND score sign-flipped.
        r1 = out.filter(pl.col("pos") == 200).to_dicts()[0]
        assert (r1["ref"], r1["alt"]) == ("G", "C")
        assert r1["effect"] == pytest.approx(-0.4)
        assert r1["chrombpnet_atac_logfc"] == pytest.approx(-0.6)

    def test_score_effect_alignment_is_swap_invariant(self) -> None:
        """The core guard: sign(score · effect) per row is preserved through
        orientation (both flip together), so the direction Pearson is unchanged."""
        V = self._frame()
        before = {
            row["pos"]: np.sign(row["effect"] * row["chrombpnet_atac_logfc"])
            for row in V.to_dicts()
        }
        genome = _FakeGenome({("1", 100): "A", ("1", 200): "G", ("1", 300): "C"})
        out, _ = orient_variants(V, genome)
        for row in out.to_dicts():
            after = np.sign(row["effect"] * row["chrombpnet_atac_logfc"])
            assert after == before[row["pos"]]
