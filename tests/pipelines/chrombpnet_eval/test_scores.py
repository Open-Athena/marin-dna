"""Tests for ARSENAL score loading + orientation-aware join (non-lift path).

The dsQTL ``lift=True`` path is exercised end-to-end by the M1a driver (it needs
the hg19→hg38 liftover chain); here we test the orientation/sign logic and the
coverage guard with synthetic hg38 variants.
"""

import polars as pl
import pytest

from marin_dna.pipelines.chrombpnet_eval.scores import (
    align_scores_to_variants,
    load_arsenal_scores,
)


def _variants() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "chrom": ["1", "1", "1"],
            "pos": [100, 200, 300],
            "ref": ["C", "A", "T"],  # row 3: our ref/alt swapped vs the study
            "alt": ["T", "G", "C"],
            "label": [True, False, True],
            "effect": [0.5, None, -0.2],
        }
    )


def _scores() -> pl.DataFrame:
    # row1: alt(T)==allele2 -> +logfc; row2: alt(G)==allele1 -> -logfc;
    # row3: study alleles in the opposite order from our ref/alt; alt(C)==allele1 -> -logfc
    return pl.DataFrame(
        {
            "chrom": ["1", "1", "1"],
            "pos": [100, 200, 300],
            "allele1": ["C", "G", "C"],
            "allele2": ["T", "A", "T"],
            "variant_id": ["rs1", "rs2", "rs3"],
            "logfc": [0.3, 0.4, 0.5],
        }
    )


def test_orientation_to_our_alt():
    out = align_scores_to_variants(_variants(), _scores(), lift=False).sort("pos")
    got = dict(zip(out["pos"].to_list(), out["score"].to_list()))
    assert got[100] == pytest.approx(0.3)  # alt == allele2
    assert got[200] == pytest.approx(-0.4)  # alt == allele1
    assert got[300] == pytest.approx(-0.5)  # alt == allele1 (study order flipped)


def test_preserves_variant_columns_and_drops_helpers():
    out = align_scores_to_variants(_variants(), _scores(), lift=False)
    assert {"chrom", "pos", "ref", "alt", "label", "effect", "score"} <= set(
        out.columns
    )
    for helper in ("_pair", "allele2", "logfc", "allele1", "variant_id"):
        assert helper not in out.columns


def test_low_coverage_raises():
    # add a variant with no matching score → coverage 3/4 < 0.95
    variants = _variants().vstack(
        pl.DataFrame(
            {
                "chrom": ["2"],
                "pos": [999],
                "ref": ["A"],
                "alt": ["G"],
                "label": [False],
                "effect": [None],
            }
        )
    )
    with pytest.raises(AssertionError, match="coverage"):
        align_scores_to_variants(variants, _scores(), lift=False)


def test_unmatched_dropped_when_coverage_ok():
    variants = _variants().vstack(
        pl.DataFrame(
            {
                "chrom": ["2"],
                "pos": [999],
                "ref": ["A"],
                "alt": ["G"],
                "label": [False],
                "effect": [None],
            }
        )
    )
    out = align_scores_to_variants(variants, _scores(), lift=False, min_coverage=0.5)
    assert out.height == 3  # the unmatched row 999 is dropped
    assert out.filter(pl.col("score").is_null()).height == 0


def test_load_arsenal_scores(tmp_path):
    p = tmp_path / "variant_scores.tsv"
    p.write_text(
        "chr\tpos\tallele1\tallele2\tvariant_id\tlogfc\tabs_logfc\n"
        "chr1\t100\tc\tt\trs1\t0.3\t0.3\n"
        "chr1\t200\tG\tA\trs2\t-0.4\t0.4\n"
    )
    df = load_arsenal_scores(str(p))
    assert df.columns == ["chrom", "pos", "allele1", "allele2", "variant_id", "logfc"]
    assert df["chrom"].to_list() == ["1", "1"]  # 'chr' stripped
    assert df["allele1"].to_list() == ["C", "G"]  # upper-cased
    assert df["logfc"].to_list() == [pytest.approx(0.3), pytest.approx(-0.4)]
    # flip_logfc negates the score (needed for ARSENAL's released dsQTL scores).
    flipped = load_arsenal_scores(str(p), flip_logfc=True)
    assert flipped["logfc"].to_list() == [pytest.approx(-0.3), pytest.approx(0.4)]
