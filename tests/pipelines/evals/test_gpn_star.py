"""Tests for ``marin_dna.pipelines.evals.gpn_star``."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from marin_dna.pipelines.evals.gpn_star import (
    GPN_STAR_GIST_BASE,
    GPN_STAR_MODELS,
    GPN_STAR_MODEL_INFO,
    GPN_STAR_SCORE_COLUMN,
    predictions_url,
    score_variants_gpn_star,
)


def _hf(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _preds(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_score_variants_happy_path() -> None:
    hf = _hf(
        [
            {"chrom": "1", "pos": 100, "ref": "A", "alt": "T"},
            {"chrom": "1", "pos": 200, "ref": "G", "alt": "C"},
        ]
    )
    preds = _preds(
        [
            # train rows, in HF order
            {
                "chrom": "1",
                "pos": 100,
                "ref": "A",
                "alt": "T",
                "split": "train",
                "llr": -2.0,
                "abs_llr": 2.0,
                "llr_calibrated": -1.5,
                "abs_llr_calibrated": 0.5,
            },
            {
                "chrom": "1",
                "pos": 200,
                "ref": "G",
                "alt": "C",
                "split": "train",
                "llr": 1.0,
                "abs_llr": 1.0,
                "llr_calibrated": 0.7,
                "abs_llr_calibrated": -0.3,
            },
            # one test row to confirm filtering
            {
                "chrom": "2",
                "pos": 99,
                "ref": "A",
                "alt": "G",
                "split": "test",
                "llr": 0.0,
                "abs_llr": 0.0,
                "llr_calibrated": 0.0,
                "abs_llr_calibrated": 0.0,
            },
        ]
    )
    out = score_variants_gpn_star(hf, preds, split="train")
    assert list(out.columns) == [
        "llr",
        "abs_llr",
        "llr_calibrated",
        "abs_llr_calibrated",
        "minus_llr",
        "minus_llr_calibrated",
    ]
    assert out.shape == (2, 6)
    # minus_* derived correctly
    assert out["minus_llr"].tolist() == [2.0, -1.0]
    assert out["minus_llr_calibrated"].tolist() == [1.5, -0.7]
    # abs_* passed through unchanged (incl. the negative abs_llr_calibrated
    # value, which is fine — calibration can push it below zero).
    assert out["abs_llr_calibrated"].tolist() == [0.5, -0.3]


def test_score_variants_row_count_mismatch() -> None:
    hf = _hf([{"chrom": "1", "pos": 100, "ref": "A", "alt": "T"}])
    preds = _preds(
        [
            {
                "chrom": "1",
                "pos": 100,
                "ref": "A",
                "alt": "T",
                "split": "train",
                "llr": 0.0,
                "abs_llr": 0.0,
                "llr_calibrated": 0.0,
                "abs_llr_calibrated": 0.0,
            },
            {
                "chrom": "1",
                "pos": 200,
                "ref": "G",
                "alt": "C",
                "split": "train",
                "llr": 0.0,
                "abs_llr": 0.0,
                "llr_calibrated": 0.0,
                "abs_llr_calibrated": 0.0,
            },
        ]
    )
    with pytest.raises(AssertionError, match="row count mismatch"):
        score_variants_gpn_star(hf, preds, split="train")


def test_score_variants_row_alignment_mismatch() -> None:
    hf = _hf(
        [
            {"chrom": "1", "pos": 100, "ref": "A", "alt": "T"},
            {"chrom": "1", "pos": 200, "ref": "G", "alt": "C"},
        ]
    )
    preds = _preds(
        [
            {
                "chrom": "1",
                "pos": 100,
                "ref": "A",
                "alt": "T",
                "split": "train",
                "llr": 0.0,
                "abs_llr": 0.0,
                "llr_calibrated": 0.0,
                "abs_llr_calibrated": 0.0,
            },
            # pos 300 instead of 200 → alignment breaks at index 1
            {
                "chrom": "1",
                "pos": 300,
                "ref": "G",
                "alt": "C",
                "split": "train",
                "llr": 0.0,
                "abs_llr": 0.0,
                "llr_calibrated": 0.0,
                "abs_llr_calibrated": 0.0,
            },
        ]
    )
    with pytest.raises(AssertionError, match="row alignment broken at index 1"):
        score_variants_gpn_star(hf, preds, split="train")


def test_score_variants_nan_score() -> None:
    hf = _hf([{"chrom": "1", "pos": 100, "ref": "A", "alt": "T"}])
    preds = _preds(
        [
            {
                "chrom": "1",
                "pos": 100,
                "ref": "A",
                "alt": "T",
                "split": "train",
                "llr": math.nan,
                "abs_llr": 1.0,
                "llr_calibrated": 0.0,
                "abs_llr_calibrated": 0.0,
            },
        ]
    )
    with pytest.raises(AssertionError, match="NaN"):
        score_variants_gpn_star(hf, preds, split="train")


def test_score_variants_chrom_dtype_string_vs_object() -> None:
    """HF chrom may be pandas StringDtype while predictions has object/str;
    both sides get cast to str inside, so the alignment assert should pass."""
    hf = pd.DataFrame(
        {
            "chrom": pd.array(["1"], dtype="string"),
            "pos": [100],
            "ref": ["A"],
            "alt": ["T"],
        }
    )
    preds = pd.DataFrame(
        {
            "chrom": ["1"],
            "pos": [100],
            "ref": ["A"],
            "alt": ["T"],
            "split": ["train"],
            "llr": [-1.0],
            "abs_llr": [1.0],
            "llr_calibrated": [-0.5],
            "abs_llr_calibrated": [0.5],
        }
    )
    out = score_variants_gpn_star(hf, preds, split="train")
    assert len(out) == 1
    assert out["minus_llr"].iloc[0] == 1.0
    assert out["minus_llr_calibrated"].iloc[0] == 0.5


def test_score_variants_split_filter_test() -> None:
    """``split`` arg controls which subset of predictions is used."""
    hf = _hf([{"chrom": "X", "pos": 5, "ref": "T", "alt": "G"}])
    preds = _preds(
        [
            {
                "chrom": "1",
                "pos": 1,
                "ref": "A",
                "alt": "C",
                "split": "train",
                "llr": 0.0,
                "abs_llr": 0.0,
                "llr_calibrated": 0.0,
                "abs_llr_calibrated": 0.0,
            },
            {
                "chrom": "X",
                "pos": 5,
                "ref": "T",
                "alt": "G",
                "split": "test",
                "llr": -3.0,
                "abs_llr": 3.0,
                "llr_calibrated": -2.5,
                "abs_llr_calibrated": 2.5,
            },
        ]
    )
    out = score_variants_gpn_star(hf, preds, split="test")
    assert out["minus_llr"].iloc[0] == 3.0


def test_score_variants_missing_hf_column() -> None:
    hf = _hf([{"chrom": "1", "pos": 100, "ref": "A"}])  # no alt
    preds = _preds(
        [
            {
                "chrom": "1",
                "pos": 100,
                "ref": "A",
                "alt": "T",
                "split": "train",
                "llr": 0.0,
                "abs_llr": 0.0,
                "llr_calibrated": 0.0,
                "abs_llr_calibrated": 0.0,
            }
        ]
    )
    with pytest.raises(AssertionError, match="hf_df missing column 'alt'"):
        score_variants_gpn_star(hf, preds, split="train")


def test_score_variants_missing_pred_column() -> None:
    hf = _hf([{"chrom": "1", "pos": 100, "ref": "A", "alt": "T"}])
    preds = _preds(
        [
            {
                "chrom": "1",
                "pos": 100,
                "ref": "A",
                "alt": "T",
                "split": "train",
                "llr": 0.0,
                "abs_llr": 0.0,
                "llr_calibrated": 0.0,
            }
        ]
    )  # no abs_llr_calibrated
    with pytest.raises(
        AssertionError, match="predictions missing column 'abs_llr_calibrated'"
    ):
        score_variants_gpn_star(hf, preds, split="train")


def test_predictions_url_format() -> None:
    # Current-revision upload uses an underscore before ``GPN-Star`` (the earlier
    # 1:1-revision upload used a dot). Datasets are mendelian + complex (eQTL
    # retired, #172/#194).
    assert predictions_url("mendelian_traits", "V") == (
        f"{GPN_STAR_GIST_BASE}/bolinas_mendelian_traits_GPN-Star-V.parquet"
    )
    assert predictions_url("complex_traits", "P") == (
        f"{GPN_STAR_GIST_BASE}/bolinas_complex_traits_GPN-Star-P.parquet"
    )


def test_predictions_url_unknown_model() -> None:
    with pytest.raises(AssertionError, match="unknown GPN-Star model"):
        predictions_url("mendelian_traits", "X")


def test_score_column_per_dataset_convention() -> None:
    # Mendelian uses signed (pathogenic ⇒ alt depleted under purifying selection).
    assert GPN_STAR_SCORE_COLUMN["mendelian_traits"] == "minus_llr_calibrated"
    # Complex uses magnitude (direction-agnostic).
    assert GPN_STAR_SCORE_COLUMN["complex_traits"] == "abs_llr_calibrated"
    # No surprise datasets (eQTL #172 retired in #194).
    assert set(GPN_STAR_SCORE_COLUMN) == {"mendelian_traits", "complex_traits"}


def test_model_info_keys_match_models_tuple() -> None:
    assert set(GPN_STAR_MODEL_INFO) == set(GPN_STAR_MODELS)
