"""Tests for the AlphaGenome scoring helpers.

The ``alphagenome`` package is an optional dep — the parse-response tests run
without it; the scorer-construction test is gated on the import.
"""

import pandas as pd
import pytest

from marin_dna.pipelines.evals.alphagenome import (
    ALPHAGENOME_TRACKS,
    SCORE_VARIANT_MAX_ATTEMPTS,
    SCORE_VARIANT_RETRY_STATUS,
    SEQUENCE_LENGTH,
    parse_score_response,
)


def test_alphagenome_constants():
    assert ALPHAGENOME_TRACKS == (
        "ATAC",
        "DNASE",
        "CHIP_TF",
        "CHIP_HISTONE",
        "CAGE",
        "PROCAP",
        "RNA_SEQ",
    )
    assert SEQUENCE_LENGTH == "1MB"


def test_score_variant_retry_includes_internal():
    """Regression guard: INTERNAL must stay in the per-variant retry set —
    it's the code AlphaGenome's "bad machine" outages raise, and the SDK's
    default retry set excludes it (dropping it would re-break large runs)."""
    assert "INTERNAL" in SCORE_VARIANT_RETRY_STATUS
    # The SDK already covers these two; we widen, not narrow.
    assert {"UNAVAILABLE", "RESOURCE_EXHAUSTED"} <= set(SCORE_VARIANT_RETRY_STATUS)
    assert SCORE_VARIANT_MAX_ATTEMPTS >= 2


def test_parse_score_response_single_track_per_assay():
    scorer_repr_to_assay = {"scorer_atac": "ATAC", "scorer_cage": "CAGE"}
    tidy = pd.DataFrame(
        {
            "variant_scorer": ["scorer_atac", "scorer_cage"],
            "raw_score": [1.5, 2.5],
        }
    )
    out = parse_score_response(tidy, scorer_repr_to_assay)
    assert out.shape == (1, 2)
    assert set(out.columns) == {"ATAC_0", "CAGE_0"}
    assert out.loc[0, "ATAC_0"] == 1.5
    assert out.loc[0, "CAGE_0"] == 2.5


def test_parse_score_response_multiple_tracks_per_assay():
    scorer_repr_to_assay = {"scorer_atac": "ATAC", "scorer_cage": "CAGE"}
    tidy = pd.DataFrame(
        {
            "variant_scorer": [
                "scorer_atac",
                "scorer_atac",
                "scorer_atac",
                "scorer_cage",
                "scorer_cage",
            ],
            "raw_score": [0.1, 0.2, 0.3, 1.0, 2.0],
        }
    )
    out = parse_score_response(tidy, scorer_repr_to_assay)
    assert out.shape == (1, 5)
    assert list(out.columns) == ["ATAC_0", "ATAC_1", "ATAC_2", "CAGE_0", "CAGE_1"]
    assert out.loc[0, "ATAC_0"] == 0.1
    assert out.loc[0, "ATAC_2"] == 0.3
    assert out.loc[0, "CAGE_1"] == 2.0


def test_parse_score_response_unknown_scorer_fails_loud():
    scorer_repr_to_assay = {"scorer_atac": "ATAC"}
    tidy = pd.DataFrame(
        {
            "variant_scorer": ["scorer_atac", "scorer_unknown"],
            "raw_score": [1.0, 2.0],
        }
    )
    with pytest.raises(AssertionError, match="not in scorer_repr_to_assay"):
        parse_score_response(tidy, scorer_repr_to_assay)


def test_parse_score_response_missing_columns():
    scorer_repr_to_assay = {"scorer_atac": "ATAC"}
    tidy = pd.DataFrame({"variant_scorer": ["scorer_atac"], "score": [1.0]})
    with pytest.raises(AssertionError, match="unexpected tidy_scores columns"):
        parse_score_response(tidy, scorer_repr_to_assay)


def test_parse_score_response_no_nan_in_normal_path():
    scorer_repr_to_assay = {f"s_{t}": t for t in ALPHAGENOME_TRACKS}
    tidy = pd.DataFrame(
        {
            "variant_scorer": [f"s_{t}" for t in ALPHAGENOME_TRACKS],
            "raw_score": list(range(len(ALPHAGENOME_TRACKS))),
        }
    )
    out = parse_score_response(tidy, scorer_repr_to_assay)
    assert not out.isna().any().any()
    assert len(out.columns) == len(ALPHAGENOME_TRACKS)


def test_make_scorers_uses_l2_diff_log1p():
    pytest.importorskip("alphagenome")
    from alphagenome.models import variant_scorers

    from marin_dna.pipelines.evals.alphagenome import make_scorers

    scorers, repr_to_assay = make_scorers()
    assert len(scorers) == len(ALPHAGENOME_TRACKS)
    assert set(repr_to_assay.values()) == set(ALPHAGENOME_TRACKS)
    for s in scorers:
        assert s.aggregation_type == variant_scorers.AggregationType.L2_DIFF_LOG1P
