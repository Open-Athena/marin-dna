"""Tests for the AlphaGenome scoring helpers.

The ``alphagenome`` package is an optional dep — the parse-response tests run
without it; the scorer-construction test is gated on the import.
"""

import pandas as pd
import polars as pl
import pytest

from marin_dna.pipelines.evals.alphagenome import (
    ALPHAGENOME_TRACKS,
    DNASE_LFC_MASK_WIDTH,
    GM12878_ONTOLOGY_CURIE,
    SCORE_VARIANT_MAX_ATTEMPTS,
    SCORE_VARIANT_RETRY_STATUS,
    SEQUENCE_LENGTH,
    parse_score_response,
    score_dnase_lfc_resumable,
    select_gm12878_dnase_lfc,
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


# --- GM12878-DNase LFC scorer (caQTL/dsQTL, #262/#311) --------------------------


def _dnase_tidy() -> pd.DataFrame:
    """Mock tidy_scores from the DNase scorer: many DNASE tracks (one per cell type)
    plus a stray non-DNASE row; GM12878 is ontology EFO:0002784."""
    return pd.DataFrame(
        {
            "output_type": ["DNASE", "DNASE", "DNASE", "ATAC"],
            "ontology_curie": [
                "CL:0000047",
                GM12878_ONTOLOGY_CURIE,
                "CL:0000084",
                GM12878_ONTOLOGY_CURIE,
            ],
            "biosample_name": ["neuronal stem cell", "GM12878", "T-cell", "GM12878"],
            "raw_score": [-0.0164, -0.0059, 0.1333, 9.9],
        }
    )


def test_gm12878_constants():
    assert GM12878_ONTOLOGY_CURIE == "EFO:0002784"
    assert DNASE_LFC_MASK_WIDTH == 501


def test_select_gm12878_dnase_lfc_picks_the_one_track():
    # exactly the DNASE + EFO:0002784 row (not the ATAC GM12878 row, not other cells)
    assert select_gm12878_dnase_lfc(_dnase_tidy()) == pytest.approx(-0.0059)


def test_select_gm12878_dnase_lfc_means_replicates():
    tidy = pd.DataFrame(
        {
            "output_type": ["DNASE", "DNASE", "DNASE"],
            "ontology_curie": [
                GM12878_ONTOLOGY_CURIE,
                GM12878_ONTOLOGY_CURIE,
                "CL:0000084",
            ],
            "raw_score": [0.2, 0.4, 9.9],
        }
    )
    assert select_gm12878_dnase_lfc(tidy) == pytest.approx(0.3)  # mean of the two


def test_select_gm12878_dnase_lfc_no_match_fails_loud():
    tidy = pd.DataFrame(
        {"output_type": ["DNASE"], "ontology_curie": ["CL:0000047"], "raw_score": [0.1]}
    )
    with pytest.raises(AssertionError, match="no GM12878 DNase track"):
        select_gm12878_dnase_lfc(tidy)


def test_select_gm12878_dnase_lfc_missing_columns_fails_loud():
    with pytest.raises(AssertionError, match="tidy_scores missing"):
        select_gm12878_dnase_lfc(pd.DataFrame({"output_type": ["DNASE"]}))


def test_make_dnase_lfc_scorer_is_diff_log2_sum():
    pytest.importorskip("alphagenome")
    from alphagenome.models import dna_client, variant_scorers

    from marin_dna.pipelines.evals.alphagenome import make_dnase_lfc_scorer

    scorer = make_dnase_lfc_scorer()
    assert scorer.requested_output == dna_client.OutputType.DNASE
    assert scorer.width == DNASE_LFC_MASK_WIDTH
    assert scorer.aggregation_type == variant_scorers.AggregationType.DIFF_LOG2_SUM


# --- Resumable DNase-LFC scoring (no API; injected score_fn) ---------------------


def _variants(n: int = 4) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "chrom": ["1", "1", "2", "2"][:n],
            "pos": [10, 20, 30, 40][:n],
            "ref": ["A", "C", "G", "T"][:n],
            "alt": ["G", "T", "A", "C"][:n],
        }
    )


def _counting_score_fn(counter: dict):
    """Deterministic fake scorer: lfc = pos * 0.1; records #variants it was asked."""

    def fn(V, **kwargs):
        counter["calls"] += 1
        counter["rows"] += len(V)
        return V["pos"].to_numpy().astype(float) * 0.1

    return fn


def test_score_dnase_lfc_resumable_scores_and_caches(tmp_path):
    ckpt = str(tmp_path / "ckpt.parquet")
    counter = {"calls": 0, "rows": 0}
    out = score_dnase_lfc_resumable(
        _variants(), ckpt, chunk_size=2, score_fn=_counting_score_fn(counter)
    )
    assert counter["rows"] == 4
    assert out.columns == ["chrom", "pos", "ref", "alt", "alphagenome_dnase_lfc"]
    # Rows must stay aligned to `variants` order (the scorer's maintain_order joins). Scores
    # are pos-derived, so a non-order-preserving join surfaces here as a permuted list — this
    # was the symptom of the row-order bug that only appeared under the full-suite run.
    assert out.select("chrom", "pos", "ref", "alt").rows() == _variants().rows()
    assert out["alphagenome_dnase_lfc"].to_list() == [1.0, 2.0, 3.0, 4.0]

    # Resume: the checkpoint now has all 4 → zero new API calls, same scores.
    counter2 = {"calls": 0, "rows": 0}
    out2 = score_dnase_lfc_resumable(
        _variants(), ckpt, chunk_size=2, score_fn=_counting_score_fn(counter2)
    )
    assert counter2["rows"] == 0
    assert out2["alphagenome_dnase_lfc"].to_list() == [1.0, 2.0, 3.0, 4.0]


def test_score_dnase_lfc_resumable_partial_resume(tmp_path):
    ckpt = tmp_path / "ckpt.parquet"
    # Pre-seed 2 of the 4 variants with sentinel scores that must survive untouched.
    pl.DataFrame(
        {
            "chrom": ["1", "1"],
            "pos": [10, 20],
            "ref": ["A", "C"],
            "alt": ["G", "T"],
            "alphagenome_dnase_lfc": [99.0, 98.0],
        }
    ).write_parquet(ckpt)
    counter = {"calls": 0, "rows": 0}
    out = score_dnase_lfc_resumable(
        _variants(), str(ckpt), score_fn=_counting_score_fn(counter)
    )
    assert counter["rows"] == 2  # only the 2 missing variants scored
    by_pos = dict(zip(out["pos"].to_list(), out["alphagenome_dnase_lfc"].to_list()))
    assert by_pos == {10: 99.0, 20: 98.0, 30: 3.0, 40: 4.0}


def test_score_dnase_lfc_resumable_cap_fails_loud(tmp_path):
    counter = {"calls": 0, "rows": 0}
    with pytest.raises(RuntimeError, match="max_new_calls"):
        score_dnase_lfc_resumable(
            _variants(),
            str(tmp_path / "ckpt.parquet"),
            max_new_calls=0,
            score_fn=_counting_score_fn(counter),
        )
    assert counter["rows"] == 0  # cap tripped before any scoring
