"""Tests for ``marin_dna.pipelines.evals.blog_metrics``.

The reader hits S3 for a raw run-name/step id (the ladder/lineage intermediate
checkpoints the blog figures need, which aren't in ``models.yaml``). Tests bypass
the cached S3 reader by monkeypatching ``_read_parquet`` with synthetic frames
keyed by path — mirroring ``test_leaderboard.py`` — so the tidy transform and
path templates are covered without network.
"""

from __future__ import annotations

import polars as pl
import pytest

from marin_dna.pipelines.evals import blog_metrics
from marin_dna.pipelines.evals.metrics import MACRO_AVG_SUBSET


def _patch_read(
    monkeypatch: pytest.MonkeyPatch, responses: dict[str, pl.DataFrame]
) -> None:
    blog_metrics._read_parquet.cache_clear()

    def fake(path: str) -> pl.DataFrame:
        if path not in responses:
            raise FileNotFoundError(path)
        return responses[path]

    monkeypatch.setattr(blog_metrics, "_read_parquet", fake)


def _llr_parquet(
    *, score_type: str = "minus_llr_avg", split: str = "train"
) -> pl.DataFrame:
    """Synthetic evals_v2 metrics parquet: 2 subsets + the macro row."""

    def row(subset: str, value: float, n_groups: int, n_rows: int) -> dict:
        return {
            "score_type": score_type,
            "split": split,
            "subset": subset,
            "value": value,
            "se": 0.02,
            "n_groups": n_groups,
            "n_rows": n_rows,
        }

    return pl.DataFrame(
        [
            row("missense_variant", 0.50, 100, 1000),
            row("splicing", 0.60, 40, 400),
            row(MACRO_AVG_SUBSET, 0.55, 2, 1400),
        ]
    )


def _probe_parquet(*, split: str = "train") -> pl.DataFrame:
    """Synthetic compute_probe_metrics output: probe_score rows + macro + baseline."""

    def row(score_type: str, subset: str, value, n: int, n_pos: int) -> dict:
        return {
            "score_type": score_type,
            "split": split,
            "subset": subset,
            "value": value,
            "se": 0.03,
            "n": n,
            "n_pos": n_pos,
        }

    return pl.DataFrame(
        [
            row("probe_score", "missense_variant", 0.55, 500, 50),
            row("probe_score", "splicing", 0.60, 400, 40),
            row("probe_score", MACRO_AVG_SUBSET, 0.575, 900, 90),
            # baseline rows the probe parquet also carries — must be dropped.
            row("minus_llr_avg", "missense_variant", 0.50, 500, 50),
        ]
    )


# ---- path builders ----------------------------------------------------------


def test_llr_metrics_path():
    assert blog_metrics.llr_metrics_path(
        "scaling-v0.5-h1920-p1B-step-215573", "mendelian_traits"
    ) == (
        "s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics/"
        "scaling-v0.5-h1920-p1B-step-215573/mendelian_traits.parquet"
    )


def test_probe_metrics_path():
    assert blog_metrics.probe_metrics_path("m1", "mendelian_traits") == (
        "s3://oa-bolinas/snakemake/analysis/evals_v2/results/probe_metrics/"
        "m1/mendelian_traits.parquet"
    )


# ---- read_llr_metrics -------------------------------------------------------


def test_read_llr_metrics_tidies_and_macro(monkeypatch: pytest.MonkeyPatch):
    path = blog_metrics.llr_metrics_path("ckpt-x", "mendelian_traits")
    _patch_read(monkeypatch, {path: _llr_parquet()})
    df = blog_metrics.read_llr_metrics("ckpt-x", "mendelian_traits")

    assert df.columns == ["model_id", "subset", "value", "se", "n", "n_positives"]
    assert (df["model_id"] == "ckpt-x").all()
    # per-subset: n = n_rows, n_positives = n_groups
    mis = df.filter(pl.col("subset") == "missense_variant")
    assert mis["n"][0] == 1000 and mis["n_positives"][0] == 100
    # macro: n and n_positives both carry K (= n_groups on the macro row = 2)
    macro = df.filter(pl.col("subset") == MACRO_AVG_SUBSET)
    assert macro["n"][0] == 2 and macro["n_positives"][0] == 2


def test_read_llr_metrics_missing_split_raises(monkeypatch: pytest.MonkeyPatch):
    """Parquet exists but has no rows for the requested (score_type, split)."""
    path = blog_metrics.llr_metrics_path("ckpt-x", "mendelian_traits")
    _patch_read(monkeypatch, {path: _llr_parquet(split="test")})  # train filtered out
    with pytest.raises(LookupError, match="no LLR"):
        blog_metrics.read_llr_metrics("ckpt-x", "mendelian_traits")


def test_read_llr_metrics_complex_uses_abs_score_type(monkeypatch: pytest.MonkeyPatch):
    """score_type is dataset-specific via score_type_for: complex → abs_llr_avg."""
    path = blog_metrics.llr_metrics_path("ckpt-x", "complex_traits")
    _patch_read(monkeypatch, {path: _llr_parquet(score_type="abs_llr_avg")})
    df = blog_metrics.read_llr_metrics("ckpt-x", "complex_traits")
    assert df.height == 3  # would be empty if it filtered on minus_llr_avg


def test_read_llr_metrics_unknown_id_propagates(monkeypatch: pytest.MonkeyPatch):
    """A wholly-missing parquet (no synthetic response) surfaces as FileNotFoundError."""
    _patch_read(monkeypatch, {})
    with pytest.raises(FileNotFoundError):
        blog_metrics.read_llr_metrics("nope", "mendelian_traits")


# ---- read_probe_metrics -----------------------------------------------------


def test_read_probe_metrics_drops_baseline_and_overloads_macro(
    monkeypatch: pytest.MonkeyPatch,
):
    path = blog_metrics.probe_metrics_path("m1", "mendelian_traits")
    _patch_read(monkeypatch, {path: _probe_parquet()})
    df = blog_metrics.read_probe_metrics("m1", "mendelian_traits")

    assert df.columns == ["model_id", "subset", "value", "se", "n", "n_positives"]
    # baseline (minus_llr_avg) row dropped: only probe_score subsets + macro remain
    assert df.height == 3
    macro = df.filter(pl.col("subset") == MACRO_AVG_SUBSET)
    assert macro["n"][0] == 2 and macro["n_positives"][0] == 2  # K
    mis = df.filter(pl.col("subset") == "missense_variant")
    assert mis["n"][0] == 500 and mis["n_positives"][0] == 50


def _probe_parquet_lean(*, split: str = "train") -> pl.DataFrame:
    """The leaner per-subset-only probe schema (no ``se``, no ``_macro_avg_`` row) — what
    the blog ladder/lineage cells emit; the #347 SE + aggregate isn't produced there."""

    def row(score_type: str, subset: str, value, n: int, n_pos: int) -> dict:
        return {
            "score_type": score_type,
            "split": split,
            "subset": subset,
            "value": value,
            "n": n,
            "n_pos": n_pos,
        }

    return pl.DataFrame(
        [
            row("probe_score", "missense_variant", 0.48, 5800, 580),
            row("probe_score", "splicing", 0.55, 3190, 319),
            row("probe_score", "mature_miRNA_variant", None, 40, 4),  # below gate
            row(
                "minus_llr_avg", "missense_variant", 0.42, 5800, 580
            ),  # baseline, dropped
        ]
    )


def test_read_probe_metrics_tolerates_lean_schema(monkeypatch: pytest.MonkeyPatch):
    """Pre-#347 probe parquet (no ``se``, no ``_macro_avg_``): ``se`` fills NaN, per-subset
    ``n`` / ``n_positives`` pass through, baseline dropped, below-gate subset kept as NaN."""
    path = blog_metrics.probe_metrics_path("lean", "mendelian_traits")
    _patch_read(monkeypatch, {path: _probe_parquet_lean()})
    df = blog_metrics.read_probe_metrics("lean", "mendelian_traits")

    assert df.columns == ["model_id", "subset", "value", "se", "n", "n_positives"]
    assert df.height == 3  # 3 probe_score subsets; baseline dropped
    assert df["se"].is_nan().all()  # se absent → NaN
    mis = df.filter(pl.col("subset") == "missense_variant")
    assert mis["n"][0] == 5800 and mis["n_positives"][0] == 580  # passthrough, no macro
    mir = df.filter(pl.col("subset") == "mature_miRNA_variant")
    assert mir["value"][0] is None  # below-gate null preserved
