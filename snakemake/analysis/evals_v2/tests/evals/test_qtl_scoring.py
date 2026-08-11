"""Tests for the Accessibility-QTL dashboard assembly (#311/#312).

Covers the model-agnostic formatter half (macro-average + tidy rows). The benchmark
scorer was removed with the ChromBPNet island in #332; these exercise only the
frozen-metrics → dashboard-rows reshaping.
"""

from __future__ import annotations

import math

import polars as pl
import pytest
from marin_dna_evals.qtl_scoring import (
    QTL_BENCHMARK_MODELS,
    assemble_benchmark_rows,
    macro_average_metrics,
)


def _bench_metrics() -> pl.DataFrame:
    """Two models × {caqtl, dsqtl}, ``train`` split, with hand-chosen SEs so the
    combined-SE arithmetic lands on exact values (0.03²+0.04² = 0.05²)."""
    rows = [
        # model, dataset, auPRC, auPRC_se, pearson, pearson_se, n_rows, n_pos
        ("alphagenome", "caqtl", 0.6, 0.03, 0.8, 0.04, 100, 10),
        ("alphagenome", "dsqtl", 0.4, 0.04, 0.6, 0.03, 200, 20),
        ("chrombpnet", "caqtl", 0.5, 0.05, 0.7, 0.05, 100, 10),
        ("chrombpnet", "dsqtl", 0.3, 0.05, 0.5, 0.05, 200, 20),
    ]
    return pl.DataFrame(
        {
            "model": [r[0] for r in rows],
            "dataset": [r[1] for r in rows],
            "split": ["train"] * len(rows),
            "causality_auPRC": [r[2] for r in rows],
            "causality_se": [r[3] for r in rows],
            "direction_pearson": [r[4] for r in rows],
            "direction_pearson_se": [r[5] for r in rows],
            "n_rows": [r[6] for r in rows],
            "n_pos": [r[7] for r in rows],
        }
    )


def test_macro_average_metrics_mean_and_combined_se():
    macro = macro_average_metrics(_bench_metrics())
    assert set(macro["dataset"]) == {"macro"}
    by = {r["model"]: r for r in macro.to_dicts()}
    ag = by["alphagenome"]
    assert ag["causality_auPRC"] == pytest.approx(0.5)  # mean(0.6, 0.4)
    assert ag["causality_se"] == pytest.approx(0.025)  # sqrt(.03²+.04²)/2 = .05/2
    assert ag["direction_pearson"] == pytest.approx(0.7)  # mean(0.8, 0.6)
    assert ag["direction_pearson_se"] == pytest.approx(0.025)
    assert ag["n_rows"] == 300 and ag["n_pos"] == 30  # pooled counts
    cb = by["chrombpnet"]
    assert cb["causality_se"] == pytest.approx(math.sqrt(0.05**2 + 0.05**2) / 2)


def test_macro_average_metrics_incomplete_group_fails_loud():
    # Drop one assay for one model → that (model, split) group is incomplete.
    partial = _bench_metrics().filter(
        ~((pl.col("model") == "chrombpnet") & (pl.col("dataset") == "dsqtl"))
    )
    with pytest.raises(AssertionError, match="macro-average needs all"):
        macro_average_metrics(partial)


def test_assemble_benchmark_rows_scopes_and_registry():
    out = assemble_benchmark_rows(_bench_metrics())
    assert out.columns == [
        "model",
        "display",
        "group",
        "scope",
        "split",
        "causality_auPRC",
        "causality_se",
        "direction_pearson",
        "direction_pearson_se",
        "n_rows",
        "n_pos",
    ]
    assert set(out["scope"]) == {"caqtl", "dsqtl", "macro"}
    assert out.height == 2 * 3  # 2 models × {caqtl, dsqtl, macro}
    ag_macro = out.filter(
        (pl.col("model") == "alphagenome") & (pl.col("scope") == "macro")
    ).row(0, named=True)
    assert ag_macro["display"] == "AlphaGenome" and ag_macro["group"] == "supervised"
    assert ag_macro["causality_auPRC"] == pytest.approx(0.5)


def test_assemble_benchmark_rows_unknown_model_passthrough():
    """A future fine-tuned gLM dropped on S3 with no registry entry still appears
    (opaque key + ``other`` group) — the #311/#312 plug-in acceptance criterion."""
    base = _bench_metrics()
    extra = base.filter(pl.col("model") == "alphagenome").with_columns(
        pl.lit("exp999_glm").alias("model")
    )
    out = assemble_benchmark_rows(pl.concat([base, extra]))
    glm = out.filter(
        (pl.col("model") == "exp999_glm") & (pl.col("scope") == "caqtl")
    ).row(0, named=True)
    assert glm["display"] == "exp999_glm"  # falls back to the raw key
    assert glm["group"] == "other"
    assert "exp999_glm" not in QTL_BENCHMARK_MODELS
