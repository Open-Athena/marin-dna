"""Tests for the model-agnostic QTL benchmark logic (issue #311).

Synthetic data with *known* causality (perfectly separable ``|causality_score|``) and
direction (``direction_score`` ∝ ``effect`` over positives) pins the metric to 1.0, so
the assertions are exact and the two-column vs single-signal paths are both exercised.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import math

from marin_dna.pipelines.evals.qtl_scoring import (
    AG_TEST_CHROMS,
    MODEL_SCORE_COLUMNS,
    QTL_BENCHMARK_MODELS,
    RANDOM_BASELINE_AUPRC,
    SUPPL_TABLE4_REFERENCE,
    align_score_sign,
    assemble_benchmark_rows,
    compute_qtl_split_metrics,
    macro_average_metrics,
    qtl_split_masks,
    reference_metrics,
    to_score_interface,
)


def _synthetic(single_signal: bool) -> tuple[pl.DataFrame, pl.DataFrame]:
    """20 variants (10 positive / 10 control) with every split populated.

    Positives: chroms 3 (4) + 1 (6), split_source train (5) / test (5); controls the
    same layout. ``|causality_score|`` is large for positives and 0 for controls
    (auPRC → 1.0); ``direction_score`` ∝ ``effect`` over positives (Pearson → 1.0).
    """
    pos_effect = [round(0.1 * (i + 1), 2) for i in range(10)]  # 0.1 .. 1.0
    chrom = (["3"] * 4 + ["1"] * 6) * 2  # positives then controls
    pos = list(range(1, 21))
    label = [True] * 10 + [False] * 10
    effect = pos_effect + [None] * 10
    split_source = (["train"] * 5 + ["test"] * 5) * 2
    dataset = pl.DataFrame(
        {
            "chrom": chrom,
            "pos": pos,
            "ref": ["A"] * 20,
            "alt": ["G"] * 20,
            "label": label,
            "effect": effect,
            "split_source": split_source,
        }
    )
    causality = [e * 10 for e in pos_effect] + [0.0] * 10
    direction = causality if single_signal else (pos_effect + [0.0] * 10)
    scores = pl.DataFrame(
        {
            "chrom": chrom,
            "pos": pos,
            "ref": ["A"] * 20,
            "alt": ["G"] * 20,
            "causality_score": causality,
            "direction_score": direction,
        }
    )
    return scores, dataset


def test_to_score_interface_aliases_and_selects():
    df = pl.DataFrame(
        {
            "chrom": ["1"],
            "pos": [10],
            "ref": ["A"],
            "alt": ["G"],
            "ips": [2.0],
            "logfc": [0.5],
            "other": [9.9],
        }
    )
    out = to_score_interface(df, "ips", "logfc")
    assert out.columns == [
        "chrom",
        "pos",
        "ref",
        "alt",
        "causality_score",
        "direction_score",
    ]
    assert out["causality_score"].to_list() == [2.0]
    assert out["direction_score"].to_list() == [0.5]


def test_to_score_interface_missing_column_fails_loud():
    df = pl.DataFrame({"chrom": ["1"], "pos": [10], "ref": ["A"], "alt": ["G"]})
    with pytest.raises(AssertionError, match="missing column 'ips'"):
        to_score_interface(df, "ips", "ips")


def test_align_score_sign_keeps_and_flips():
    reference = np.array([1.0, 2.0, 3.0, -1.0, -2.0])
    base = np.array([0.5, 1.0, 1.5, -0.5, -1.0])  # agrees with reference
    aligned, sign = align_score_sign(base, reference)
    assert sign == 1.0 and np.allclose(aligned, base)
    # A flipped-convention score is flipped back to agree with the reference.
    aligned, sign = align_score_sign(-base, reference)
    assert sign == -1.0 and np.allclose(aligned, base)


def test_align_score_sign_ambiguous_fails_loud():
    a = np.array([1.0, 1.0, -1.0, -1.0])
    b = np.array([1.0, -1.0, 1.0, -1.0])  # orthogonal → corr 0 → ambiguous
    with pytest.raises(AssertionError, match="ambiguous"):
        align_score_sign(a, b)


def test_qtl_split_masks():
    chrom = np.array(["1", "3", "2", "6"])
    split_source = np.array(["train", "train", "test", "test"])
    masks = qtl_split_masks(chrom, split_source)
    assert masks["all"].tolist() == [True, True, True, True]
    assert masks["train"].tolist() == [True, True, False, False]
    assert masks["test"].tolist() == [False, False, True, True]
    # ag_test is chrom membership (3 and 6 are AG-test chroms; 1 and 2 are not).
    assert masks["ag_test"].tolist() == [False, True, False, True]
    assert {"3", "6"} <= AG_TEST_CHROMS and "1" not in AG_TEST_CHROMS


def test_qtl_split_masks_rejects_unknown_split_source():
    with pytest.raises(AssertionError, match="unexpected split_source"):
        qtl_split_masks(np.array(["1"]), np.array(["holdout"]))


@pytest.mark.parametrize("single_signal", [True, False])
def test_compute_qtl_split_metrics_known_values(single_signal: bool):
    scores, dataset = _synthetic(single_signal)
    out = compute_qtl_split_metrics(
        scores, dataset, dataset="caqtl", model="chrombpnet", n_bootstrap=50
    )
    assert list(out["split"]) == ["all", "train", "test", "ag_test"]
    # Perfectly separable causality and perfectly correlated direction → 1.0 everywhere.
    assert np.allclose(out["causality_auPRC"], 1.0)
    assert np.allclose(out["causality_AUROC"], 1.0)
    assert np.allclose(out["direction_pearson"], 1.0)
    assert np.allclose(out["direction_spearman"], 1.0)
    by_split = {r["split"]: r for r in out.to_dict("records")}
    assert (by_split["all"]["n_rows"], by_split["all"]["n_pos"]) == (20, 10)
    assert (by_split["train"]["n_rows"], by_split["train"]["n_pos"]) == (10, 5)
    assert (by_split["test"]["n_rows"], by_split["test"]["n_pos"]) == (10, 5)
    assert (by_split["ag_test"]["n_rows"], by_split["ag_test"]["n_pos"]) == (8, 4)
    assert (out["dataset"] == "caqtl").all() and (out["coverage"] == 1.0).all()


def test_compute_qtl_split_metrics_is_model_agnostic():
    """Acceptance (#311): an arbitrary, never-before-seen model name flows through the
    shared metric with no code change."""
    scores, dataset = _synthetic(single_signal=True)
    out = compute_qtl_split_metrics(
        scores, dataset, dataset="dsqtl", model="some_future_glm_v7", n_bootstrap=20
    )
    assert (out["model"] == "some_future_glm_v7").all()
    assert np.allclose(out["causality_auPRC"], 1.0)
    assert "some_future_glm_v7" not in MODEL_SCORE_COLUMNS  # truly unknown to the code


def test_compute_qtl_split_metrics_rejects_key_mismatch():
    scores, dataset = _synthetic(single_signal=True)
    # A scored variant absent from the dataset must fail loud (orientation/key bug).
    bad = scores.with_columns(
        pl.when(pl.col("pos") == 1)
        .then(pl.lit(999))
        .otherwise(pl.col("pos"))
        .alias("pos")
    )
    with pytest.raises(AssertionError, match="scored variants matched the dataset"):
        compute_qtl_split_metrics(
            bad, dataset, dataset="caqtl", model="x", splits=("all",), n_bootstrap=10
        )


# --- dashboard assembly (#312): macro-average + tidy rows ---------------------------


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


def test_reference_metrics():
    out = reference_metrics("dsqtl")
    assert (out["split"] == "ag_test").all()
    by_model = {r["model"]: r for r in out.to_dict("records")}
    for model, (auprc, pearson) in SUPPL_TABLE4_REFERENCE["dsqtl"].items():
        assert by_model[model]["causality_auPRC"] == pytest.approx(auprc)
        assert by_model[model]["direction_pearson"] == pytest.approx(pearson)
    assert by_model["Random"]["causality_auPRC"] == pytest.approx(
        RANDOM_BASELINE_AUPRC["dsqtl"]
    )
    assert by_model["Random"]["direction_pearson"] == 0.0
