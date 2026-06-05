"""Tests for the standardized caQTL/dsQTL baseline eval (no Synapse / no network).

Synthetic TSVs reproduce the two schema conventions the real files use: caQTL has
a boolean ``obs.label`` and a ``variantscore`` infix; dsQTL has an integer
``obs.label`` (``1`` = significant, ``-1`` = control) and a ``varscore`` infix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from marin_dna.pipelines.chrombpnet_eval.standardized_qtl import (
    STANDARDIZED_QTL,
    _positive_label,
    baselines_for,
    evaluate_score_columns,
    load_standardized_qtl,
)


def _write_tsv(path, rows: dict, infix: str) -> str:
    """Write a synthetic standardized TSV with the given infix for score columns."""
    df = pd.DataFrame(rows)
    # rename the canonical score keys to the file's real column names
    rename = {
        "atac_ips": f"pred.chrombpnet.encsr637xsc.{infix}.ips",
        "atac_logfc": f"pred.chrombpnet.encsr637xsc.{infix}.logfc",
        "dnase_ips": f"pred.chrombpnet.encsr000emt.{infix}.ips",
        "dnase_logfc": f"pred.chrombpnet.encsr000emt.{infix}.logfc",
        "enformer": f"pred.enformer.encsr000emt.{infix}.local_logfc",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df.to_csv(path, sep="\t", index=False)
    return str(path)


def test_positive_label_encodings() -> None:
    assert _positive_label(pd.Series([True, False, True])).tolist() == [
        True,
        False,
        True,
    ]
    # dsQTL integer encoding: 1 = positive, -1 = control (must NOT be truthy)
    assert _positive_label(pd.Series([1, -1, 1, -1])).tolist() == [
        True,
        False,
        True,
        False,
    ]
    assert _positive_label(pd.Series([0, 1, 0])).tolist() == [False, True, False]
    assert _positive_label(pd.Series(["True", "false", "1"])).tolist() == [
        True,
        False,
        True,
    ]
    with pytest.raises(AssertionError):
        _positive_label(pd.Series([1, 2, -1]))  # unexpected value 2


def test_load_caqtl_bool_label(tmp_path) -> None:
    spec = STANDARDIZED_QTL["caqtl"]
    p = _write_tsv(
        tmp_path / "caqtl.tsv",
        {
            "var.chr": ["chr1", "chr2", "chr3", "chr4"],
            "var.pos_hg38": [10, 20, 30, 40],
            "var.allele1": ["a", "c", "g", "t"],
            "var.allele2": ["c", "g", "t", "a"],
            "obs.label": [True, False, True, False],
            "var.isused": [True, True, True, False],  # last row dropped
            "obs.beta": [0.5, np.nan, -0.3, np.nan],
            "atac_ips": [1.0, 0.1, -0.8, 0.0],
            "atac_logfc": [0.9, 0.05, -0.7, 0.0],
            "dnase_ips": [0.8, 0.1, -0.6, 0.0],
            "dnase_logfc": [0.7, 0.05, -0.5, 0.0],
            "enformer": [0.6, 0.05, -0.4, 0.0],
        },
        infix="variantscore",
    )
    df = load_standardized_qtl(p, spec)
    assert len(df) == 3  # var.isused filter dropped the 4th
    assert df["label"].tolist() == [True, False, True]
    assert df["chrom"].tolist() == ["1", "2", "3"]  # chr stripped
    assert df["allele1"].tolist() == ["A", "C", "G"]  # upper-cased
    # canonical score columns present
    for c in (
        "chrombpnet_atac_ips",
        "chrombpnet_dnase_logfc",
        "enformer_dnase_local_logfc",
    ):
        assert c in df.columns


def test_load_dsqtl_int_label(tmp_path) -> None:
    """dsQTL's {1,-1} obs.label: the -1 controls must not become positives."""
    spec = STANDARDIZED_QTL["dsqtl"]
    p = _write_tsv(
        tmp_path / "dsqtl.tsv",
        {
            "var.chr": ["chr1", "chr2", "chr3", "chr4", "chr5"],
            "var.pos_hg19": [10, 20, 30, 40, 50],
            "var.allele1": ["A", "C", "G", "T", "A"],
            "var.allele2": ["C", "G", "T", "A", "G"],
            "obs.label": [1, -1, 1, -1, -1],
            "var.isused": [True, True, True, True, True],
            "obs.estimate": [0.5, np.nan, -0.3, np.nan, np.nan],
            "dnase_ips": [1.0, 0.1, -0.8, 0.0, 0.2],
            "dnase_logfc": [0.9, 0.05, -0.7, 0.0, 0.1],
            "enformer": [0.6, 0.05, -0.4, 0.0, 0.1],
        },
        infix="varscore",
    )
    df = load_standardized_qtl(p, spec)
    assert int(df["label"].sum()) == 2  # exactly the two label==1 rows
    assert df["label"].tolist() == [True, False, True, False, False]


def test_load_asserts_positive_with_nan_effect(tmp_path) -> None:
    spec = STANDARDIZED_QTL["caqtl"]
    p = _write_tsv(
        tmp_path / "bad.tsv",
        {
            "var.chr": ["chr1", "chr2"],
            "var.pos_hg38": [10, 20],
            "var.allele1": ["A", "C"],
            "var.allele2": ["C", "G"],
            "obs.label": [True, False],
            "var.isused": [True, True],
            "obs.beta": [np.nan, np.nan],  # positive has no effect → must fail
            "atac_ips": [1.0, 0.1],
            "atac_logfc": [0.9, 0.05],
        },
        infix="variantscore",
    )
    with pytest.raises(AssertionError, match="positives with NaN effect"):
        load_standardized_qtl(p, spec)


def test_baselines_for_subsets_to_present_columns() -> None:
    # only DNase + enformer columns present → ATAC baseline excluded
    df = pd.DataFrame(
        {
            "label": [True, False],
            "effect": [0.1, np.nan],
            "chrombpnet_dnase_ips": [1.0, 0.0],
            "chrombpnet_dnase_logfc": [0.9, 0.0],
            "enformer_dnase_local_logfc": [0.5, 0.0],
        }
    )
    bs = baselines_for(df)
    models = {(b.model, b.assay) for b in bs}
    assert ("ChromBPNet", "GM12878 DNase") in models
    assert ("Enformer", "GM12878 DNase") in models
    assert ("ChromBPNet", "GM12878 ATAC") not in models


def test_evaluate_score_columns_drops_nan_and_reports_coverage() -> None:
    # 3 positives, 3 controls; one control has a NaN score → coverage 5/6.
    df = pd.DataFrame(
        {
            "label": [True, True, True, False, False, False],
            "effect": [0.9, 0.5, -0.7, np.nan, np.nan, np.nan],
            "score": [0.9, 0.5, -0.7, 0.05, -0.02, np.nan],
        }
    )
    out = evaluate_score_columns(df, ["score"], n_bootstrap=5)
    assert set(out["metric"]) == {"AUROC", "AUPRC", "pearson", "spearman"}
    auprc = out[out["metric"] == "AUPRC"].iloc[0]
    assert auprc["coverage"] == pytest.approx(round(5 / 6, 4))  # module rounds to 4dp
    assert auprc["n_rows"] == 5  # NaN-score row dropped
    # signed score perfectly orders effect over positives → pearson ~ 1
    pear = out[out["metric"] == "pearson"].iloc[0]
    assert pear["value"] > 0.9
    assert pear["n_rows"] == 3  # positives only
