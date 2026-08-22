"""Contracts for the issue #473 CDS damage-control analysis."""

import pandas as pd
import pytest
from exp473_center_seeded_projection.repair_cds_evals import (
    artifact_uri,
    baseline_model_name,
    center_model_name,
    select_official_endpoint,
)


def test_repair_artifact_names_are_stable_and_separate():
    commit = "a" * 40
    assert baseline_model_name(1_000) == ("exp417-cds-combined-vertebrates-step-1000")
    assert center_model_name(1_000, experiment_commit=commit) == (
        f"exp473-{commit}-cds-center-1-step-1000"
    )
    assert artifact_uri(
        "s3://bucket/root/",
        "metrics",
        baseline_model_name(1_000),
        "mendelian_traits",
    ) == (
        "s3://bucket/root/metrics/"
        "exp417-cds-combined-vertebrates-step-1000/mendelian_traits.parquet"
    )


def test_select_official_complex_endpoint():
    frame = pd.DataFrame(
        {
            "score_type": ["abs_llr_avg", "abs_llr_avg", "minus_llr_avg"],
            "subset": ["missense_variant", "splicing", "missense_variant"],
            "value": [0.2, 0.3, 0.4],
            "split": ["train", "train", "train"],
        }
    )
    selected = select_official_endpoint(frame, "complex_traits")
    assert selected[["subset", "value"]].to_dict("records") == [
        {"subset": "missense_variant", "value": 0.2}
    ]


def test_select_official_sge_endpoints():
    frame = pd.DataFrame(
        {
            "score_type": ["minus_llr_avg"] * 3,
            "subset": ["missense_variant", "splicing", "synonymous_variant"],
            "value": [0.2, 0.3, 0.4],
            "split": ["train"] * 3,
            "metric": ["AUPRC"] * 3,
            "accession": ["_macro_avg_"] * 3,
            "gene": ["_macro_avg_"] * 3,
        }
    )
    selected = select_official_endpoint(frame, "sge")
    assert selected["subset"].tolist() == ["missense_variant", "splicing"]


def test_select_official_endpoint_refuses_non_development_rows():
    frame = pd.DataFrame(
        {
            "score_type": ["abs_llr_avg"],
            "subset": ["missense_variant"],
            "value": [0.2],
            "split": ["test"],
        }
    )
    with pytest.raises(AssertionError):
        select_official_endpoint(frame, "complex_traits")
