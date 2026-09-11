"""Explicit model registration and one-job routing for the combined backend."""

import copy
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from marin_dna_evals.ci_dry_run import prepare_overlay
from marin_dna_evals.workflow_config import validate_rag_models

PROJECT = Path(__file__).parents[2]


def test_registered_models_dry_run_offline_without_changing_contracts(tmp_path):
    source = PROJECT / "config/config.yaml"
    original_text = source.read_text()
    original = yaml.safe_load(original_text)
    overlay = prepare_overlay(source, tmp_path)
    rewritten = yaml.safe_load(overlay.read_text())
    assert set(rewritten) == {"models"}
    assert source.read_text() == original_text
    assert len(rewritten["models"]) == len(original["models"])
    rag_models = []
    for before, after in zip(original["models"], rewritten["models"], strict=True):
        restored = copy.deepcopy(after)
        if before.get("inference_backend") == "rag_combined":
            rag_models.append(before)
            stub = Path(after["rag_harness"]["uri"])
            assert stub.is_file() and stub.stat().st_size == 0
            restored["rag_harness"]["uri"] = before["rag_harness"]["uri"]
        assert restored == before
    assert rag_models, "exercise the registered combined RAG path"
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("AWS_")
    }
    environment.update(
        AWS_EC2_METADATA_DISABLED="true",
        AWS_CONFIG_FILE=str(tmp_path / "absent-aws-config"),
        AWS_SHARED_CREDENTIALS_FILE=str(tmp_path / "absent-aws-credentials"),
    )
    result = subprocess.run(
        [
            str(Path(sys.executable).with_name("snakemake")),
            "all",
            "--snakefile",
            str(PROJECT / "workflow/Snakefile"),
            "--directory",
            str(PROJECT),
            "--workflow-profile",
            "none",
            "--default-storage-provider",
            "none",
            "--configfile",
            str(overlay),
            "--cores",
            "1",
            "--dry-run",
        ],
        env=environment,
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert output.count("rule compute_rag_scores:") == len(rag_models)
    for model in rag_models:
        for dataset in model["datasets"]:
            assert f"results/scores/{model['name']}/{dataset}.parquet" in output


def model_config():
    return {
        "name": "synthetic-rag-fixture",
        "gcs_path": "gs://fixture/checkpoint",
        "window_size": 255,
        "document_tokens": 10240,
        "datasets": ["mendelian_traits", "complex_traits", "sge"],
        "inference_backend": "rag_combined",
        "rag_harness": {"uri": "harness.parquet", "sha256": "a" * 64},
    }


def test_rag_registration_is_explicit_and_development_only():
    model = model_config()
    validate_rag_models([model, {"name": "legacy"}], split="train")
    for field, value in (
        ("inference_backend", "typo"),
        ("window_size", 10240),
        ("document_tokens", 255),
        ("datasets", ["mendelian_traits"]),
        ("rag_harness", {"uri": "harness.parquet", "sha256": "bad"}),
    ):
        bad = copy.deepcopy(model)
        bad[field] = value
        with pytest.raises(ValueError):
            validate_rag_models([bad], split="train")
    with pytest.raises(ValueError, match="development"):
        validate_rag_models([model], split="test")


def test_combined_score_target_schedules_one_inference_job(tmp_path):
    model = model_config()
    harness = tmp_path / "harness.parquet"
    harness.touch()
    model["rag_harness"]["uri"] = str(harness)
    overlay = tmp_path / "rag.yaml"
    overlay.write_text(
        yaml.safe_dump(
            {
                "models": [model],
                "nuc_dep": {"models": []},
                "umap_embeddings": {"models": []},
                "ll_gap": {"models": []},
                "probe": {"models": []},
            }
        )
    )
    command = [
        str(Path(sys.executable).with_name("snakemake")),
        "results/scores/synthetic-rag-fixture/mendelian_traits.parquet",
        "--snakefile",
        str(PROJECT / "workflow/Snakefile"),
        "--directory",
        str(PROJECT),
        "--workflow-profile",
        "none",
        "--default-storage-provider",
        "none",
        "--configfile",
        str(overlay),
        "--cores",
        "1",
        "--dry-run",
    ]
    result = subprocess.run(
        command, text=True, capture_output=True, timeout=90, check=False
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "compute_rag_scores" in output and "rule compute_scores:" not in output
    assert "complex_traits.parquet" in output and "sge.parquet" in output


def test_reference_only_subset_config_does_not_resolve_rag_datasets(tmp_path):
    model = {
        "name": "reference-fixture",
        "gcs_path": "gs://fixture/checkpoint",
        "window_size": 255,
        "datasets": ["mendelian_traits"],
    }
    config = yaml.safe_load((PROJECT / "config/config.yaml").read_text())
    config["models"] = [model]
    config["datasets"] = [
        dataset
        for dataset in config["datasets"]
        if dataset["name"] == "mendelian_traits"
    ]
    for name in ("nuc_dep", "umap_embeddings", "ll_gap", "probe"):
        config[name] = {"models": []}
    overlay = tmp_path / "reference-only.yaml"
    overlay.write_text(yaml.safe_dump(config))
    result = subprocess.run(
        [
            str(Path(sys.executable).with_name("snakemake")),
            "results/scores/reference-fixture/mendelian_traits.parquet",
            "--snakefile",
            str(PROJECT / "workflow/Snakefile"),
            "--directory",
            str(PROJECT),
            "--workflow-profile",
            "none",
            "--default-storage-provider",
            "none",
            "--configfile",
            str(overlay),
            "--cores",
            "1",
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "rule compute_scores:" in output and "rule compute_rag_scores:" not in output
