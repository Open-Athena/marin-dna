from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
from marin_dna_vertebrate_projection.provenance import (
    hash_pipeline_config,
    write_producer_manifest,
)

PROJECT_ROOT = Path(__file__).parents[1]
SNAKEMAKE = Path(sys.executable).with_name("snakemake")
PIPELINE_COMMIT = "a" * 40


def _run_snakemake(workdir: Path, *args: str) -> str:
    env = {**os.environ, "PIPELINE_COMMIT_SHA": PIPELINE_COMMIT}
    result = subprocess.run(
        [str(SNAKEMAKE), *args],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    return output


def test_filesystem_storage_restores_durable_input_in_clean_workdir(
    tmp_path: Path,
) -> None:
    storage = tmp_path / "storage"
    snakefile = """\
rule producer:
    input:
        local("config.txt"),
    output:
        "durable.txt",
    shell:
        "cp {input} {output}"


rule consumer:
    input:
        "durable.txt",
    output:
        local("consumed.txt"),
    shell:
        "cp {input} {output}"
"""
    workdirs = [tmp_path / name for name in ["worker-a", "worker-b"]]
    for workdir in workdirs:
        workdir.mkdir()
        (workdir / "Snakefile").write_text(snakefile)
        (workdir / "config.txt").write_text("restored across workers\n")

    common = (
        "--default-storage-provider",
        "fs",
        "--default-storage-prefix",
        str(storage),
        "--cores",
        "1",
    )
    _run_snakemake(workdirs[0], "producer", *common)
    assert (storage / "durable.txt").read_text() == "restored across workers\n"

    _run_snakemake(
        workdirs[1],
        "consumer",
        *common,
        "--allowed-rules",
        "consumer",
    )
    assert (workdirs[1] / "consumed.txt").read_text() == "restored across workers\n"


def test_hf_only_dag_closes_from_clean_storage_backed_workdir(tmp_path: Path) -> None:
    workdir = tmp_path / "pipeline"
    shutil.copytree(
        PROJECT_ROOT,
        workdir,
        ignore=shutil.ignore_patterns(".venv", ".snakemake", "results", "__pycache__"),
    )
    config = yaml.safe_load((workdir / "config/config.yaml").read_text())
    config["tier"] = "full"
    config_sha256 = hash_pipeline_config(config)
    base = (
        Path("results")
        / str(config["pipeline_version"])
        / PIPELINE_COMMIT
        / config_sha256
        / "full"
    )
    storage = tmp_path / "storage"
    producer = storage / base / "metadata/producer.json"
    write_producer_manifest(
        producer,
        pipeline_commit=PIPELINE_COMMIT,
        config_sha256=config_sha256,
        pipeline_version=str(config["pipeline_version"]),
        tier="full",
    )
    (storage / base / "metadata/species_active.tsv").write_text("fixture\n")
    for cohort in config["region_cohorts"]:
        for split in ["train", "validation"]:
            source = storage / base / f"datasets/{cohort}/{split}.parquet"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"fixture")
        for audit_name in [
            "validation_selection.tsv",
            "validation_composition.tsv",
            "split_summary.json",
        ]:
            audit_path = storage / base / f"datasets/{cohort}/{audit_name}"
            audit_path.write_text("fixture\n")

    output = _run_snakemake(
        workdir,
        "all_hf_files",
        "--dry-run",
        "--quiet",
        "all",
        "--profile",
        "workflow/profiles/default",
        "--default-storage-provider",
        "fs",
        "--default-storage-prefix",
        str(storage),
        "--cores",
        "1",
        "--config",
        "tier=full",
        "--allowed-rules",
        "prepare_train_jsonl_shards",
        "prepare_validation_jsonl_shards",
        "compress_publication_shard",
        "dataset_card",
        "hf_artifact_manifest",
        "all_hf_files",
    )
    assert "prepare_train_jsonl_shards" in output
    assert "hf_artifact_manifest" in output
    assert "stage_hal" not in output
    assert "dataset_splits" not in output
