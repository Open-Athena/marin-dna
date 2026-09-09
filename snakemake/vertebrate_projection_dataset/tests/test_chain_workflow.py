"""End-to-end fixture with real Kent tools; required by the dedicated CI job."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest
import yaml

PROJECT = Path(__file__).parents[1]
TOOLS = ("liftOver", "faToTwoBit", "twoBitInfo", "twoBitToFa", "zstd")


def test_chain_workflow_both_cohorts_through_sequences_and_splits(
    tmp_path: Path,
) -> None:
    missing = [name for name in TOOLS if not shutil.which(name)]
    if missing:
        if os.environ.get("CHAIN_SMOKE_REQUIRED") == "1":
            pytest.fail(f"required Kent tools are missing: {missing}")
        pytest.skip(f"Kent tools not installed: {missing}")
    workdir = tmp_path / "pipeline"
    shutil.copytree(
        PROJECT,
        workdir,
        ignore=shutil.ignore_patterns(
            ".venv",
            ".snakemake",
            "results",
            "__pycache__",
            ".pytest_cache",
            "*.egg-info",
        ),
    )
    # Use the installed project environment without resolving editable paths
    # relative to this isolated copy. The fixture never accesses cloud storage.
    env = {
        **os.environ,
        "PIPELINE_COMMIT_SHA": "a" * 40,
        "UV_PROJECT_ENVIRONMENT": str(Path(sys.executable).parent.parent),
        "UV_NO_SYNC": "1",
    }
    common = [
        str(Path(sys.executable).with_name("snakemake")),
        "--workflow-profile",
        "none",
        "--default-storage-provider",
        "none",
        "--cores",
        "1",
    ]

    def run(*arguments: str) -> str:
        result = subprocess.run(
            common + list(arguments),
            cwd=workdir,
            env=env,
            text=True,
            capture_output=True,
            timeout=180,
            check=False,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        return output

    coordinate_plan = run("all_projections", "--dry-run")
    assert "chain_liftover" in coordinate_plan
    assert "genome_twobit" not in coordinate_plan
    plan = run("all", "--dry-run")
    for retired in ("halLiftover", "hal2fasta", "multiz_candidates", "stage_hal"):
        assert retired not in plan
    assert "chain_sequences" in plan and "dataset_splits" in plan
    run("all")

    bases = list((workdir / "results/chains-v1" / ("a" * 40)).glob("*/smoke"))
    assert len(bases) == 1
    base = bases[0]
    sequences = pl.read_parquet(base / "sequences/all_sources.parquet")
    assert sequences.height == 7  # Three human rows, two accepted rows per target.
    assert (sequences["sequence"].str.len_chars() == 255).all()
    assert sequences.filter(pl.col("alignment_source") == "human_reference").height == 3
    target = "ACGTacgtAA" * 200
    complement = str.maketrans("ACGTacgt", "TGCAtgca")
    for species in ("Papio_anubis", "galGal4"):
        rows = sequences.filter(pl.col("alignment_name") == species)
        assert rows.height == 2
        plus = rows.filter(pl.col("query_name") == "plus").row(0, named=True)
        minus = rows.filter(pl.col("query_name") == "minus").row(0, named=True)
        assert plus["sequence"] == target[400:655]
        assert minus["sequence"] == target[1145:1400].translate(complement)[::-1]
        assert plus["sequence"][127] == target[527]
        assert minus["sequence"][127] == target[1272].translate(complement)
        audit = json.loads((base / f"chains/{species}/audit.json").read_text())
        assert audit["input_queries"] == 3
        assert audit["accepted_queries"] == 2 and audit["unmapped_queries"] == 1
    qc = pl.read_parquet(base / "qc/per_anchor.parquet")
    assert qc.filter(pl.col("query_name") == "unmapped")["no_mapping_count"][0] == 2
    train = pl.read_parquet(base / "datasets/all/train.parquet")
    validation = pl.read_parquet(base / "datasets/all/validation.parquet")
    assert train.height == 12 and validation.height == 1
    assert (
        json.loads((base / "metadata/assets.json").read_text())["projector"]
        == "UCSC liftOver"
    )
    run("all_hf_files")
    publication = json.loads(
        (base / "hf_validation/hf_publication_manifest.json").read_text()
    )
    assert set(publication["cohorts"]) == {"all"}
    assert publication["cohorts"]["all"]["splits"]["train"]["rows"] == 12
    card = (base / "hf/all/README.md").read_text()
    assert "fabricated synthetic test assets" in card
    assert "pinned smoke catalog" in card

    # Non-default settings must reach artifact validation through the resolved
    # config, not be silently replaced by the committed config/config.yaml.
    overlay = workdir / "publication-overlay.yaml"
    overlay.write_text(
        yaml.safe_dump(
            {
                "hf_owner": "fixture-owner",
                "smoke_cohorts": ["cds"],
                "publication_smoke_train_shards": 2,
                "validation_seed": 17,
                "publication_shuffle_seed": 23,
            }
        )
    )
    run("all_hf_files", "--configfile", str(overlay))
    manifests = list(
        (workdir / "results").rglob("hf_validation/hf_publication_manifest.json")
    )
    assert len(manifests) == 2
    alternate = next(path for path in manifests if path.parent.parent != base)
    publication = json.loads(alternate.read_text())
    assert set(publication["cohorts"]) == {"cds"}
    assert publication["cohorts"]["cds"]["splits"]["train"]["rows"] == 4
    alternate_card = (alternate.parent.parent / "hf/cds/README.md").read_text()
    assert "# `fixture-owner/vertebrate-chains-v1-cds`" in alternate_card
    assert "seed 17" in alternate_card
