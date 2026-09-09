"""Exercise the complete additive DAG with tiny, synthetic Kent inputs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from marin_dna_vertebrate_projection.rag.documents import REGIONS
from marin_dna_vertebrate_projection.rag.tables import read_rows, sha256_file

PROJECT = Path(__file__).parents[1]


def test_rag_combined_workflow(tmp_path: Path) -> None:
    if any(
        not shutil.which(name)
        for name in ("liftOver", "faToTwoBit", "twoBitInfo", "twoBitToFa")
    ):
        pytest.skip("Kent tools required for the RAG integration test")
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
    anchors = workdir / "rag-fixture.tsv"
    anchors.write_text(
        "id\tchrom\tstart\tend\tregion\nplus\tchr1\t100\t355\tall\nminus\tchr1\t300\t555\tall\nunmapped\tchr1\t600\t855\tall\n"
    )
    spec = {
        "uri": str(anchors),
        "sha256": sha256_file(anchors),
        "columns": {
            "id": "id",
            "chrom": "chrom",
            "start": "start",
            "end": "end",
            "region": "region",
        },
        "region_value": "all",
        "chrom_names": "ucsc",
        "expected_rows": 3,
    }
    benchmarks = {}
    for name in ("mendelian_traits", "complex_traits", "sge"):
        source = tmp_path / name / ("b" * 40) / "train.parquet"
        source.parent.mkdir(parents=True)
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {"chrom": "1", "pos": pos, "ref": "T", "alt": "A", "match_group": 1}
                    for pos in (228, 428, 728)
                ]
            ),
            source,
        )
        benchmarks[name] = {
            "name": name,
            "revision": "b" * 40,
            "expected_rows": 3,
            "split": "train",
            "url": source.as_uri(),
            "sha256": sha256_file(source),
            "chrom_names": "ensembl",
        }
    overlay = workdir / "rag-overlay.yaml"
    overlay.write_text(
        yaml.safe_dump(
            {
                "rag": {
                    "catalogs": {name: spec for name in REGIONS},
                    "benchmarks": benchmarks,
                    "validation_rows": 400,
                    "seed": 42,
                }
            }
        )
    )
    env = {
        **os.environ,
        "PIPELINE_COMMIT_SHA": "a" * 40,
        "UV_PROJECT_ENVIRONMENT": str(Path(sys.executable).parent.parent),
        "UV_NO_SYNC": "1",
    }
    common = [
        str(Path(sys.executable).with_name("snakemake")),
        "rag_all_documents",
        "--workflow-profile",
        "none",
        "--default-storage-provider",
        "none",
        "--cores",
        "1",
        "--configfile",
        str(overlay),
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

    plan = run("--dry-run")
    assert "rag_union_catalog" in plan and "rag_chain_liftover" in plan
    assert "localrule chain_liftover:" not in plan and "dataset_splits" not in plan
    run()
    audit_path = next((workdir / "results").rglob("rag/anchors/request_audit.json"))
    root = audit_path.parent.parent
    audit = json.loads(audit_path.read_text())
    assert audit["unique_requests"] == 3
    assert audit["source_memberships"] == 24
    for region in REGIONS:
        training = list(read_rows(root / f"datasets/{region}/train.parquet"))
        assert len(training) == 6
        assert {row["orientation"] for row in training} == {"forward", "rc"}
        assert all(row["source_chrom"] != "chr18" for row in training)
        assert not list(read_rows(root / f"datasets/{region}/validation.parquet"))
    harness = list(read_rows(root / "evaluation/combined_development.parquet"))
    assert len(harness) == 9
    assert all(row["species_order"][-1] == "hg38" for row in harness)
    assert len({row["source_row_id"] for row in harness}) == 9
    assert len({row["group_id"] for row in harness}) == 3
