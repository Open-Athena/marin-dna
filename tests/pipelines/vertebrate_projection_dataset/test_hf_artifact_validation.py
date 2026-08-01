from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
import zstandard as zstd

from scripts.issue417_validate_hf_artifacts import validate_artifacts


PIPELINE_COMMIT = "a" * 40


def _frame(source_chrom: str, rows: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "query_name": [f"anchor_{index}" for index in range(rows)],
            "source_chrom": [source_chrom] * rows,
            "sequence": ["ACgt" + "A" * 251] * rows,
            "augmentation": ["+"] * rows,
        }
    )


def _write_zstd_jsonl(frame: pl.DataFrame, path: Path) -> None:
    payload = frame.write_ndjson().encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(zstd.ZstdCompressor().compress(payload))


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    artifact_dir = tmp_path / "hf"
    source_dir = tmp_path / "datasets"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "region_cohorts: [all]",
                "publication_train_shards: 2",
                "publication_validation_shards: 1",
                "validation_chrom: chr18",
                "target_length: 255",
            ]
        )
        + "\n"
    )

    train = _frame("chr1", 5)
    validation = _frame("chr18", 2)
    (source_dir / "all").mkdir(parents=True)
    train.write_parquet(source_dir / "all" / "train.parquet")
    validation.write_parquet(source_dir / "all" / "validation.parquet")

    card = artifact_dir / "all" / "README.md"
    card.parent.mkdir(parents=True)
    card.write_text(
        "\n".join(
            [
                "# `marin-dna/vertebrate-v1-all`",
                f"https://github.com/Open-Athena/marin-dna/blob/{PIPELINE_COMMIT}/README.md",
                "path: data/train/*.jsonl.zst",
                "path: data/validation/*.jsonl.zst",
            ]
        )
        + "\n"
    )
    _write_zstd_jsonl(
        train.slice(0, 3), artifact_dir / "all/data/train/shard_0000.jsonl.zst"
    )
    _write_zstd_jsonl(
        train.slice(3, 2), artifact_dir / "all/data/train/shard_0001.jsonl.zst"
    )
    _write_zstd_jsonl(
        validation, artifact_dir / "all/data/validation/shard_0000.jsonl.zst"
    )
    return artifact_dir, source_dir, config_path


def test_validate_hf_artifacts_checks_rows_schema_and_split_invariants(
    tmp_path: Path,
) -> None:
    artifact_dir, source_dir, config_path = _fixture(tmp_path)
    output_path = tmp_path / "manifest.json"
    manifest = validate_artifacts(
        artifact_dir,
        source_dir,
        output_path,
        config_path=config_path,
        pipeline_commit=PIPELINE_COMMIT,
        workers=1,
    )

    persisted = json.loads(output_path.read_text())
    assert persisted == manifest
    assert persisted["cohorts"]["all"]["splits"]["train"]["rows"] == 5
    assert persisted["cohorts"]["all"]["splits"]["validation"]["rows"] == 2
    assert len(persisted["cohorts"]["all"]["splits"]["train"]["shards"]) == 2


def test_validate_hf_artifacts_rejects_qc_sidecars(tmp_path: Path) -> None:
    artifact_dir, source_dir, config_path = _fixture(tmp_path)
    (artifact_dir / "all" / "validation_selection.tsv").write_text("row_id\n")

    with pytest.raises(AssertionError, match="unexpected"):
        validate_artifacts(
            artifact_dir,
            source_dir,
            tmp_path / "manifest.json",
            config_path=config_path,
            pipeline_commit=PIPELINE_COMMIT,
            workers=1,
        )
