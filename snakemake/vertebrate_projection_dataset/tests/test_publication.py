from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
import zstandard as zstd
from marin_dna_vertebrate_projection import publication

PIPELINE_COMMIT = "a" * 40
CONFIG_SHA256 = "b" * 64


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(zstd.ZstdCompressor().compress(frame.write_ndjson().encode()))


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    artifact_dir = tmp_path / "hf"
    source_dir = tmp_path / "datasets"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "region_cohorts: [all]\n"
        "hf_owner: marin-dna\n"
        "hf_repo_prefix: vertebrate-v2\n"
        "publication_train_shards: 2\n"
        "publication_smoke_train_shards: 1\n"
        "publication_validation_shards: 1\n"
        "validation_chrom: chr18\n"
        "target_length: 255\n"
    )
    train = _frame("chr1", 5)
    validation = _frame("chr18", 2)
    (source_dir / "all").mkdir(parents=True)
    train.write_parquet(source_dir / "all/train.parquet")
    validation.write_parquet(source_dir / "all/validation.parquet")
    card = artifact_dir / "all/README.md"
    card.parent.mkdir(parents=True)
    card.write_text(
        "# `marin-dna/vertebrate-v2-all`\n"
        f"https://github.com/Open-Athena/marin-dna/blob/{PIPELINE_COMMIT}/README.md\n"
        "path: data/train/*.jsonl.zst\n"
        "path: data/validation/*.jsonl.zst\n"
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


def test_validate_artifacts_reconciles_rows_and_rejects_sidecars(
    tmp_path: Path,
) -> None:
    artifact_dir, source_dir, config_path = _fixture(tmp_path)
    output = tmp_path / "manifest.json"
    manifest = publication.validate_artifacts(
        artifact_dir,
        source_dir,
        output,
        config_path=config_path,
        pipeline_commit=PIPELINE_COMMIT,
        config_sha256=CONFIG_SHA256,
        workers=1,
    )
    assert json.loads(output.read_text()) == manifest
    assert manifest["config_sha256"] == CONFIG_SHA256
    assert manifest["cohorts"]["all"]["splits"]["train"]["rows"] == 5
    (artifact_dir / "all/validation_selection.tsv").write_text("row_id\n")
    with pytest.raises(AssertionError, match="unexpected"):
        publication.validate_artifacts(
            artifact_dir,
            source_dir,
            output,
            config_path=config_path,
            pipeline_commit=PIPELINE_COMMIT,
            config_sha256=CONFIG_SHA256,
            workers=1,
        )


def test_upload_rejects_unexpected_remote_file_before_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_dir, source_dir, config_path = _fixture(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    publication.validate_artifacts(
        artifact_dir,
        source_dir,
        manifest_path,
        config_path=config_path,
        pipeline_commit=PIPELINE_COMMIT,
        config_sha256=CONFIG_SHA256,
        workers=1,
    )

    class FakeApi:
        def dataset_info(self, *_args: object, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                siblings=[
                    SimpleNamespace(rfilename=".gitattributes"),
                    SimpleNamespace(rfilename="README.md"),
                    SimpleNamespace(rfilename="stale.tsv"),
                ]
            )

    monkeypatch.setattr(publication, "HfApi", FakeApi)
    called = False

    def fail_run(*_args: object, **_kwargs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(publication.subprocess, "run", fail_run)
    with pytest.raises(AssertionError, match="stale.tsv"):
        publication.upload_validated_dataset(
            artifact_dir,
            manifest_path,
            tmp_path / "done.json",
            cohort="all",
            repo_id="marin-dna/vertebrate-v2-all",
            workers=1,
        )
    assert not called
