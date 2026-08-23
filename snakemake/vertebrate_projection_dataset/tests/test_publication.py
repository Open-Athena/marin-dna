from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
import zstandard as zstd
from marin_dna_vertebrate_projection import publication
from marin_dna_vertebrate_projection.pipeline_io import write_dataset_split_files

PIPELINE_COMMIT = "a" * 40
CONFIG_SHA256 = "b" * 64


def _frame(rows: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "query_name": [f"anchor_{index}" for index in range(rows)],
            "source_chrom": [
                "chr1" if index % 2 == 0 else "chr18" for index in range(rows)
            ],
            "source_start": [index * 255 for index in range(rows)],
            "source_end": [(index + 1) * 255 for index in range(rows)],
            "species": ["Homo sapiens"] * rows,
            "alignment_source": ["human_reference"] * rows,
            "region_label": ["cds"] * rows,
            "sequence": ["ACgt" + "A" * 251] * rows,
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
        "pipeline_version: v2\n"
        "publication_train_shards: 2\n"
        "publication_smoke_train_shards: 1\n"
        "publication_validation_shards: 1\n"
        "validation_rows: 2\n"
        "smoke_validation_rows: 1\n"
        "validation_seed: 42\n"
        "add_rc: false\n"
    )
    (source_dir / "all").mkdir(parents=True)
    combined = source_dir / "combined.parquet"
    _frame(7).write_parquet(combined)
    write_dataset_split_files(
        combined,
        source_dir / "all/train.parquet",
        source_dir / "all/validation.parquet",
        source_dir / "all/validation_selection.tsv",
        source_dir / "all/validation_composition.tsv",
        source_dir / "all/split_summary.json",
        region_label="all",
        add_rc=False,
        validation_rows=2,
        seed=42,
    )
    train = pl.read_parquet(source_dir / "all/train.parquet")
    validation = pl.read_parquet(source_dir / "all/validation.parquet")
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


def test_validate_artifacts_rejects_tampered_composition(tmp_path: Path) -> None:
    artifact_dir, source_dir, config_path = _fixture(tmp_path)
    composition_path = source_dir / "all/validation_composition.tsv"
    composition = pl.read_csv(composition_path, separator="\t")
    rows = composition.to_dicts()
    chrom_indices = [
        index for index, row in enumerate(rows) if row["dimension"] == "source_chrom"
    ]
    assert len(chrom_indices) == 2
    first, second = chrom_indices
    rows[first]["eligible_rows"] = int(rows[first]["eligible_rows"]) + 1
    rows[second]["eligible_rows"] = int(rows[second]["eligible_rows"]) - 1
    pl.DataFrame(rows, schema=composition.schema).write_csv(
        composition_path, separator="\t"
    )
    with pytest.raises(AssertionError):
        publication.validate_artifacts(
            artifact_dir,
            source_dir,
            tmp_path / "manifest.json",
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
                private=False,
                gated=False,
                siblings=[
                    SimpleNamespace(rfilename=".gitattributes"),
                    SimpleNamespace(rfilename="README.md"),
                    SimpleNamespace(rfilename="stale.tsv"),
                ],
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


def test_upload_rejects_private_existing_repository(
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
            return SimpleNamespace(private=True, gated=False, siblings=[])

    monkeypatch.setattr(publication, "HfApi", FakeApi)
    with pytest.raises(AssertionError, match="must be public"):
        publication.upload_validated_dataset(
            artifact_dir,
            manifest_path,
            tmp_path / "done.json",
            cohort="all",
            repo_id="marin-dna/vertebrate-v2-all",
            workers=1,
        )


def test_upload_requests_and_verifies_public_access_without_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_dir, source_dir, config_path = _fixture(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = publication.validate_artifacts(
        artifact_dir,
        source_dir,
        manifest_path,
        config_path=config_path,
        pipeline_commit=PIPELINE_COMMIT,
        config_sha256=CONFIG_SHA256,
        workers=1,
    )
    cohort_manifest = manifest["cohorts"]["all"]
    shards = {
        shard["path"]: shard
        for split in cohort_manifest["splits"].values()
        for shard in split["shards"]
    }
    siblings = []
    for path in sorted(publication._expected_remote_files(cohort_manifest)):
        shard = shards.get(path)
        siblings.append(
            SimpleNamespace(
                rfilename=path,
                size=None if shard is None else shard["compressed_bytes"],
                lfs=(
                    None if shard is None else SimpleNamespace(sha256=shard["sha256"])
                ),
            )
        )
    remote_info = SimpleNamespace(
        private=False, gated=False, sha="c" * 40, siblings=siblings
    )
    api_tokens: list[object] = []

    class FakeApi:
        def __init__(self, *, token: object = None) -> None:
            api_tokens.append(token)

        def dataset_info(self, *_args: object, **_kwargs: object) -> SimpleNamespace:
            return remote_info

    commands: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        commands.append(command)
        return SimpleNamespace(returncode=0)

    download_kwargs: list[dict[str, object]] = []

    def fake_download(*_args: object, **kwargs: object) -> str:
        download_kwargs.append(kwargs)
        return str(artifact_dir / "all/README.md")

    monkeypatch.setattr(publication, "HfApi", FakeApi)
    monkeypatch.setattr(publication.subprocess, "run", fake_run)
    monkeypatch.setattr(publication, "hf_hub_download", fake_download)

    output_path = tmp_path / "done.json"
    publication.upload_validated_dataset(
        artifact_dir,
        manifest_path,
        output_path,
        cohort="all",
        repo_id="marin-dna/vertebrate-v2-all",
        workers=1,
    )

    assert "--no-private" in commands[0]
    assert False in api_tokens
    assert download_kwargs == [
        {
            "repo_type": "dataset",
            "revision": "c" * 40,
            "token": False,
        }
    ]
    assert json.loads(output_path.read_text()) == {
        "repo_id": "marin-dna/vertebrate-v2-all",
        "revision": "c" * 40,
    }
