"""Fail-closed validation and upload of Hugging Face dataset artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl
import yaml
import zstandard as zstd
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError
from huggingface_hub.hf_api import DatasetInfo


@dataclass(frozen=True)
class ShardResult:
    cohort: str
    split: str
    index: int
    path: str
    compressed_bytes: int
    uncompressed_bytes: int
    rows: int
    sha256: str


def _validate_record(
    raw: bytes,
    *,
    expected_columns: frozenset[str],
    split: str,
    validation_chrom: str,
    target_length: int,
) -> None:
    record = json.loads(raw)
    assert set(record) == expected_columns
    assert isinstance(record["sequence"], str)
    assert len(record["sequence"]) == target_length
    assert record["augmentation"] in {"+", "-"}
    if split == "train":
        assert record["source_chrom"] != validation_chrom
    else:
        assert record["source_chrom"] == validation_chrom
        assert record["augmentation"] == "+"


def _validate_shard(
    task: tuple[Path, str, str, int, frozenset[str], str, int],
) -> ShardResult:
    path, cohort, split, index, columns, validation_chrom, target_length = task
    digest = hashlib.sha256()
    decompressor = zstd.ZstdDecompressor().decompressobj()
    first_line: bytes | None = None
    last_line: bytes | None = None
    tail = b""
    rows = 0
    uncompressed_bytes = 0
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
            decoded = decompressor.decompress(chunk)
            uncompressed_bytes += len(decoded)
            parts = (tail + decoded).split(b"\n")
            tail = parts.pop()
            if parts:
                first_line = parts[0] if first_line is None else first_line
                last_line = parts[-1]
                rows += len(parts)
        decoded = decompressor.flush()
        uncompressed_bytes += len(decoded)
        parts = (tail + decoded).split(b"\n")
        tail = parts.pop()
        if parts:
            first_line = parts[0] if first_line is None else first_line
            last_line = parts[-1]
            rows += len(parts)
    assert decompressor.eof, f"truncated zstd frame: {path}"
    assert tail == b"", f"JSONL file lacks terminal newline: {path}"
    assert first_line is not None and last_line is not None, f"empty shard: {path}"
    for record in [first_line, last_line]:
        _validate_record(
            record,
            expected_columns=columns,
            split=split,
            validation_chrom=validation_chrom,
            target_length=target_length,
        )
    return ShardResult(
        cohort=cohort,
        split=split,
        index=index,
        path=f"data/{split}/shard_{index:04d}.jsonl.zst",
        compressed_bytes=path.stat().st_size,
        uncompressed_bytes=uncompressed_bytes,
        rows=rows,
        sha256=digest.hexdigest(),
    )


def validate_artifacts(
    artifact_dir: str | Path,
    source_datasets_dir: str | Path,
    output_path: str | Path,
    *,
    config_path: str | Path,
    pipeline_commit: str,
    tier: str | None = None,
    workers: int,
) -> dict[str, object]:
    """Validate the exact local artifact tree and persist its content manifest."""
    artifact_root = Path(artifact_dir)
    source_root = Path(source_datasets_dir)
    assert len(pipeline_commit) == 40 and workers > 0
    config = yaml.safe_load(Path(config_path).read_text())
    assert tier in {None, "smoke", "full"}
    cohorts = (
        ["all", "cds", "ccre_non_promoter", "background"]
        if tier == "smoke"
        else list(config["region_cohorts"])
    )
    train_shards = int(
        config[
            "publication_smoke_train_shards"
            if tier == "smoke"
            else "publication_train_shards"
        ]
    )
    validation_shards = int(config["publication_validation_shards"])
    validation_chrom = str(config["validation_chrom"])
    target_length = int(config["target_length"])
    assert len(cohorts) == len(set(cohorts))
    assert train_shards > 0 and validation_shards > 0

    expected_files: set[Path] = set()
    tasks: list[tuple[Path, str, str, int, frozenset[str], str, int]] = []
    cohort_metadata: dict[str, dict[str, object]] = {}
    forbidden_card_text = [
        "bolinas-dna",
        "loss weight",
        "0.01",
        "train.parquet",
        "validation.parquet",
    ]
    for cohort in cohorts:
        card = artifact_root / cohort / "README.md"
        expected_files.add(card.relative_to(artifact_root))
        card_bytes = card.read_bytes()
        card_text = card_bytes.decode()
        assert f"# `marin-dna/vertebrate-v1-{cohort}`" in card_text
        assert f"blob/{pipeline_commit}/" in card_text
        assert "path: data/train/*.jsonl.zst" in card_text
        assert "path: data/validation/*.jsonl.zst" in card_text
        for forbidden in forbidden_card_text:
            assert forbidden not in card_text

        train_source = source_root / cohort / "train.parquet"
        validation_source = source_root / cohort / "validation.parquet"
        train_schema = pl.read_parquet_schema(train_source)
        assert train_schema == pl.read_parquet_schema(validation_source)
        assert {"sequence", "augmentation", "source_chrom"} <= set(train_schema)
        columns = frozenset(train_schema)
        source_rows = {
            split: int(
                pl.scan_parquet(source_root / cohort / f"{split}.parquet")
                .select(pl.len())
                .collect(engine="streaming")
                .item()
            )
            for split in ["train", "validation"]
        }
        assert source_rows["train"] > 0 and source_rows["validation"] > 0
        cohort_metadata[cohort] = {
            "source_rows": source_rows,
            "schema": {name: str(dtype) for name, dtype in train_schema.items()},
            "card_bytes": len(card_bytes),
            "card_sha256": hashlib.sha256(card_bytes).hexdigest(),
        }
        for split, count in [
            ("train", train_shards),
            ("validation", validation_shards),
        ]:
            for index in range(count):
                path = (
                    artifact_root / cohort / f"data/{split}/shard_{index:04d}.jsonl.zst"
                )
                expected_files.add(path.relative_to(artifact_root))
                tasks.append(
                    (
                        path,
                        cohort,
                        split,
                        index,
                        columns,
                        validation_chrom,
                        target_length,
                    )
                )

    actual_files = {
        path.relative_to(artifact_root)
        for path in artifact_root.rglob("*")
        if path.is_file()
    }
    assert actual_files == expected_files, {
        "missing": sorted(str(path) for path in expected_files - actual_files),
        "unexpected": sorted(str(path) for path in actual_files - expected_files),
    }
    with ThreadPoolExecutor(max_workers=workers) as executor:
        shard_results = list(executor.map(_validate_shard, tasks))
    shard_results.sort(key=lambda result: (result.cohort, result.split, result.index))

    manifest_cohorts: dict[str, object] = {}
    for cohort in cohorts:
        metadata = cohort_metadata[cohort]
        source_rows = metadata["source_rows"]
        assert isinstance(source_rows, dict)
        split_manifests: dict[str, object] = {}
        for split in ["train", "validation"]:
            results = [
                result
                for result in shard_results
                if result.cohort == cohort and result.split == split
            ]
            row_counts = [result.rows for result in results]
            assert sum(row_counts) == source_rows[split]
            assert max(row_counts) - min(row_counts) <= 1
            split_manifests[split] = {
                "rows": sum(row_counts),
                "compressed_bytes": sum(x.compressed_bytes for x in results),
                "uncompressed_bytes": sum(x.uncompressed_bytes for x in results),
                "shards": [asdict(result) for result in results],
            }
        manifest_cohorts[cohort] = {
            "schema": metadata["schema"],
            "card_bytes": metadata["card_bytes"],
            "card_sha256": metadata["card_sha256"],
            "splits": split_manifests,
        }
    manifest: dict[str, object] = {
        "pipeline_commit": pipeline_commit,
        "artifact_format": "JSONL.zst",
        "validation_chrom": validation_chrom,
        "target_length": target_length,
        "cohorts": manifest_cohorts,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _expected_remote_files(cohort_manifest: dict[str, object]) -> set[str]:
    expected = {".gitattributes", "README.md"}
    splits = cohort_manifest["splits"]
    assert isinstance(splits, dict)
    for split in ["train", "validation"]:
        split_manifest = splits[split]
        assert isinstance(split_manifest, dict)
        for shard in split_manifest["shards"]:
            assert isinstance(shard, dict)
            expected.add(str(shard["path"]))
    return expected


def _assert_local_cohort(cohort_dir: Path, cohort_manifest: dict[str, object]) -> None:
    expected = _expected_remote_files(cohort_manifest) - {".gitattributes"}
    actual = {
        str(path.relative_to(cohort_dir))
        for path in cohort_dir.rglob("*")
        if path.is_file()
    }
    assert actual == expected, {
        "missing": sorted(expected - actual),
        "unexpected": sorted(actual - expected),
    }
    card = cohort_dir / "README.md"
    assert card.stat().st_size == cohort_manifest["card_bytes"]
    assert (
        hashlib.sha256(card.read_bytes()).hexdigest() == cohort_manifest["card_sha256"]
    )
    splits = cohort_manifest["splits"]
    assert isinstance(splits, dict)
    for split in ["train", "validation"]:
        split_manifest = splits[split]
        assert isinstance(split_manifest, dict)
        for shard in split_manifest["shards"]:
            assert isinstance(shard, dict)
            assert (cohort_dir / str(shard["path"])).stat().st_size == shard[
                "compressed_bytes"
            ]


def _remote_info_or_none(api: HfApi, repo_id: str) -> DatasetInfo | None:
    try:
        return api.dataset_info(repo_id, files_metadata=True)
    except RepositoryNotFoundError:
        return None


def upload_validated_dataset(
    artifact_dir: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    cohort: str,
    repo_id: str,
    workers: int,
) -> None:
    """Reject stale trees, upload one cohort, then verify the exact Hub revision."""
    assert workers > 0
    manifest = json.loads(Path(manifest_path).read_text())
    cohorts = manifest["cohorts"]
    assert isinstance(cohorts, dict) and cohort in cohorts
    cohort_manifest = cohorts[cohort]
    assert isinstance(cohort_manifest, dict)
    cohort_dir = Path(artifact_dir) / cohort
    _assert_local_cohort(cohort_dir, cohort_manifest)
    expected_remote = _expected_remote_files(cohort_manifest)

    api = HfApi()
    before = _remote_info_or_none(api, repo_id)
    if before is not None:
        observed = {item.rfilename for item in before.siblings}
        unexpected = observed - expected_remote
        assert not unexpected, {"repo": repo_id, "unexpected": sorted(unexpected)}

    subprocess.run(
        [
            "hf",
            "upload-large-folder",
            repo_id,
            "--repo-type",
            "dataset",
            "--num-workers",
            str(workers),
            str(cohort_dir),
        ],
        check=True,
        env={**os.environ, "HF_XET_HIGH_PERFORMANCE": "1"},
    )
    subprocess.run(
        [
            "hf",
            "upload",
            repo_id,
            str(cohort_dir / "README.md"),
            "README.md",
            "--repo-type",
            "dataset",
        ],
        check=True,
        env={**os.environ, "HF_XET_HIGH_PERFORMANCE": "1"},
    )

    after = api.dataset_info(repo_id, files_metadata=True)
    assert after.sha is not None and len(after.sha) == 40
    observed = {item.rfilename: item for item in after.siblings}
    assert set(observed) == expected_remote, {
        "repo": repo_id,
        "missing": sorted(expected_remote - set(observed)),
        "unexpected": sorted(set(observed) - expected_remote),
    }
    splits = cohort_manifest["splits"]
    assert isinstance(splits, dict)
    for split in ["train", "validation"]:
        split_manifest = splits[split]
        assert isinstance(split_manifest, dict)
        for shard in split_manifest["shards"]:
            assert isinstance(shard, dict)
            remote = observed[str(shard["path"])]
            assert remote.size == shard["compressed_bytes"]
            assert remote.lfs is not None
            assert remote.lfs.sha256 == shard["sha256"]
    remote_card = Path(
        hf_hub_download(repo_id, "README.md", repo_type="dataset", revision=after.sha)
    ).read_bytes()
    assert hashlib.sha256(remote_card).hexdigest() == cohort_manifest["card_sha256"]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"repo_id": repo_id, "revision": after.sha}, sort_keys=True) + "\n"
    )
