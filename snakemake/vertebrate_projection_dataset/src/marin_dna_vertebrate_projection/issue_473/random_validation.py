"""Uniform row-random validation control for issue #473.

This module is isolated from the established chromosome-18 split and
publication code. It samples original-orientation CDS rows before training
reverse-complement augmentation and publishes one public control dataset.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import zstandard as zstd
from huggingface_hub import HfApi

from marin_dna_vertebrate_projection.publication import upload_validated_dataset

_ROW_INDEX = "__issue473_random_validation_row"
_SAMPLE_HASH = "__issue473_random_validation_hash"


@dataclass(frozen=True)
class ShardResult:
    """Content receipt for one compressed publication shard."""

    split: str
    index: int
    path: str
    compressed_bytes: int
    uncompressed_bytes: int
    rows: int
    sha256: str


def write_uniform_random_split(
    source_path: str | Path,
    train_path: str | Path,
    validation_path: str | Path,
    summary_path: str | Path,
    *,
    region_label: str,
    validation_rows: int,
    seed: int,
    target_length: int,
) -> dict[str, object]:
    """Sample validation rows uniformly before reverse-complement augmentation."""
    assert validation_rows > 0
    assert target_length > 0
    schema = pl.read_parquet_schema(source_path)
    required = {"region_label", "sequence"}
    assert required <= set(schema), f"missing columns: {sorted(required - set(schema))}"
    assert "augmentation" not in schema, "source must contain original-orientation rows"
    assert {_ROW_INDEX, _SAMPLE_HASH}.isdisjoint(schema)

    original = pl.scan_parquet(source_path).filter(
        pl.col("region_label") == region_label
    )
    source_rows = int(original.select(pl.len()).collect(engine="streaming").item())
    assert source_rows > validation_rows

    indexed = original.with_row_index(_ROW_INDEX)
    selected = (
        indexed.select(_ROW_INDEX)
        .with_columns(pl.col(_ROW_INDEX).hash(seed=seed).alias(_SAMPLE_HASH))
        .bottom_k(validation_rows, by=[_SAMPLE_HASH, _ROW_INDEX])
        .collect(engine="streaming")
    )
    assert selected.height == validation_rows
    assert selected[_ROW_INDEX].n_unique() == validation_rows
    selected_ids = selected[_ROW_INDEX].to_list()

    validation = (
        indexed.filter(pl.col(_ROW_INDEX).is_in(selected_ids))
        .sort(_ROW_INDEX)
        .drop(_ROW_INDEX)
        .with_columns(pl.lit("+").alias("augmentation"))
        .collect(engine="streaming")
    )
    assert validation.height == validation_rows
    assert set(validation["augmentation"].to_list()) == {"+"}

    train_output = Path(train_path)
    validation_output = Path(validation_path)
    summary_output = Path(summary_path)
    for path in (train_output, validation_output, summary_output):
        path.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(
        prefix=".random-validation-", dir=train_output.parent
    ) as temp:
        temporary_train = Path(temp) / "train_original.parquet"
        (
            indexed.filter(~pl.col(_ROW_INDEX).is_in(selected_ids))
            .drop(_ROW_INDEX)
            .sink_parquet(temporary_train, engine="streaming")
        )
        temporary_train.replace(train_output)
    validation.write_parquet(validation_output)

    train_rows = int(
        pl.scan_parquet(train_output)
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    )
    assert train_rows == source_rows - validation_rows
    assert pl.read_parquet_schema(train_output) == schema
    assert pl.read_parquet_schema(validation_output) == schema | {
        "augmentation": pl.String
    }

    summary: dict[str, object] = {
        "split_strategy": "uniform_row_random_before_reverse_complement",
        "region_label": region_label,
        "seed": seed,
        "source_rows": source_rows,
        "train_original_rows": train_rows,
        "published_train_rows": train_rows * 2,
        "validation_rows": validation_rows,
        "target_length": target_length,
    }
    summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def write_dataset_card(
    summary_path: str | Path,
    output_path: str | Path,
    *,
    hf_repo: str,
    pipeline_commit: str,
) -> None:
    """Write the public dataset card for the row-random control."""
    assert len(pipeline_commit) == 40
    summary = json.loads(Path(summary_path).read_text())
    assert summary["split_strategy"] == ("uniform_row_random_before_reverse_complement")
    train_rows = int(summary["published_train_rows"])
    validation_rows = int(summary["validation_rows"])
    seed = int(summary["seed"])
    code_url = (
        "https://github.com/Open-Athena/marin-dna/blob/"
        f"{pipeline_commit}/snakemake/vertebrate_projection_dataset/"
        "workflow/Issue473RandomValidation.smk"
    )
    text = f"""---
tags:
- biology
- genomics
- dna
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train/*.jsonl.zst
  - split: validation
    path: data/validation/*.jsonl.zst
---

# `{hf_repo}`

CDS full-window vertebrate projection sequences for the issue #473 random
validation control. The source is the immutable issue #417 accepted-sequence
table.

The split uniformly samples {validation_rows:,} original-orientation CDS rows
without replacement using seed {seed}. Sampling occurs before
reverse-complement augmentation. Selected rows are removed from training;
reverse complements are then added only to the remaining training rows.
Different species projections associated with the same human anchor may occur
on opposite sides of this row-level split.

## Splits

- `train`: {train_rows:,} rows after reverse-complement augmentation.
- `validation`: {validation_rows:,} original-orientation rows
  ({validation_rows * 256:,} DNA-character-plus-BOS tokens).

The complete recipe is in the [commit-pinned additive workflow]({code_url}).
All coordinates are 0-based and half-open. Every sequence is 255 bases and
preserves source repeat-masking case.
"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)


def _validate_record(
    raw: bytes,
    *,
    split: str,
    expected_columns: frozenset[str],
    target_length: int,
) -> None:
    record = json.loads(raw)
    assert set(record) == expected_columns
    assert isinstance(record["sequence"], str)
    assert len(record["sequence"]) == target_length
    assert record["augmentation"] in {"+", "-"}
    if split == "validation":
        assert record["augmentation"] == "+"


def _validate_shard(
    task: tuple[Path, str, int, frozenset[str], int],
) -> ShardResult:
    path, split, index, expected_columns, target_length = task
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
    assert decompressor.eof
    assert tail == b""
    assert first_line is not None and last_line is not None
    for record in (first_line, last_line):
        _validate_record(
            record,
            split=split,
            expected_columns=expected_columns,
            target_length=target_length,
        )
    return ShardResult(
        split=split,
        index=index,
        path=f"data/{split}/shard_{index:04d}.jsonl.zst",
        compressed_bytes=path.stat().st_size,
        uncompressed_bytes=uncompressed_bytes,
        rows=rows,
        sha256=digest.hexdigest(),
    )


def validate_publication(
    artifact_dir: str | Path,
    train_source_path: str | Path,
    validation_source_path: str | Path,
    summary_path: str | Path,
    output_path: str | Path,
    *,
    cohort: str,
    hf_repo: str,
    pipeline_commit: str,
    config_sha256: str,
    train_shards: int,
    validation_shards: int,
    target_length: int,
    workers: int,
) -> dict[str, object]:
    """Validate the exact public tree and write an upload-compatible manifest."""
    assert len(pipeline_commit) == 40
    assert len(config_sha256) == 64
    assert train_shards > 0 and validation_shards > 0 and workers > 0
    summary = json.loads(Path(summary_path).read_text())
    train_original_rows = int(
        pl.scan_parquet(train_source_path)
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    )
    validation_rows = int(
        pl.scan_parquet(validation_source_path)
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    )
    assert train_original_rows == int(summary["train_original_rows"])
    assert validation_rows == int(summary["validation_rows"])
    assert validation_rows == 16_384
    train_schema = pl.read_parquet_schema(train_source_path)
    validation_schema = pl.read_parquet_schema(validation_source_path)
    assert validation_schema == train_schema | {"augmentation": pl.String}
    expected_columns = frozenset(validation_schema)

    root = Path(artifact_dir)
    card = root / cohort / "README.md"
    card_bytes = card.read_bytes()
    card_text = card_bytes.decode()
    assert f"# `{hf_repo}`" in card_text
    assert f"blob/{pipeline_commit}/" in card_text
    expected_files = {card.relative_to(root)}
    tasks: list[tuple[Path, str, int, frozenset[str], int]] = []
    for split, count in (("train", train_shards), ("validation", validation_shards)):
        for index in range(count):
            path = root / cohort / f"data/{split}/shard_{index:04d}.jsonl.zst"
            expected_files.add(path.relative_to(root))
            tasks.append((path, split, index, expected_columns, target_length))
    actual_files = {
        path.relative_to(root) for path in root.rglob("*") if path.is_file()
    }
    assert actual_files == expected_files

    with ThreadPoolExecutor(max_workers=workers) as executor:
        shard_results = list(executor.map(_validate_shard, tasks))
    shard_results.sort(key=lambda item: (item.split, item.index))
    split_manifests: dict[str, object] = {}
    expected_rows = {
        "train": train_original_rows * 2,
        "validation": validation_rows,
    }
    for split in ("train", "validation"):
        results = [item for item in shard_results if item.split == split]
        row_counts = [item.rows for item in results]
        assert sum(row_counts) == expected_rows[split]
        assert max(row_counts) - min(row_counts) <= 1
        split_manifests[split] = {
            "rows": sum(row_counts),
            "compressed_bytes": sum(item.compressed_bytes for item in results),
            "uncompressed_bytes": sum(item.uncompressed_bytes for item in results),
            "shards": [asdict(item) for item in results],
        }

    manifest: dict[str, object] = {
        "producer_commit": pipeline_commit,
        "producer_config_sha256": config_sha256,
        "publication_commit": pipeline_commit,
        "publication_config_sha256": config_sha256,
        "artifact_format": "JSONL.zst",
        "split_strategy": summary["split_strategy"],
        "target_length": target_length,
        "cohorts": {
            cohort: {
                "hf_repo": hf_repo,
                "projection_policy": "full_window",
                "region_label": "cds",
                "schema": {
                    name: str(dtype) for name, dtype in validation_schema.items()
                },
                "card_bytes": len(card_bytes),
                "card_sha256": hashlib.sha256(card_bytes).hexdigest(),
                "splits": split_manifests,
            }
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def upload_public_dataset(
    artifact_dir: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    cohort: str,
    repo_id: str,
    workers: int,
) -> None:
    """Force public visibility, upload, and verify the resulting repository."""
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
    api.update_repo_settings(repo_id, repo_type="dataset", private=False)
    upload_validated_dataset(
        artifact_dir,
        manifest_path,
        output_path,
        cohort=cohort,
        repo_id=repo_id,
        workers=workers,
    )
    assert api.dataset_info(repo_id).private is False
