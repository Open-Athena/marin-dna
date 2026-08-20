"""Reviewable, fail-closed Hugging Face artifacts for issue #473."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl
import zstandard as zstd
from huggingface_hub import HfApi

from marin_dna_vertebrate_projection.manifest import read_species_manifest
from marin_dna_vertebrate_projection.publication import upload_validated_dataset


@dataclass(frozen=True)
class PublicationDataset:
    """One immutable #473 publication dataset."""

    key: str
    hf_repo: str
    projection_policy: str
    region_label: str


@dataclass(frozen=True)
class ShardResult:
    """Validated content metadata for one compressed shard."""

    dataset: str
    split: str
    index: int
    path: str
    compressed_bytes: int
    uncompressed_bytes: int
    rows: int
    sha256: str


def parse_publication_datasets(raw: dict[str, object]) -> list[PublicationDataset]:
    """Parse and validate the additive publication mapping from config."""
    datasets: list[PublicationDataset] = []
    for key, value in raw.items():
        assert isinstance(value, dict)
        datasets.append(
            PublicationDataset(
                key=key,
                hf_repo=str(value["hf_repo"]),
                projection_policy=str(value["projection_policy"]),
                region_label=str(value["region_label"]),
            )
        )
    assert datasets and len({item.key for item in datasets}) == len(datasets)
    assert len({item.hf_repo for item in datasets}) == len(datasets)
    assert {item.projection_policy for item in datasets} <= {
        "full_window",
        "center_1",
    }
    assert {item.region_label for item in datasets} <= {
        "cds",
        "ccre_enhancer_centered",
    }
    return datasets


def _row_count(path: str | Path) -> int:
    return int(
        pl.scan_parquet(path).select(pl.len()).collect(engine="streaming").item()
    )


def _assert_split_contract(
    train_path: str | Path,
    validation_path: str | Path,
    *,
    validation_chrom: str,
    target_length: int,
) -> tuple[int, int, dict[str, pl.DataType]]:
    train_schema = pl.read_parquet_schema(train_path)
    validation_schema = pl.read_parquet_schema(validation_path)
    assert train_schema == validation_schema
    assert {"source_chrom", "sequence", "augmentation"} <= set(train_schema)
    checks = {
        "train_rows": pl.len(),
        "train_validation_chrom_rows": (
            pl.col("source_chrom") == validation_chrom
        ).sum(),
        "train_bad_length_rows": (
            pl.col("sequence").str.len_chars() != target_length
        ).sum(),
        "train_bad_augmentation_rows": (
            ~pl.col("augmentation").is_in(["+", "-"])
        ).sum(),
        "train_forward_rows": (pl.col("augmentation") == "+").sum(),
        "train_reverse_rows": (pl.col("augmentation") == "-").sum(),
    }
    train = pl.scan_parquet(train_path).select(**checks).collect(engine="streaming")
    validation = (
        pl.scan_parquet(validation_path)
        .select(
            validation_rows=pl.len(),
            validation_wrong_chrom_rows=(
                pl.col("source_chrom") != validation_chrom
            ).sum(),
            validation_bad_length_rows=(
                pl.col("sequence").str.len_chars() != target_length
            ).sum(),
            validation_bad_augmentation_rows=(pl.col("augmentation") != "+").sum(),
        )
        .collect(engine="streaming")
    )
    assert train["train_validation_chrom_rows"].item() == 0
    assert train["train_bad_length_rows"].item() == 0
    assert train["train_bad_augmentation_rows"].item() == 0
    assert train["train_forward_rows"].item() > 0
    assert train["train_forward_rows"].item() == train["train_reverse_rows"].item()
    assert validation["validation_wrong_chrom_rows"].item() == 0
    assert validation["validation_bad_length_rows"].item() == 0
    assert validation["validation_bad_augmentation_rows"].item() == 0
    train_rows = int(train["train_rows"].item())
    validation_rows = int(validation["validation_rows"].item())
    assert train_rows > 0 and validation_rows > 0
    return train_rows, validation_rows, train_schema


def write_dataset_card(
    train_path: str | Path,
    validation_path: str | Path,
    split_summary_path: str | Path,
    species_manifest_path: str | Path,
    output_path: str | Path,
    *,
    dataset: PublicationDataset,
    producer_commit: str,
    producer_config_sha256: str,
    publication_commit: str,
    validation_chrom: str,
    target_length: int,
) -> None:
    """Write a complete draft card whose generated values are independently checked."""
    assert len(producer_commit) == 40 and len(publication_commit) == 40
    assert len(producer_config_sha256) == 64
    train_rows, validation_rows, schema = _assert_split_contract(
        train_path,
        validation_path,
        validation_chrom=validation_chrom,
        target_length=target_length,
    )
    split_summary = json.loads(Path(split_summary_path).read_text())
    assert split_summary["train_rows"] == train_rows
    assert split_summary["validation_rows"] == validation_rows
    assert split_summary["validation_chrom"] == validation_chrom
    assert split_summary["region_label"] == dataset.region_label

    selected = read_species_manifest(str(species_manifest_path)).filter(
        pl.col("selected")
    )
    species_counts = (
        selected.group_by("backend", "clade")
        .len(name="species")
        .sort("backend", "clade")
    )
    species_lines = "\n".join(
        f"| {backend} | {clade} | {count} |"
        for backend, clade, count in species_counts.iter_rows()
    )
    schema_lines = "\n".join(
        f"- `{column}`: `{dtype}`" for column, dtype in schema.items()
    )
    policy_text = {
        "center_1": (
            "projects the exact central human nucleotide, requires one target locus, "
            "and emits the 255 bp target window centered on that mapped nucleotide"
        ),
        "full_window": (
            "projects the complete 255 bp human window, applies the established "
            "128--512 bp compatible-fragment gate, and resizes around the accepted "
            "target-span midpoint"
        ),
    }[dataset.projection_policy]
    anchor_text = {
        "cds": "the fixed #417 protein-coding CDS anchor catalog",
        "ccre_enhancer_centered": (
            "the fixed exp351 ENCODE dELS/pELS-centered catalog after exon-overlap "
            "removal and the preregistered 20% conservation eligibility gate"
        ),
    }[dataset.region_label]
    producer_url = (
        "https://github.com/Open-Athena/marin-dna/blob/"
        f"{producer_commit}/snakemake/vertebrate_projection_dataset/"
        "experiments/issue_473/README.md"
    )
    publication_url = (
        "https://github.com/Open-Athena/marin-dna/blob/"
        f"{publication_commit}/snakemake/vertebrate_projection_dataset/"
        "workflow/Issue473Publication.smk"
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

# `{dataset.hf_repo}`

Review status: **draft generated for issue #473 review before upload**.

Human-anchored 255 bp vertebrate sequences for the
`{dataset.region_label}` cohort under the `{dataset.projection_policy}` policy.
The policy {policy_text}. Human anchors come from {anchor_text}.

The source projection was produced by the [commit-pinned issue #473
workflow]({producer_url}) at config SHA-256
`{producer_config_sha256}`. These publication artifacts were built by the
[commit-pinned additive publisher]({publication_url}).

## Splits

- `train`: {train_rows:,} rows after deterministic reverse-complement
  augmentation; no `{validation_chrom}` source anchors.
- `validation`: {validation_rows:,} original-orientation `{validation_chrom}`
  rows ({validation_rows * 256:,} DNA-character-plus-BOS tokens).

The selected target manifest contains {selected.height:,} family-deduplicated
projection targets. One human-reference row is also emitted per anchor. Train
and validation are split by the human source chromosome, not the target locus.

| Projection backend | Clade | Selected species |
|---|---|---:|
{species_lines}

## Sequence semantics

All coordinates inside the producer are 0-based, half-open. Every emitted
sequence is exactly {target_length} bases and is oriented to its human anchor.
Letter case is preserved from FASTA/2bit inputs: lowercase denotes source
repeat masking and uppercase denotes source non-repeat-masked sequence.
Conservation scores do not rewrite sequence characters or case.

Center-seeded recovery is not proof that both 127 bp flanks are homologous.
Issue #473 reports paired recovery diagnostics and a sampled bidirectional HAL
alignment trace separately; users should not substitute target-span length for
aligned base coverage.

## Schema

{schema_lines}

## Intended use and limitations

This dataset is intended for matched-token genomic language-model research.
It is not a clinical resource. Assemblies, alignment gaps, repeat masking,
family-deduplicated species selection, and the projection acceptance contract
all affect the observed sequence distribution. The chromosome-18 split is a
pipeline validation split and must not be confused with the protected final
variant-effect test chromosomes used by MarinDNA.
"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)


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
    path, dataset, split, index, columns, validation_chrom, target_length = task
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
        dataset=dataset,
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
    datasets: list[PublicationDataset],
    producer_commit: str,
    producer_config_sha256: str,
    publication_commit: str,
    publication_config_sha256: str,
    train_shards: int,
    validation_shards: int,
    validation_chrom: str,
    target_length: int,
    workers: int,
) -> dict[str, object]:
    """Validate the exact #473 artifact trees and write a content manifest."""
    assert workers > 0 and train_shards > 0 and validation_shards > 0
    assert len(producer_commit) == 40 and len(publication_commit) == 40
    assert len(producer_config_sha256) == 64
    assert len(publication_config_sha256) == 64
    artifact_root = Path(artifact_dir)
    source_root = Path(source_datasets_dir)
    expected_files: set[Path] = set()
    tasks: list[tuple[Path, str, str, int, frozenset[str], str, int]] = []
    metadata: dict[str, dict[str, object]] = {}
    for dataset in datasets:
        card = artifact_root / dataset.key / "README.md"
        expected_files.add(card.relative_to(artifact_root))
        card_bytes = card.read_bytes()
        card_text = card_bytes.decode()
        assert f"# `{dataset.hf_repo}`" in card_text
        assert f"blob/{producer_commit}/" in card_text
        assert f"blob/{publication_commit}/" in card_text
        assert "path: data/train/*.jsonl.zst" in card_text
        assert "path: data/validation/*.jsonl.zst" in card_text

        dataset_source = source_root / dataset.projection_policy / dataset.region_label
        train_source = dataset_source / "train.parquet"
        validation_source = dataset_source / "validation.parquet"
        train_rows, validation_rows, schema = _assert_split_contract(
            train_source,
            validation_source,
            validation_chrom=validation_chrom,
            target_length=target_length,
        )
        columns = frozenset(schema)
        metadata[dataset.key] = {
            "hf_repo": dataset.hf_repo,
            "projection_policy": dataset.projection_policy,
            "region_label": dataset.region_label,
            "source_rows": {"train": train_rows, "validation": validation_rows},
            "schema": {name: str(dtype) for name, dtype in schema.items()},
            "card_bytes": len(card_bytes),
            "card_sha256": hashlib.sha256(card_bytes).hexdigest(),
        }
        for split, count in [
            ("train", train_shards),
            ("validation", validation_shards),
        ]:
            for index in range(count):
                path = (
                    artifact_root
                    / dataset.key
                    / f"data/{split}/shard_{index:04d}.jsonl.zst"
                )
                expected_files.add(path.relative_to(artifact_root))
                tasks.append(
                    (
                        path,
                        dataset.key,
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
    shard_results.sort(key=lambda result: (result.dataset, result.split, result.index))

    manifest_datasets: dict[str, object] = {}
    for dataset in datasets:
        dataset_metadata = metadata[dataset.key]
        source_rows = dataset_metadata["source_rows"]
        assert isinstance(source_rows, dict)
        split_manifests: dict[str, object] = {}
        for split in ["train", "validation"]:
            results = [
                result
                for result in shard_results
                if result.dataset == dataset.key and result.split == split
            ]
            row_counts = [result.rows for result in results]
            assert sum(row_counts) == source_rows[split]
            assert max(row_counts) - min(row_counts) <= 1
            split_manifests[split] = {
                "rows": sum(row_counts),
                "compressed_bytes": sum(item.compressed_bytes for item in results),
                "uncompressed_bytes": sum(item.uncompressed_bytes for item in results),
                "shards": [asdict(result) for result in results],
            }
        manifest_datasets[dataset.key] = {
            key: value
            for key, value in dataset_metadata.items()
            if key != "source_rows"
        } | {"splits": split_manifests}
    manifest: dict[str, object] = {
        "producer_commit": producer_commit,
        "producer_config_sha256": producer_config_sha256,
        "publication_commit": publication_commit,
        "publication_config_sha256": publication_config_sha256,
        "artifact_format": "JSONL.zst",
        "validation_chrom": validation_chrom,
        "target_length": target_length,
        "cohorts": manifest_datasets,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def upload_private_validated_dataset(
    artifact_dir: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    cohort: str,
    repo_id: str,
    workers: int,
) -> None:
    """Create or verify a private #473 repo, upload, then recheck privacy."""
    api = HfApi()
    if not api.repo_exists(repo_id, repo_type="dataset"):
        api.create_repo(
            repo_id,
            repo_type="dataset",
            private=True,
            exist_ok=False,
        )
    before = api.dataset_info(repo_id)
    assert before.private is True, f"refusing to upload #473 data publicly: {repo_id}"

    upload_validated_dataset(
        artifact_dir,
        manifest_path,
        output_path,
        cohort=cohort,
        repo_id=repo_id,
        workers=workers,
    )

    after = api.dataset_info(repo_id)
    assert after.private is True, f"#473 dataset became public: {repo_id}"
