"""Sequence-only Hub shards with producer-owned row mappings and release hashes."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from marin_dna_vertebrate_projection.rag.documents import stable_digest
from marin_dna_vertebrate_projection.rag.tables import (
    TRAIN_SCHEMA,
    read_rows,
    sha256_file,
)

PUBLIC_SCHEMA = pa.schema([("sequence", pa.string())])
MAP_SCHEMA = pa.schema(
    [
        *[field for field in TRAIN_SCHEMA if field.name != "sequence"],
        ("split", pa.string()),
        ("public_shard", pa.string()),
        ("public_row_index", pa.int64()),
    ]
)


def prepare_release(
    inputs: Mapping[str, str],
    directory: str,
    mapping_output: str,
    manifest_output: str,
    *,
    region: str,
    repo_id: str,
    producer: Mapping[str, str],
    train_shards: int = 16,
) -> dict[str, Any]:
    if set(inputs) != {"train", "validation"} or train_shards < 1:
        raise ValueError("publication requires exactly train and validation")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    Path(mapping_output).parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "repo_id": repo_id,
        "region": region,
        "producer": dict(producer),
        "source_files": {},
        "splits": {},
        "files": {},
    }
    with pq.ParquetWriter(
        mapping_output, MAP_SCHEMA, compression="zstd"
    ) as mapping_writer:
        for split, path in sorted(inputs.items()):
            shards = train_shards if split == "train" else 1
            paths = [
                root / f"{split}/{index:05d}-of-{shards:05d}.parquet"
                for index in range(shards)
            ]
            for shard in paths:
                shard.parent.mkdir(parents=True, exist_ok=True)
            writers = [
                pq.ParquetWriter(shard, PUBLIC_SCHEMA, compression="zstd")
                for shard in paths
            ]
            buffers: list[list[dict[str, str]]] = [[] for _ in paths]
            counts = [0] * shards
            mappings = []
            unpadded = 0
            identities = set()
            try:
                for row in read_rows(path):
                    identity = stable_digest(
                        region, row["source_row_id"], row["orientation"]
                    )
                    if identity in identities:
                        raise ValueError("duplicate public source row identity")
                    identities.add(identity)
                    if (split == "train") == (row["source_chrom"] == "chr18"):
                        raise ValueError("public row violates chr18 split")
                    if split == "validation" and row["orientation"] != "forward":
                        raise ValueError(
                            "validation publication contains reverse-complement augmentation"
                        )
                    index = int(identity[:16], 16) % shards
                    buffers[index].append({"sequence": row["sequence"]})
                    mappings.append(
                        {
                            **{
                                key: value
                                for key, value in row.items()
                                if key != "sequence"
                            },
                            "split": split,
                            "public_shard": str(paths[index].relative_to(root)),
                            "public_row_index": counts[index],
                        }
                    )
                    counts[index] += 1
                    unpadded += row["unpadded_tokens"]
                    if len(buffers[index]) == 256:
                        writers[index].write_table(
                            pa.Table.from_pylist(buffers[index], schema=PUBLIC_SCHEMA)
                        )
                        buffers[index] = []
                    if len(mappings) == 4096:
                        mapping_writer.write_table(
                            pa.Table.from_pylist(mappings, schema=MAP_SCHEMA)
                        )
                        mappings = []
                for index, buffer in enumerate(buffers):
                    if buffer:
                        writers[index].write_table(
                            pa.Table.from_pylist(buffer, schema=PUBLIC_SCHEMA)
                        )
                if mappings:
                    mapping_writer.write_table(
                        pa.Table.from_pylist(mappings, schema=MAP_SCHEMA)
                    )
            finally:
                for writer in writers:
                    writer.close()
            manifest["source_files"][split] = {"sha256": sha256_file(path)}
            manifest["splits"][split] = {
                "rows": sum(counts),
                "unpadded_tokens": unpadded,
                "allocated_tokens": sum(counts) * 10240,
                "shard_rows": counts,
            }
            for index, shard in enumerate(paths):
                parquet = pq.ParquetFile(shard)
                if (
                    parquet.schema_arrow != PUBLIC_SCHEMA
                    or parquet.metadata.num_rows != counts[index]
                ):
                    raise ValueError("public shard schema or count mismatch")
    card = f"""---
license: openmdw-1.1
tags: [biology, genomics, dna]
configs:
- config_name: default
  data_files:
  - split: train
    path: train/*.parquet
  - split: validation
    path: validation/*.parquet
---

# {repo_id}

Human-anchored RAG documents for the `{region}` region in [MarinDNA experiment 550](https://github.com/Open-Athena/marin-dna/issues/550).
The [producing workflow](https://github.com/Open-Athena/marin-dna/tree/{producer["pipeline_commit"]}/snakemake/vertebrate_projection_dataset) owns source identities, projected coordinates, sequence provenance, public shard row mappings, and release checksums.
Its immutable artifact root is `{producer["root"]}`.

Both splits contain only a string `sequence` column.
Training has {manifest["splits"]["train"]["rows"]:,} rows; validation has {manifest["splits"]["validation"]["rows"]:,} rows sampled from the complete chr18 training holdout.
Every retained training locus contributes forward and reverse-complement rows.
Each document joins available 255-bp species windows with atomic `[SEQ]` separators in a fixed per-row permutation.
Human is included once; up to 18 non-human mammalian and 21 other vertebrate order representatives supply context.
Missing projections are omitted, genuine Ns and source letter case are retained, and reverse complementation acts within each segment.
Coordinates in producer artifacts use hg38 and 0-based, half-open intervals.

The training consumer adds one BOS token and right padding to 10,240 positions, with padding targets excluded from loss.
The strings here contain neither BOS nor padding.
Public shard assignment is deterministic; `public_row_index` in the producer's row mapping identifies each exact published document.

MarinDNA releases the processed dataset under OpenMDW 1.1.
The underlying public genome assemblies and alignments retain their original source terms and attribution, recorded in the pinned source manifests.
This release adds document assembly and does not relicense the underlying source assets.
"""
    (root / "README.md").write_text(card)
    for path in sorted(root.rglob("*")):
        if path.is_file():
            manifest["files"][str(path.relative_to(root))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    manifest["row_mapping"] = {"sha256": sha256_file(mapping_output)}
    Path(manifest_output).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_output).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def publish_release(directory: str, manifest_path: str, receipt_path: str) -> None:
    """Commit the validated release atomically and verify anonymous Hub access."""
    from huggingface_hub import HfApi, hf_hub_download

    manifest = json.loads(Path(manifest_path).read_text())
    root = Path(directory)
    observed = {
        str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()
    }
    if observed != set(manifest["files"]):
        raise ValueError("publication folder differs from its release manifest")
    for name, expected in manifest["files"].items():
        path = root / name
        if (
            path.stat().st_size != expected["bytes"]
            or sha256_file(path) != expected["sha256"]
        ):
            raise ValueError("publication file changed after release validation")
    api = HfApi()
    repo = manifest["repo_id"]
    api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
    if api.repo_info(repo, repo_type="dataset").private:
        raise ValueError("MarinDNA publication cannot use a private repository")
    commit = api.upload_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=root,
        commit_message="Publish reproducible RAG train and chr18 validation release",
        delete_patterns=["train/*", "validation/*", "README.md"],
    )
    anonymous = HfApi(token=False)
    info = anonymous.repo_info(
        repo, repo_type="dataset", revision=commit.oid, files_metadata=True
    )
    if info.private or info.gated:
        raise ValueError("published dataset is not anonymously accessible")
    files = {
        item.rfilename: item
        for item in info.siblings
        if item.rfilename != ".gitattributes"
    }
    if set(files) != set(manifest["files"]):
        raise ValueError("published file inventory differs from release")
    for name, expected in manifest["files"].items():
        item = files[name]
        if item.size != expected["bytes"]:
            raise ValueError("published file size mismatch")
        if item.lfs is not None:
            if item.lfs.sha256 != expected["sha256"]:
                raise ValueError("published LFS checksum mismatch")
        else:
            downloaded = hf_hub_download(
                repo, name, repo_type="dataset", revision=commit.oid, token=False
            )
            if sha256_file(downloaded) != expected["sha256"]:
                raise ValueError("published file checksum mismatch")
    validation_path = hf_hub_download(
        repo,
        "validation/00000-of-00001.parquet",
        repo_type="dataset",
        revision=commit.oid,
        token=False,
    )
    if (
        sha256_file(validation_path)
        != manifest["files"]["validation/00000-of-00001.parquet"]["sha256"]
    ):
        raise ValueError("anonymous validation download checksum mismatch")
    if pq.ParquetFile(validation_path).schema_arrow != PUBLIC_SCHEMA:
        raise ValueError("anonymous dataset download has the wrong schema")
    Path(receipt_path).parent.mkdir(parents=True, exist_ok=True)
    Path(receipt_path).write_text(
        json.dumps(
            {
                "repo_id": repo,
                "revision": commit.oid,
                "release_manifest_sha256": sha256_file(manifest_path),
                "anonymous_verified": True,
            },
            indent=2,
        )
        + "\n"
    )
