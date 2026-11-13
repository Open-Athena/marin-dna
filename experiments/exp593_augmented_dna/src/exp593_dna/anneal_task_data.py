"""Freeze seeded benchmark diagnostics, wholly isolated from training inputs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from exp593_dna.audit import file_sha256
from exp593_dna.storage import coreweave_client


def source_catalog() -> dict[str, Any]:
    return json.loads(
        files("exp593_dna").joinpath("assets/anneal_task_sources.json").read_text()
    )


def fetch_source(
    client: Any, record: dict[str, Any], cache: Path, shared: Path | None = None
) -> Path | io.BytesIO:
    """Read an existing immutable cache without changing its owning experiment."""
    candidates = (
        []
        if shared is None
        else [
            shared / "samples" / record["path"],
            shared / "cache" / record["sha256"] / Path(record["path"]).name,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            if (
                candidate.stat().st_size != record["bytes"]
                or file_sha256(candidate) != record["sha256"]
            ):
                raise ValueError("Shared benchmark cache differs from immutable source")
            return candidate
    # A task has at most four source shards; retain only that task in RAM.
    # No new disk cache is needed on the space-constrained operator VM.
    bucket, key = record["uri"].removeprefix("s3://").split("/", 1)
    response = client.get_object(Bucket=bucket, Key=key)
    payload = io.BytesIO()
    digest = hashlib.sha256()
    with response["Body"] as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
            payload.write(block)
    if payload.tell() != record["bytes"] or digest.hexdigest() != record["sha256"]:
        raise ValueError("Remote benchmark source differs from frozen identity")
    payload.seek(0)
    return payload


def sample_rows(
    paths: list[Path | io.BytesIO], records: list[dict[str, Any]], count: int, seed: int
) -> tuple[list[dict[str, Any]], list[int]]:
    """Uniform across the entire task, never just its first Parquet shard."""
    total = sum(r["rows"] for r in records)
    chosen = np.sort(
        np.random.default_rng(seed).choice(total, min(count, total), replace=False)
    )
    result = []
    global_offset = 0
    for path, record in zip(paths, records, strict=True):
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != record["rows"]:
            raise ValueError("Benchmark source row count differs")
        offset = global_offset
        for group in range(parquet.num_row_groups):
            size = parquet.metadata.row_group(group).num_rows
            points = chosen[(chosen >= offset) & (chosen < offset + size)]
            if len(points):
                table = parquet.read_row_group(group).take(pa.array(points - offset))
                rows = table.to_pylist()
                for index, row in zip(points.tolist(), rows, strict=True):
                    result.append({"monitor_source_row": index, **row})
            offset += size
        global_offset += record["rows"]
    if [r["monitor_source_row"] for r in result] != chosen.tolist():
        raise ValueError("Diagnostic membership is missing, duplicated or out of order")
    return result, chosen.tolist()


def prepare(output: Path, cache: Path, *, shared: Path | None = None) -> dict[str, Any]:
    catalog = source_catalog()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise ValueError("Do not replace a frozen monitoring panel")
    client = coreweave_client(refresh_keys=True)
    specifications = [
        *catalog["catalog"]["tasks"],
        {"key": "maize_af", "kind": "af", "file": "maize-af.parquet"},
    ]
    manifest: dict[str, Any] = {
        "protocol": "exp593-anneal-zero-shot-monitoring-v1",
        "use": "evaluation_only",
        "source_catalog_sha256": hashlib.sha256(canonical_catalog(catalog)).hexdigest(),
        "development_policy": catalog["development_policy"],
        "sampling": catalog["sampling"],
        "tasks": {},
    }
    for number, spec in enumerate(specifications):
        source = catalog["inputs"][spec["file"]]
        seed = 593_100 + number
        paths = [fetch_source(client, r, cache, shared) for r in source["files"]]
        rows, membership = sample_rows(
            paths, source["files"], catalog["samples_per_task"], seed
        )
        for path in paths:
            if isinstance(path, io.BytesIO):
                path.close()
        del paths
        if spec["kind"] != "af" and spec["kind"] != "motif":
            labels = {int(r["label"]) for r in rows}
            if labels != {0, 1}:
                raise ValueError("A diagnostic binary task lacks both classes")
        for row in rows:
            for field in ("sequence", "RefSeq", "MutSeq"):
                if field in row and not set(row[field].upper()) <= set(
                    "ACGTNRYSWKMBDHV"
                ):
                    raise ValueError("Benchmark DNA contains an unexpected alphabet")
        path = output / (spec["key"] + ".jsonl.gz")
        with (
            path.open("wb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream,
        ):
            for row in rows:
                stream.write(
                    (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode()
                )
        counts: dict[str, int] = {}
        if "label" in rows[0]:
            for row in rows:
                name = str(row["label"])
                counts[name] = counts.get(name, 0) + 1
        manifest["tasks"][spec["key"]] = {
            "spec": spec,
            "path": path.name,
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
            "rows": len(rows),
            "seed": seed,
            "source_rows": source["rows"],
            "membership": membership,
            "label_counts": counts,
            "source_files": source["files"],
        }
        print(
            json.dumps(
                {"task": spec["key"], "rows": len(rows), "label_counts": counts}
            ),
            flush=True,
        )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def canonical_catalog(catalog: dict[str, Any]) -> bytes:
    return json.dumps(catalog, sort_keys=True, separators=(",", ":")).encode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--shared-cache", type=Path)
    args = parser.parse_args()
    prepare(args.output, args.cache, shared=args.shared_cache)


if __name__ == "__main__":
    main()
