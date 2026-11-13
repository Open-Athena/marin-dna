"""Materialize the original immutable exp586 corpus for train-only CPU preparation."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pyarrow.parquet as pq

from exp593_dna.storage import coreweave_client


def download_source(
    record: dict[str, Any], output: Path, client: Any
) -> dict[str, Any]:
    """Check object identity before and after a conditional, atomic download."""
    uri = urlsplit(record["uri"])
    bucket, key = uri.netloc, uri.path.lstrip("/")
    path = output / uri.path.split("/")[-2] / Path(key).name
    path.parent.mkdir(parents=True, exist_ok=True)
    head = client.head_object(Bucket=bucket, Key=key)
    if head["ContentLength"] != record["bytes"] or head["ETag"] != record["etag"]:
        raise ValueError("Original corpus object identity changed: " + record["uri"])
    if not path.exists():
        response = client.get_object(Bucket=bucket, Key=key, IfMatch=record["etag"])
        temporary = path.with_suffix(".part")
        with temporary.open("wb") as stream:
            for chunk in response["Body"].iter_chunks(chunk_size=1024 * 1024):
                stream.write(chunk)
        if temporary.stat().st_size != record["bytes"]:
            raise ValueError("Corpus download size differs")
        temporary.replace(path)
    if path.stat().st_size != record["bytes"]:
        raise ValueError("Cached corpus size differs")
    parquet = pq.ParquetFile(path)
    if (
        parquet.metadata.num_rows != record["rows"]
        or [
            parquet.metadata.row_group(i).num_rows
            for i in range(parquet.num_row_groups)
        ]
        != record["row_groups"]
    ):
        raise ValueError("Corpus Parquet row membership changed")
    after = client.head_object(Bucket=bucket, Key=key)
    if after["ContentLength"] != record["bytes"] or after["ETag"] != record["etag"]:
        raise ValueError("Corpus changed during download")
    sha = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
    result = {
        **record,
        "local_path": str(path),
        "sha256": sha,
        "columns": parquet.schema_arrow.names,
    }
    path.with_suffix(".source.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({"source": record["uri"], "rows": record["rows"], "ready": True}),
        flush=True,
    )
    return result


def main() -> None:
    """Download original train and validation objects; never construct new DNA."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--refresh-keys", action="store_true")
    args = parser.parse_args()
    inventory = json.loads(
        files("exp593_dna").joinpath("assets/corpus.json").read_text()
    )
    client = coreweave_client(refresh_keys=args.refresh_keys)
    records = [r for split in ("train", "validation") for r in inventory[split]]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        downloaded = list(
            pool.map(lambda r: download_source(r, args.output, client), records)
        )
    report = {
        "scope": "Original exp586 objects only; no added sequences",
        "sources": downloaded,
        "train_rows": sum(r["rows"] for r in inventory["train"]),
    }
    if report["train_rows"] != 2_638_656:
        raise ValueError("Original training corpus row count changed")
    (args.output / "sources.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
