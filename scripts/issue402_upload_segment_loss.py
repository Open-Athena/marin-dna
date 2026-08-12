#!/usr/bin/env python3
"""Upload issue #402 segment-loss artifacts with the worker's AWS SDK."""

from __future__ import annotations

import argparse
from pathlib import Path

import boto3

FILENAMES = (
    "manifest.json",
    "validation_position_loss.parquet",
    "validation_document_segment_loss.parquet",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-uri", required=True)
    return parser.parse_args()


def split_s3_uri(uri: str) -> tuple[str, str]:
    assert uri.startswith("s3://"), uri
    bucket, separator, key = uri.removeprefix("s3://").partition("/")
    assert bucket and separator and key, uri
    return bucket, key.rstrip("/")


def main() -> None:
    args = parse_args()
    bucket, prefix = split_s3_uri(args.output_uri)
    client = boto3.client("s3", region_name="us-east-2")
    for filename in FILENAMES:
        source = args.input_dir / filename
        assert source.is_file() and source.stat().st_size > 0, source
        key = f"{prefix}/{filename}"
        client.upload_file(str(source), bucket, key)
        response = client.head_object(Bucket=bucket, Key=key)
        assert response["ContentLength"] == source.stat().st_size
        print(f"uploaded s3://{bucket}/{key} ({source.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
