"""Copy immutable S3 prefixes without depending on a system AWS CLI."""

from __future__ import annotations

from typing import Any

from botocore.exceptions import ClientError

from marin_dna_linclust_conservation.staging import parse_s3_uri


def copy_s3_prefix(
    *,
    source_uri: str,
    destination_uri: str,
    s3_client: Any,
    transfer_config: Any = None,
) -> dict[str, object]:
    """Copy every source object to the corresponding destination key."""
    assert source_uri.endswith("/")
    assert destination_uri.endswith("/")
    assert source_uri != destination_uri
    source_bucket, source_prefix = parse_s3_uri(source_uri)
    destination_bucket, destination_prefix = parse_s3_uri(destination_uri)

    copied = 0
    reused = 0
    total_bytes = 0
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=source_bucket, Prefix=source_prefix):
        for source in page.get("Contents", []):
            source_key = str(source["Key"])
            assert source_key.startswith(source_prefix)
            relative_key = source_key.removeprefix(source_prefix)
            if not relative_key:
                continue
            source_size = int(source["Size"])
            destination_key = f"{destination_prefix}{relative_key}"
            try:
                destination = s3_client.head_object(
                    Bucket=destination_bucket,
                    Key=destination_key,
                )
            except ClientError as error:
                if str(error.response.get("Error", {}).get("Code")) not in {
                    "404",
                    "NoSuchKey",
                    "NotFound",
                }:
                    raise
            else:
                if int(destination["ContentLength"]) == source_size:
                    reused += 1
                    total_bytes += source_size
                    continue

            copy_kwargs: dict[str, object] = {
                "CopySource": {"Bucket": source_bucket, "Key": source_key},
                "Bucket": destination_bucket,
                "Key": destination_key,
            }
            if transfer_config is not None:
                copy_kwargs["Config"] = transfer_config
            s3_client.copy(**copy_kwargs)
            destination = s3_client.head_object(
                Bucket=destination_bucket,
                Key=destination_key,
            )
            assert int(destination["ContentLength"]) == source_size
            copied += 1
            total_bytes += source_size

    assert copied + reused > 0, f"empty source prefix: {source_uri}"
    return {
        "source_uri": source_uri,
        "destination_uri": destination_uri,
        "objects_copied": copied,
        "objects_reused": reused,
        "total_bytes": total_bytes,
    }
