"""Genome-source inventory helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import polars as pl
from botocore.exceptions import ClientError


def audit_existing_genome_mirror(
    accessions: Iterable[str],
    *,
    bucket: str,
    prefix: str,
    suffix: str,
    s3_client: Any,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return exact-version S3 matches and explicitly missing accessions."""
    normalized_accessions = sorted(set(accessions))
    assert normalized_accessions, "manifest contains no selected accessions"
    normalized_prefix = prefix.strip("/")
    rows: list[dict[str, object]] = []
    missing: list[dict[str, str]] = []
    for accession in normalized_accessions:
        key = f"{normalized_prefix}/{accession}{suffix}"
        try:
            metadata = s3_client.head_object(Bucket=bucket, Key=key)
        except ClientError as error:
            status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if status == 404:
                missing.append(
                    {
                        "accession": accession,
                        "reason": "exact version absent from existing S3 mirror",
                    }
                )
                continue
            raise
        rows.append(
            {
                "accession": accession,
                "download_uri": f"s3://{bucket}/{key}",
                "source_checksum_type": "s3_etag",
                "source_checksum": str(metadata["ETag"]).strip('"'),
                "source_size_bytes": int(metadata["ContentLength"]),
            }
        )
    inventory = pl.DataFrame(
        rows,
        schema={
            "accession": pl.String,
            "download_uri": pl.String,
            "source_checksum_type": pl.String,
            "source_checksum": pl.String,
            "source_size_bytes": pl.Int64,
        },
    )
    missing_inventory = pl.DataFrame(
        missing,
        schema={"accession": pl.String, "reason": pl.String},
    )
    return inventory, missing_inventory
