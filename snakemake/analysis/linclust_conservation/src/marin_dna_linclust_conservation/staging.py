"""Stage immutable 2bit inputs into the pipeline-owned S3 namespace."""

from __future__ import annotations

import hashlib
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split an S3 URI into bucket and key."""
    assert uri.startswith("s3://"), f"not an S3 URI: {uri}"
    bucket, separator, key = uri.removeprefix("s3://").partition("/")
    assert bucket and separator and key, f"incomplete S3 URI: {uri}"
    return bucket, key


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_staged_genome(
    *,
    receipt: dict[str, object],
    destination_path: str | Path,
    s3_client: Any,
) -> tuple[str, int]:
    """Conditionally download and verify the exact S3 object in a staging receipt."""
    bucket, key = parse_s3_uri(str(receipt["destination_uri"]))
    expected_etag = str(receipt["destination_etag"])
    expected_size = int(receipt["destination_size_bytes"])
    response = s3_client.get_object(
        Bucket=bucket,
        Key=key,
        IfMatch=f'"{expected_etag}"',
    )
    assert str(response["ETag"]).strip('"') == expected_etag
    assert int(response["ContentLength"]) == expected_size

    digest = hashlib.sha256()
    observed_size = 0
    body = response["Body"]
    try:
        with Path(destination_path).open("wb") as output:
            for block in iter(lambda: body.read(8 * 1024 * 1024), b""):
                output.write(block)
                digest.update(block)
                observed_size += len(block)
    finally:
        body.close()
    assert observed_size == expected_size
    observed_sha256 = digest.hexdigest()
    expected_sha256 = receipt.get("sequence_sha256")
    if expected_sha256 is not None:
        assert observed_sha256 == expected_sha256
    return observed_sha256, observed_size


def copy_reused_genome(
    *,
    accession: str,
    source_uri: str,
    destination_uri: str,
    source_etag: str,
    source_size_bytes: int,
    s3_client: Any,
) -> dict[str, object]:
    """Copy an exact source object after an ETag-guarded metadata check."""
    source_bucket, source_key = parse_s3_uri(source_uri)
    destination_bucket, destination_key = parse_s3_uri(destination_uri)
    source = s3_client.head_object(Bucket=source_bucket, Key=source_key)
    observed_etag = str(source["ETag"]).strip('"')
    assert observed_etag == source_etag
    assert int(source["ContentLength"]) == source_size_bytes
    started = time.monotonic()
    response = s3_client.copy_object(
        Bucket=destination_bucket,
        Key=destination_key,
        CopySource={"Bucket": source_bucket, "Key": source_key},
        CopySourceIfMatch=f'"{source_etag}"',
        MetadataDirective="REPLACE",
        Metadata={
            "assembly-accession": accession,
            "source-etag": source_etag,
            "source-uri": source_uri,
        },
    )
    destination = s3_client.head_object(
        Bucket=destination_bucket,
        Key=destination_key,
    )
    assert int(destination["ContentLength"]) == source_size_bytes
    return {
        "accession": accession,
        "source_kind": "existing_s3_2bit",
        "source_uri": source_uri,
        "source_checksum_type": "s3_etag",
        "source_checksum": source_etag,
        "source_size_bytes": source_size_bytes,
        "destination_uri": destination_uri,
        "destination_etag": str(destination["ETag"]).strip('"'),
        "destination_size_bytes": int(destination["ContentLength"]),
        "copy_request_id": response.get("ResponseMetadata", {}).get("RequestId", ""),
        "wall_seconds": time.monotonic() - started,
    }


def download_convert_upload_genome(
    *,
    accession: str,
    destination_uri: str,
    s3_client: Any,
) -> dict[str, object]:
    """Download one NCBI assembly, convert it to 2bit, and upload it."""
    destination_bucket, destination_key = parse_s3_uri(destination_uri)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=f"linclust_{accession}_") as directory:
        temporary = Path(directory)
        archive = temporary / "ncbi_dataset.zip"
        subprocess.run(
            [
                "datasets",
                "download",
                "genome",
                "accession",
                accession,
                "--include",
                "genome,seq-report",
                "--filename",
                str(archive),
                "--no-progressbar",
            ],
            check=True,
        )
        archive_sha256 = sha256_file(archive)
        with zipfile.ZipFile(archive) as package:
            package.extractall(temporary / "package")
        fasta_files = list((temporary / "package").glob("ncbi_dataset/data/**/*.fna"))
        assert len(fasta_files) == 1, fasta_files
        fasta = fasta_files[0]
        fasta_sha256 = sha256_file(fasta)
        twobit = temporary / f"{accession}.2bit"
        subprocess.run(["faToTwoBit", str(fasta), str(twobit)], check=True)
        sequence_sha256 = sha256_file(twobit)
        s3_client.upload_file(
            str(twobit),
            destination_bucket,
            destination_key,
            ExtraArgs={
                "Metadata": {
                    "assembly-accession": accession,
                    "sequence-sha256": sequence_sha256,
                    "source-fasta-sha256": fasta_sha256,
                }
            },
        )
        destination = s3_client.head_object(
            Bucket=destination_bucket,
            Key=destination_key,
        )
        assert int(destination["ContentLength"]) == twobit.stat().st_size
        return {
            "accession": accession,
            "source_kind": "ncbi_datasets_genome",
            "source_uri": f"ncbi-datasets://genome/accession/{accession}",
            "source_checksum_type": "sha256",
            "source_checksum": archive_sha256,
            "source_size_bytes": archive.stat().st_size,
            "source_fasta_sha256": fasta_sha256,
            "destination_uri": destination_uri,
            "destination_etag": str(destination["ETag"]).strip('"'),
            "destination_size_bytes": int(destination["ContentLength"]),
            "sequence_sha256": sequence_sha256,
            "wall_seconds": time.monotonic() - started,
        }
