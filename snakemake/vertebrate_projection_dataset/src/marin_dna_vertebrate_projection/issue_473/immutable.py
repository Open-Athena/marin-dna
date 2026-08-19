"""Checksum-verified immutable S3 inputs for issue #473."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import polars as pl


@dataclass(frozen=True)
class ImmutableS3Object:
    """An S3 object pinned by path, byte size, and full-object checksum."""

    name: str
    s3_uri: str
    byte_size: int
    checksum_crc64nvme: str


def read_immutable_sources(path: str | Path) -> dict[str, ImmutableS3Object]:
    """Read the small, committed #473 immutable-source manifest."""
    frame = pl.read_csv(path, separator="\t")
    required = {"name", "s3_uri", "byte_size", "checksum_crc64nvme"}
    missing = required - set(frame.columns)
    assert not missing, f"immutable source manifest missing columns: {sorted(missing)}"
    assert frame.height > 0
    assert frame["name"].n_unique() == frame.height
    assert frame["s3_uri"].n_unique() == frame.height
    assert (frame["byte_size"] > 0).all()
    assert frame["s3_uri"].str.starts_with("s3://").all()
    assert frame["checksum_crc64nvme"].str.contains(r"^[A-Za-z0-9+/]+={0,2}$").all()
    return {
        str(row["name"]): ImmutableS3Object(
            name=str(row["name"]),
            s3_uri=str(row["s3_uri"]),
            byte_size=int(row["byte_size"]),
            checksum_crc64nvme=str(row["checksum_crc64nvme"]),
        )
        for row in frame.to_dicts()
    }


def _bucket_key(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    assert parsed.scheme == "s3" and parsed.netloc
    key = parsed.path.lstrip("/")
    assert key
    return parsed.netloc, key


def _sha256(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _head_object(uri: str) -> dict[str, object]:
    bucket, key = _bucket_key(uri)
    result = subprocess.run(
        [
            "aws",
            "s3api",
            "head-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--checksum-mode",
            "ENABLED",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert isinstance(payload, dict)
    return payload


def _verify_remote_metadata(expected: ImmutableS3Object) -> None:
    payload = _head_object(expected.s3_uri)
    assert int(payload["ContentLength"]) == expected.byte_size, (
        f"immutable S3 size changed for {expected.name}: "
        f"{payload['ContentLength']} != {expected.byte_size}"
    )
    assert payload.get("ChecksumType") == "FULL_OBJECT"
    assert payload.get("ChecksumCRC64NVME") == expected.checksum_crc64nvme, (
        f"immutable S3 checksum changed for {expected.name}: "
        f"{payload.get('ChecksumCRC64NVME')} != {expected.checksum_crc64nvme}"
    )


def stage_immutable_s3_object(
    expected: ImmutableS3Object,
    destination: str | Path,
    receipt_path: str | Path,
) -> None:
    """Download one pinned object atomically and preserve its local SHA-256."""
    destination_path = Path(destination)
    receipt = Path(receipt_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    _verify_remote_metadata(expected)

    if destination_path.is_file() and receipt.is_file():
        recorded = json.loads(receipt.read_text())
        if (
            destination_path.stat().st_size == expected.byte_size
            and recorded.get("name") == expected.name
            and recorded.get("s3_uri") == expected.s3_uri
            and recorded.get("byte_size") == expected.byte_size
            and recorded.get("checksum_crc64nvme") == expected.checksum_crc64nvme
            and recorded.get("sha256") == _sha256(destination_path)
        ):
            return

    partial = destination_path.with_name(f".{destination_path.name}.partial")
    partial.unlink(missing_ok=True)
    bucket, key = _bucket_key(expected.s3_uri)
    try:
        result = subprocess.run(
            [
                "aws",
                "s3api",
                "get-object",
                "--bucket",
                bucket,
                "--key",
                key,
                "--checksum-mode",
                "ENABLED",
                str(partial),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        response = json.loads(result.stdout)
        assert response.get("ChecksumCRC64NVME") == expected.checksum_crc64nvme
        assert partial.stat().st_size == expected.byte_size
        local_sha256 = _sha256(partial)
        partial.replace(destination_path)
        receipt_tmp = receipt.with_name(f".{receipt.name}.partial")
        receipt_tmp.write_text(
            json.dumps(
                {
                    "name": expected.name,
                    "s3_uri": expected.s3_uri,
                    "byte_size": expected.byte_size,
                    "checksum_crc64nvme": expected.checksum_crc64nvme,
                    "sha256": local_sha256,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        receipt_tmp.replace(receipt)
    finally:
        partial.unlink(missing_ok=True)


def read_artifact_inventory(path: str | Path) -> dict[str, int]:
    """Read #417's path/byte inventory and reject duplicate or unsafe paths."""
    frame = pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        new_columns=["relative_path", "byte_size"],
        schema_overrides={"relative_path": pl.String, "byte_size": pl.Int64},
    )
    assert frame.height > 0
    assert frame["relative_path"].n_unique() == frame.height
    assert (frame["byte_size"] > 0).all()
    paths = frame["relative_path"].to_list()
    assert all(
        path and not path.startswith("/") and ".." not in Path(path).parts
        for path in paths
    )
    return {
        str(row["relative_path"]): int(row["byte_size"]) for row in frame.to_dicts()
    }


def stage_inventory_object(
    *,
    source_prefix: str,
    inventory_path: str | Path,
    relative_path: str,
    destination: str | Path,
    receipt_path: str | Path,
) -> None:
    """Stage one object named by a pinned inventory and record its live checksum."""
    inventory = read_artifact_inventory(inventory_path)
    assert relative_path in inventory, (
        f"artifact absent from inventory: {relative_path}"
    )
    uri = f"{source_prefix.rstrip('/')}/{relative_path}"
    metadata = _head_object(uri)
    checksum = metadata.get("ChecksumCRC64NVME")
    assert metadata.get("ChecksumType") == "FULL_OBJECT"
    assert isinstance(checksum, str) and checksum
    expected = ImmutableS3Object(
        name=relative_path.replace("/", "__"),
        s3_uri=uri,
        byte_size=inventory[relative_path],
        checksum_crc64nvme=checksum,
    )
    stage_immutable_s3_object(expected, destination, receipt_path)
