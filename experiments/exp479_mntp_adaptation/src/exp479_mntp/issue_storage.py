"""Issue-owned S3 publication with immutable object checksums."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

ISSUE_BUCKET = "oa-bolinas"
ISSUE_PREFIX = "issues/479/"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_issue_s3_prefix(uri: str) -> tuple[str, str]:
    """Require a versioned issue-479 destination below the prescribed bucket."""

    parsed = urlparse(uri)
    prefix = parsed.path.lstrip("/").rstrip("/")
    if parsed.scheme != "s3" or parsed.netloc != ISSUE_BUCKET:
        raise ValueError(f"checkpoint destination must use s3://{ISSUE_BUCKET}")
    if not prefix.startswith(ISSUE_PREFIX):
        raise ValueError(
            f"checkpoint destination must stay below s3://{ISSUE_BUCKET}/{ISSUE_PREFIX}"
        )
    relative = prefix.removeprefix(ISSUE_PREFIX)
    if not relative or "/" not in relative or not relative.rsplit("/", 1)[-1].startswith("v"):
        raise ValueError("checkpoint destination must end in a distinct version directory")
    return parsed.netloc, prefix


def upload_issue_artifact(
    local_path: Path,
    *,
    destination_prefix: str,
    relative_path: str,
    client: Any | None = None,
) -> list[dict[str, object]]:
    """Upload one file or directory and return exact size/checksum records."""

    bucket, prefix = validate_issue_s3_prefix(destination_prefix)
    if (
        not relative_path
        or PurePosixPath(relative_path).is_absolute()
        or ".." in PurePosixPath(relative_path).parts
    ):
        raise ValueError("relative S3 artifact path must be nonempty and stay below the prefix")
    if not local_path.exists():
        raise FileNotFoundError(local_path)
    if client is None:
        import boto3

        client = boto3.client("s3")

    if local_path.is_file():
        files = [(local_path, PurePosixPath(local_path.name))]
    else:
        files = [
            (path, PurePosixPath(path.relative_to(local_path).as_posix()))
            for path in sorted(local_path.rglob("*"))
            if path.is_file()
        ]
    if not files:
        raise ValueError(f"artifact path contains no files: {local_path}")

    records: list[dict[str, object]] = []
    for path, suffix in files:
        key = str(PurePosixPath(prefix) / relative_path / suffix)
        size = path.stat().st_size
        checksum = _sha256(path)
        try:
            existing = client.head_object(Bucket=bucket, Key=key)
        except Exception as error:
            response = getattr(error, "response", {})
            code = str(response.get("Error", {}).get("Code", ""))
            if code not in {"404", "NoSuchKey", "NotFound"}:
                raise
        else:
            existing_checksum = existing.get("Metadata", {}).get("sha256")
            existing_size = int(existing.get("ContentLength", -1))
            if existing_checksum != checksum or existing_size != size:
                raise FileExistsError(
                    f"immutable S3 object already exists with different content: s3://{bucket}/{key}"
                )
            records.append(
                {
                    "s3_uri": f"s3://{bucket}/{key}",
                    "size_bytes": size,
                    "sha256": checksum,
                }
            )
            continue
        client.upload_file(
            str(path),
            bucket,
            key,
            ExtraArgs={"Metadata": {"sha256": checksum}},
        )
        uploaded = client.head_object(Bucket=bucket, Key=key)
        if (
            int(uploaded.get("ContentLength", -1)) != size
            or uploaded.get("Metadata", {}).get("sha256") != checksum
        ):
            raise RuntimeError(f"S3 upload verification failed: s3://{bucket}/{key}")
        records.append(
            {
                "s3_uri": f"s3://{bucket}/{key}",
                "size_bytes": size,
                "sha256": checksum,
            }
        )
    return records
