"""Read immutable native preparation inputs with complete checksum validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

from filelock import FileLock

from exp593_dna.audit import file_sha256
from exp593_dna.pipeline import DATASET_REVISION
from exp593_dna.storage import coreweave_client

INPUTS_MANIFEST = (
    "s3://marin-us-east-02a/MarinDNA/exp593_augmented_dna/inputs/"
    "prep-v2-46933a19e9bd5c0b1f33be5496f0e0c6705f382859f1f1b847b568d07a65672a/manifest.json"
)
INPUTS_MANIFEST_SHA256 = (
    "33d8d87b7ee9edbba457504632936cfd7eccb00523a74726ed6a08098047f0e3"
)


def verified_download(
    client: Any,
    uri: str,
    path: Path,
    digest: str,
    *,
    size: int | None = None,
    etag: str | None = None,
) -> Path:
    """Use a per-file lock and atomic replacement; incomplete data is never cached."""
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError("Preparation artifacts require an explicit S3 URI")
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock"):
        if (
            path.exists()
            and (size is None or path.stat().st_size == size)
            and file_sha256(path) == digest
        ):
            return path
        options = {"Bucket": parsed.netloc, "Key": parsed.path.lstrip("/")}
        if etag is not None:
            options["IfMatch"] = etag
        response = client.get_object(**options)
        temporary = path.with_suffix(path.suffix + ".part")
        actual = hashlib.sha256()
        count = 0
        with response["Body"] as stream, temporary.open("wb") as output:
            while block := stream.read(8 * 1024 * 1024):
                actual.update(block)
                count += len(block)
                output.write(block)
        if actual.hexdigest() != digest or (size is not None and count != size):
            temporary.unlink(missing_ok=True)
            raise ValueError(f"Preparation artifact identity differs: {uri}")
        temporary.replace(path)
    return path


@dataclass(frozen=True)
class PreparedInputs:
    """Resolve the content-addressed bundle independently of raw DNA storage."""

    root: Path
    manifest: dict[str, Any]
    client: Any

    @classmethod
    def open(
        cls,
        cache_root: Path,
        *,
        uri: str = INPUTS_MANIFEST,
        digest: str = INPUTS_MANIFEST_SHA256,
    ) -> PreparedInputs:
        client = coreweave_client()
        path = verified_download(
            client, uri, cache_root / f"{digest}.manifest.json", digest
        )
        manifest = json.loads(path.read_text())
        if manifest["dataset_revision"] != DATASET_REVISION:
            raise ValueError("Prepared input revision differs")
        fingerprint = manifest["fingerprint"]
        if len(fingerprint) != 64 or any(
            c not in "0123456789abcdef" for c in fingerprint
        ):
            raise ValueError("Invalid prepared input identity")
        prefix = manifest["prefix"]
        if not prefix.startswith(
            "s3://marin-us-east-02a/MarinDNA/exp593_augmented_dna/inputs/"
        ):
            raise ValueError("Prepared input owner differs")
        for name, record in manifest["files"].items():
            relative = PurePosixPath(name)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or record["uri"] != prefix + "/" + name
            ):
                raise ValueError("Unsafe or inconsistent prepared input path")
        return cls(cache_root / fingerprint, manifest, client)

    def path(self, name: str) -> Path:
        """Fetch a known frozen artifact, checking its bytes before use."""
        record = self.manifest["files"][name]
        return verified_download(
            self.client,
            record["uri"],
            self.root / name,
            record["sha256"],
            size=record["bytes"],
            etag=record["etag"],
        )

    def names(self, prefix: str, suffix: str = "") -> list[str]:
        """Return deterministic object membership from the pinned manifest."""
        return sorted(
            n
            for n in self.manifest["files"]
            if n.startswith(prefix) and n.endswith(suffix)
        )
