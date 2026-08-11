"""Checksum-verified UCSC-to-S3 mirroring and S3-to-NVMe staging."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import polars as pl


@dataclass(frozen=True)
class MirrorObject:
    kind: str
    chrom: str
    source_url: str
    s3_uri: str
    byte_size: int
    md5: str


def read_mirror_manifest(path: str | Path) -> list[MirrorObject]:
    """Read and validate the immutable source-object manifest."""
    frame = pl.read_csv(path, separator="\t")
    required = {"kind", "chrom", "source_url", "s3_uri", "byte_size", "md5"}
    missing = required - set(frame.columns)
    assert not missing, f"mirror manifest missing columns: {sorted(missing)}"
    assert frame.height > 0
    assert frame["source_url"].n_unique() == frame.height
    assert frame["s3_uri"].n_unique() == frame.height
    assert (frame["byte_size"] > 0).all()
    assert frame["md5"].str.contains(r"^[0-9a-f]{32}$").all()
    assert frame["source_url"].str.starts_with("https://").all()
    assert frame["s3_uri"].str.starts_with("s3://").all()
    return [
        MirrorObject(
            kind=str(row["kind"]),
            chrom=str(row["chrom"] or ""),
            source_url=str(row["source_url"]),
            s3_uri=str(row["s3_uri"]),
            byte_size=int(row["byte_size"]),
            md5=str(row["md5"]),
        )
        for row in frame.to_dicts()
    ]


def validate_multiz_mirror_contents(
    objects: list[MirrorObject], required_chroms: list[str]
) -> None:
    """Require one MAF per configured primary chromosome and both tree files."""
    assert objects
    assert len(required_chroms) == len(set(required_chroms))
    primary = [obj for obj in objects if obj.kind == "primary_chromosome_maf"]
    observed = [obj.chrom for obj in primary]
    assert len(observed) == len(set(observed)), "duplicate chromosome MAF objects"
    assert set(observed) == set(required_chroms), (
        f"mirror chromosomes differ: observed={sorted(observed)}, "
        f"required={sorted(required_chroms)}"
    )
    metadata_names = {
        Path(obj.source_url).name for obj in objects if obj.kind == "source_metadata"
    }
    assert "hg38.100way.nh" in metadata_names, "mirror is missing alignment-name tree"
    assert "hg38.100way.scientificNames.nh" in metadata_names, (
        "mirror is missing scientific-name tree"
    )


def file_md5(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a file without loading multi-gigabyte MAFs into memory."""
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_local_object(path: str | Path, expected: MirrorObject) -> None:
    """Fail on missing, truncated, or content-mismatched staged objects."""
    local_path = Path(path)
    assert local_path.is_file(), f"missing staged object: {local_path}"
    observed_size = local_path.stat().st_size
    assert observed_size == expected.byte_size, (
        f"size mismatch for {local_path}: {observed_size} != {expected.byte_size}"
    )
    observed_md5 = file_md5(local_path)
    assert observed_md5 == expected.md5, (
        f"MD5 mismatch for {local_path}: {observed_md5} != {expected.md5}"
    )


def _s3_bucket_key(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    assert parsed.scheme == "s3" and parsed.netloc
    key = parsed.path.lstrip("/")
    assert key
    return parsed.netloc, key


def s3_object_size(uri: str) -> int:
    """Read the authoritative S3 object size without downloading it."""
    bucket, key = _s3_bucket_key(uri)
    result = subprocess.run(
        [
            "aws",
            "s3api",
            "head-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--query",
            "ContentLength",
            "--output",
            "text",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    size = int(result.stdout.strip())
    assert size > 0, f"invalid S3 object size for {uri}: {size}"
    return size


def verify_hal_object(
    path: str | Path,
    *,
    expected_size: int,
    required_genome: str = "Homo_sapiens",
) -> None:
    """Fail on a truncated or unreadable HAL before projection begins."""
    hal_path = Path(path)
    assert hal_path.is_file(), f"missing staged HAL: {hal_path}"
    observed_size = hal_path.stat().st_size
    assert observed_size == expected_size, (
        f"HAL size mismatch for {hal_path}: {observed_size} != {expected_size}"
    )
    result = subprocess.run(
        ["halStats", "--genomes", str(hal_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    genomes = result.stdout.split()
    assert required_genome in genomes, (
        f"staged HAL is missing required genome {required_genome!r}: {genomes[:10]}"
    )


def stage_hal_object(source_uri: str, destination: str | Path) -> None:
    """Copy a HAL from S3, then verify its byte size and readable genome index."""
    expected_size = s3_object_size(source_uri)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "aws",
            "s3",
            "cp",
            source_uri,
            str(destination_path),
            "--no-progress",
        ],
        check=True,
    )
    verify_hal_object(destination_path, expected_size=expected_size)


def s3_object_matches(expected: MirrorObject) -> bool:
    """Check resumable bootstrap metadata without downloading the object."""
    bucket, key = _s3_bucket_key(expected.s3_uri)
    result = subprocess.run(
        [
            "aws",
            "s3api",
            "head-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--output",
            "json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False
    payload = json.loads(result.stdout)
    metadata = payload.get("Metadata", {})
    return (
        int(payload.get("ContentLength", -1)) == expected.byte_size
        and metadata.get("md5") == expected.md5
    )


def stage_s3_object(expected: MirrorObject, destination: str | Path) -> None:
    """Copy one mirrored object from S3 and verify it before use."""
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "aws",
            "s3",
            "cp",
            expected.s3_uri,
            str(destination_path),
            "--no-progress",
        ],
        check=True,
    )
    verify_local_object(destination_path, expected)


def mirror_source_object(expected: MirrorObject) -> None:
    """Bootstrap one UCSC source object into its immutable S3 destination."""
    if s3_object_matches(expected):
        return
    with tempfile.TemporaryDirectory(prefix="marin-dna-multiz-") as temp_dir:
        local_path = Path(temp_dir) / Path(expected.source_url).name
        request = urllib.request.Request(
            expected.source_url, headers={"User-Agent": "marin-dna/417"}
        )
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            local_path.open("wb") as output,
        ):
            shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
        verify_local_object(local_path, expected)
        subprocess.run(
            [
                "aws",
                "s3",
                "cp",
                str(local_path),
                expected.s3_uri,
                "--metadata",
                f"md5={expected.md5}",
                "--no-progress",
            ],
            check=True,
        )
        assert s3_object_matches(expected), (
            f"uploaded S3 object failed metadata verification: {expected.s3_uri}"
        )
