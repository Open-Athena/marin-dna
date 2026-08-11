"""Immutable UCSC 2bit input manifests and verified local staging."""

from __future__ import annotations

import hashlib
import re
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import polars as pl

S3_ETAG_PART_SIZE = 8 * 1024 * 1024


@dataclass(frozen=True)
class TwoBitObject:
    name: str
    source_url: str
    checksum_source_url: str
    byte_size: int
    s3_etag: str
    s3_etag_part_size: int
    published_md5: str | None


def read_twobit_manifest(path: str | Path) -> dict[str, TwoBitObject]:
    """Read and validate the committed human + MultiZ 2bit manifest."""
    frame = pl.read_csv(path, separator="\t", null_values="-")
    required = {
        "name",
        "source_url",
        "checksum_source_url",
        "byte_size",
        "s3_etag",
        "s3_etag_part_size",
        "published_md5",
    }
    missing = required - set(frame.columns)
    assert not missing, f"2bit manifest missing columns: {sorted(missing)}"
    assert frame.height > 0
    assert frame["name"].n_unique() == frame.height
    assert frame["source_url"].n_unique() == frame.height
    assert frame["checksum_source_url"].n_unique() == frame.height
    assert (frame["byte_size"] > 0).all()
    assert (frame["s3_etag_part_size"] > 0).all()
    assert frame["s3_etag"].str.contains(r"^[0-9a-f]{32}(?:-[1-9][0-9]*)?$").all()
    assert frame["source_url"].str.starts_with("https://").all()
    assert frame["checksum_source_url"].str.starts_with("https://").all()
    md5s = frame["published_md5"].drop_nulls()
    assert md5s.str.contains(r"^[0-9a-f]{32}$").all()
    return {
        str(row["name"]): TwoBitObject(
            name=str(row["name"]),
            source_url=str(row["source_url"]),
            checksum_source_url=str(row["checksum_source_url"]),
            byte_size=int(row["byte_size"]),
            s3_etag=str(row["s3_etag"]),
            s3_etag_part_size=int(row["s3_etag_part_size"]),
            published_md5=(
                None if row["published_md5"] is None else str(row["published_md5"])
            ),
        )
        for row in frame.to_dicts()
    }


def validate_twobit_manifest(
    objects: dict[str, TwoBitObject], expected_names: list[str]
) -> None:
    assert len(expected_names) == len(set(expected_names))
    assert set(objects) == set(expected_names), {
        "missing": sorted(set(expected_names) - set(objects)),
        "unexpected": sorted(set(objects) - set(expected_names)),
    }


def file_md5(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_s3_etag(path: str | Path, *, part_size: int) -> str:
    """Compute the ETag produced by a fixed-size S3 multipart upload."""
    assert part_size >= 5 * 1024 * 1024
    part_digests: list[bytes] = []
    with Path(path).open("rb") as handle:
        while chunk := handle.read(part_size):
            part_digests.append(hashlib.md5(chunk).digest())
    assert part_digests, f"empty file: {path}"
    if len(part_digests) == 1:
        return part_digests[0].hex()
    return f"{hashlib.md5(b''.join(part_digests)).hexdigest()}-{len(part_digests)}"


def verify_twobit(path: str | Path, expected: TwoBitObject) -> None:
    local_path = Path(path)
    assert local_path.is_file(), f"missing 2bit: {local_path}"
    observed_size = local_path.stat().st_size
    assert observed_size == expected.byte_size, (
        f"2bit size mismatch for {expected.name}: "
        f"{observed_size} != {expected.byte_size}"
    )
    if expected.published_md5 is not None:
        assert expected.s3_etag == expected.published_md5
        observed_md5 = file_md5(local_path)
        assert observed_md5 == expected.published_md5, (
            f"2bit published MD5 mismatch for {expected.name}: "
            f"{observed_md5} != {expected.published_md5}"
        )
    else:
        observed_etag = file_s3_etag(local_path, part_size=expected.s3_etag_part_size)
        assert observed_etag == expected.s3_etag, (
            f"2bit ETag mismatch for {expected.name}: "
            f"{observed_etag} != {expected.s3_etag}"
        )


def stage_twobit(expected: TwoBitObject, destination: str | Path) -> None:
    """Download, verify, and atomically install one pinned 2bit."""
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists():
        try:
            verify_twobit(destination_path, expected)
            return
        except AssertionError:
            destination_path.unlink()

    partial = destination_path.with_name(f".{destination_path.name}.partial")
    partial.unlink(missing_ok=True)
    try:
        subprocess.run(
            [
                "wget",
                "-q",
                "--retry-connrefused",
                "--waitretry=5",
                "--timeout=60",
                "--tries=20",
                "-O",
                str(partial),
                expected.source_url,
            ],
            check=True,
        )
        verify_twobit(partial, expected)
        partial.replace(destination_path)
    finally:
        partial.unlink(missing_ok=True)


def _published_md5_or_none(checksum_url: str, filename: str) -> str | None:
    request = urllib.request.Request(
        checksum_url, headers={"User-Agent": "marin-dna/417"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            checksum_text = response.read().decode()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise
    matches = [
        match.group(1)
        for line in checksum_text.splitlines()
        if (
            match := re.fullmatch(
                rf"([0-9a-f]{{32}})\s+\*?{re.escape(filename)}", line.strip()
            )
        )
    ]
    assert len(matches) == 1, (checksum_url, filename, matches)
    return matches[0]


def _head_metadata(url: str) -> tuple[int, str | None]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "marin-dna/417"},
        method="HEAD",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        content_length = response.headers.get("Content-Length")
        etag = response.headers.get("ETag")
    assert content_length is not None, f"missing Content-Length: {url}"
    size = int(content_length)
    assert size > 0
    return size, None if etag is None else etag.strip('"')


def build_twobit_manifest(
    species_manifest_path: str | Path, output_path: str | Path
) -> pl.DataFrame:
    """Resolve UCSC-published sizes, S3 ETags, and optional MD5s for v1 inputs."""
    species = pl.read_csv(species_manifest_path, separator="\t")
    names = (
        species.filter((pl.col("backend") == "ucsc_multiz100way") & pl.col("selected"))[
            "alignment_name"
        ]
        .sort()
        .to_list()
    )
    assert len(names) == len(set(names)) and names
    rows: list[dict[str, object]] = []
    for name in ["hg38", *names]:
        source_url = (
            f"https://hgdownload.soe.ucsc.edu/goldenPath/{name}/bigZips/{name}.2bit"
            if name == "hg38"
            else f"https://hgdownload.soe.ucsc.edu/gbdb/{name}/{name}.2bit"
        )
        checksum_index_url = (
            f"https://hgdownload.soe.ucsc.edu/goldenPath/{name}/bigZips/md5sum.txt"
        )
        source_size, _source_etag = _head_metadata(source_url)
        published_md5 = _published_md5_or_none(checksum_index_url, f"{name}.2bit")
        if published_md5 is not None:
            checksum_source_url = checksum_index_url
            content_etag = published_md5
            etag_part_size = S3_ETAG_PART_SIZE
        else:
            checksum_source_url = (
                f"https://genome-browser.s3.amazonaws.com/gbdb/{name}/{name}.2bit"
            )
            mirror_size, mirror_etag = _head_metadata(checksum_source_url)
            assert source_size == mirror_size, (name, source_size, mirror_size)
            assert mirror_etag is not None
            expected_parts = (source_size + S3_ETAG_PART_SIZE - 1) // S3_ETAG_PART_SIZE
            assert mirror_etag.endswith(f"-{expected_parts}"), (name, mirror_etag)
            content_etag = mirror_etag
            etag_part_size = S3_ETAG_PART_SIZE
        rows.append(
            {
                "name": name,
                "source_url": source_url,
                "checksum_source_url": checksum_source_url,
                "byte_size": source_size,
                "s3_etag": content_etag,
                "s3_etag_part_size": etag_part_size,
                "published_md5": published_md5,
            }
        )
    frame = pl.DataFrame(rows).sort("name")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(output, separator="\t", null_value="-")
    return frame
