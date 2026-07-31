from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from marin_dna.pipelines.vertebrate_projection_dataset.mirror import (
    MirrorObject,
    file_md5,
    s3_object_matches,
    validate_multiz_mirror_contents,
    verify_local_object,
)


def test_verify_local_object_checks_size_and_md5(tmp_path: Path) -> None:
    path = tmp_path / "object.bin"
    path.write_bytes(b"multiz")
    expected = MirrorObject(
        kind="source_metadata",
        chrom="",
        source_url="https://example.test/object.bin",
        s3_uri="s3://example/object.bin",
        byte_size=6,
        md5=hashlib.md5(b"multiz").hexdigest(),
    )
    assert file_md5(path) == expected.md5
    verify_local_object(path, expected)


def test_verify_local_object_fails_on_corruption(tmp_path: Path) -> None:
    path = tmp_path / "object.bin"
    path.write_bytes(b"corrupt")
    expected = MirrorObject(
        kind="source_metadata",
        chrom="",
        source_url="https://example.test/object.bin",
        s3_uri="s3://example/object.bin",
        byte_size=7,
        md5=hashlib.md5(b"correct").hexdigest(),
    )
    with pytest.raises(AssertionError, match="MD5 mismatch"):
        verify_local_object(path, expected)


def test_validate_multiz_mirror_requires_chromosomes_and_trees() -> None:
    def obj(kind: str, chrom: str, name: str) -> MirrorObject:
        return MirrorObject(
            kind=kind,
            chrom=chrom,
            source_url=f"https://example.test/{name}",
            s3_uri=f"s3://example/{name}",
            byte_size=1,
            md5="0" * 32,
        )

    objects = [
        obj("primary_chromosome_maf", "chr1", "chr1.maf.gz"),
        obj("primary_chromosome_maf", "chrX", "chrX.maf.gz"),
        obj("source_metadata", "", "hg38.100way.nh"),
        obj("source_metadata", "", "hg38.100way.scientificNames.nh"),
    ]
    validate_multiz_mirror_contents(objects, ["chr1", "chrX"])
    with pytest.raises(AssertionError, match="mirror chromosomes differ"):
        validate_multiz_mirror_contents(objects, ["chr1", "chr2"])


def test_s3_object_matches_requires_size_and_pinned_md5_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = MirrorObject(
        kind="primary_chromosome_maf",
        chrom="chr18",
        source_url="https://example.test/chr18.maf.gz",
        s3_uri="s3://example/prefix/chr18.maf.gz",
        byte_size=123,
        md5="a" * 32,
    )

    def matching_run(
        *_args: object, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps({"ContentLength": 123, "Metadata": {"md5": "a" * 32}}),
        )

    monkeypatch.setattr(subprocess, "run", matching_run)
    assert s3_object_matches(expected)

    def stale_run(
        *_args: object, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps({"ContentLength": 123, "Metadata": {"md5": "b" * 32}}),
        )

    monkeypatch.setattr(subprocess, "run", stale_run)
    assert not s3_object_matches(expected)
