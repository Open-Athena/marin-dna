from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest
from marin_dna_vertebrate_projection.sequence_sources import (
    TwoBitObject,
    file_s3_etag,
    read_twobit_manifest,
    stage_twobit,
    validate_twobit_manifest,
    verify_twobit,
)


def _object(payload: bytes) -> TwoBitObject:
    digest = hashlib.md5(payload).hexdigest()
    return TwoBitObject(
        name="galGal4",
        source_url="https://example.test/galGal4.2bit",
        checksum_source_url="https://example.test/md5sum.txt",
        byte_size=len(payload),
        s3_etag=digest,
        s3_etag_part_size=5 * 1024 * 1024,
        published_md5=digest,
    )


def test_read_twobit_manifest_requires_exact_names(tmp_path: Path) -> None:
    payload = b"two-bit"
    expected = _object(payload)
    path = tmp_path / "manifest.tsv"
    path.write_text(
        "name\tsource_url\tchecksum_source_url\tbyte_size\ts3_etag\t"
        "s3_etag_part_size\tpublished_md5\n"
        f"{expected.name}\t{expected.source_url}\t{expected.checksum_source_url}\t"
        f"{expected.byte_size}\t{expected.s3_etag}\t"
        f"{expected.s3_etag_part_size}\t{expected.published_md5}\n"
    )
    objects = read_twobit_manifest(path)
    assert objects == {"galGal4": expected}
    validate_twobit_manifest(objects, ["galGal4"])
    with pytest.raises(AssertionError, match="missing"):
        validate_twobit_manifest(objects, ["galGal4", "hg38"])


def test_file_s3_etag_supports_multipart(tmp_path: Path) -> None:
    path = tmp_path / "object"
    part_size = 5 * 1024 * 1024
    path.write_bytes(b"a" * part_size + b"tail")
    parts = [hashlib.md5(b"a" * part_size).digest(), hashlib.md5(b"tail").digest()]
    expected = f"{hashlib.md5(b''.join(parts)).hexdigest()}-2"
    assert file_s3_etag(path, part_size=part_size) == expected


def test_stage_twobit_installs_only_verified_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"verified-two-bit"
    expected = _object(payload)
    destination = tmp_path / "galGal4.2bit"

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        partial = Path(command[command.index("-O") + 1])
        assert partial != destination
        assert partial.name == ".galGal4.2bit.partial"
        partial.write_bytes(payload)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    stage_twobit(expected, destination)
    assert destination.read_bytes() == payload
    verify_twobit(destination, expected)
    assert not (tmp_path / ".galGal4.2bit.partial").exists()


def test_stage_twobit_does_not_install_corrupt_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = _object(b"correct")
    destination = tmp_path / "galGal4.2bit"

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        Path(command[command.index("-O") + 1]).write_bytes(b"corrupt")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(AssertionError, match="mismatch"):
        stage_twobit(expected, destination)
    assert not destination.exists()
    assert not (tmp_path / ".galGal4.2bit.partial").exists()
