import hashlib
import io
import subprocess
from pathlib import Path

import pytest
from marin_dna_linclust_conservation.staging import (
    _run_with_retries,
    copy_reused_genome,
    download_staged_genome,
    parse_s3_uri,
)


def test_run_with_retries_cleans_partial_download_and_recovers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    partial = tmp_path / "partial.zip"
    attempts = 0
    sleeps: list[float] = []

    def fake_run(command: list[str], *, check: bool) -> None:
        nonlocal attempts
        assert command == ["datasets", "download"]
        assert check
        attempts += 1
        if attempts < 3:
            partial.write_bytes(b"partial")
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("time.sleep", sleeps.append)

    completed_attempt = _run_with_retries(
        ["datasets", "download"],
        attempts=4,
        initial_delay_seconds=0.5,
        cleanup_paths=(partial,),
    )

    assert completed_attempt == 3
    assert attempts == 3
    assert sleeps == [0.5, 1.0]
    assert not partial.exists()


def test_run_with_retries_reraises_after_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    def always_fail(command: list[str], *, check: bool) -> None:
        nonlocal attempts
        attempts += 1
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", always_fail)
    monkeypatch.setattr("time.sleep", lambda _: None)

    with pytest.raises(subprocess.CalledProcessError):
        _run_with_retries(
            ["datasets", "download"],
            attempts=3,
            initial_delay_seconds=0,
        )
    assert attempts == 3


class FakeS3Client:
    def __init__(self) -> None:
        self.copied = False

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        if Bucket == "source":
            return {"ETag": '"old-etag"', "ContentLength": 42}
        assert self.copied
        return {"ETag": '"new-etag"', "ContentLength": 42}

    def copy_object(self, **kwargs: object) -> dict[str, object]:
        assert kwargs["CopySourceIfMatch"] == '"old-etag"'
        self.copied = True
        return {"ResponseMetadata": {"RequestId": "request"}}


def test_parse_s3_uri() -> None:
    assert parse_s3_uri("s3://bucket/path/object") == ("bucket", "path/object")


def test_copy_reused_genome_checks_and_records_source_identity() -> None:
    receipt = copy_reused_genome(
        accession="GCF_1.1",
        source_uri="s3://source/old/GCF_1.1.2bit",
        destination_uri="s3://destination/new/GCF_1.1.2bit",
        source_etag="old-etag",
        source_size_bytes=42,
        s3_client=FakeS3Client(),
    )
    assert receipt["source_checksum"] == "old-etag"
    assert receipt["destination_etag"] == "new-etag"
    assert receipt["destination_size_bytes"] == 42


class FakeDownloadS3Client:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def get_object(self, **kwargs: object) -> dict[str, object]:
        assert kwargs["Bucket"] == "destination"
        assert kwargs["Key"] == "new/GCF_1.1.2bit"
        assert kwargs["IfMatch"] == '"new-etag"'
        return {
            "Body": io.BytesIO(self.content),
            "ContentLength": len(self.content),
            "ETag": '"new-etag"',
        }


def test_download_staged_genome_conditionally_gets_and_hashes_receipt(
    tmp_path: Path,
) -> None:
    content = b"two-bit-content"
    sha256 = hashlib.sha256(content).hexdigest()
    destination = tmp_path / "genome.2bit"
    observed = download_staged_genome(
        receipt={
            "destination_uri": "s3://destination/new/GCF_1.1.2bit",
            "destination_etag": "new-etag",
            "destination_size_bytes": len(content),
            "sequence_sha256": sha256,
        },
        destination_path=destination,
        s3_client=FakeDownloadS3Client(content),
    )
    assert observed == (sha256, len(content))
    assert destination.read_bytes() == content


def test_download_staged_genome_rejects_fresh_source_hash_mismatch(
    tmp_path: Path,
) -> None:
    content = b"two-bit-content"
    with pytest.raises(AssertionError):
        download_staged_genome(
            receipt={
                "destination_uri": "s3://destination/new/GCF_1.1.2bit",
                "destination_etag": "new-etag",
                "destination_size_bytes": len(content),
                "sequence_sha256": "0" * 64,
            },
            destination_path=tmp_path / "genome.2bit",
            s3_client=FakeDownloadS3Client(content),
        )
