from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from exp479_mntp.issue_storage import (
    download_verified_issue_object,
    upload_issue_artifact,
    validate_issue_s3_prefix,
)


class _MissingObject(Exception):
    def __init__(self) -> None:
        self.response = {"Error": {"Code": "404"}}


class _FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], dict[str, Any]] = {}
        self.uploads = 0

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        try:
            payload = self.objects[(Bucket, Key)]
        except KeyError as error:
            raise _MissingObject from error
        return {
            "ContentLength": len(payload["body"]),
            "Metadata": payload["metadata"],
        }

    def upload_file(
        self,
        filename: str,
        bucket: str,
        key: str,
        *,
        ExtraArgs: dict[str, Any],
    ) -> None:
        self.uploads += 1
        self.objects[(bucket, key)] = {
            "body": Path(filename).read_bytes(),
            "metadata": ExtraArgs["Metadata"],
        }

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        Path(filename).write_bytes(self.objects[(bucket, key)]["body"])


@pytest.mark.parametrize(
    "uri",
    [
        "https://oa-bolinas/issues/479/run/v1",
        "s3://other-bucket/issues/479/run/v1",
        "s3://oa-bolinas/issues/480/run/v1",
        "s3://oa-bolinas/issues/479/run",
    ],
)
def test_validate_issue_s3_prefix_rejects_wrong_or_unversioned_paths(uri: str) -> None:
    with pytest.raises(ValueError):
        validate_issue_s3_prefix(uri)


def test_upload_issue_artifact_is_checksum_verified_and_retry_safe(tmp_path: Path) -> None:
    artifact = tmp_path / "adapter"
    artifact.mkdir()
    config = artifact / "adapter_config.json"
    weights = artifact / "adapter_model.safetensors"
    config.write_bytes(b'{"rank": 16}\n')
    weights.write_bytes(b"adapter-weights")
    client = _FakeS3()

    records = upload_issue_artifact(
        artifact,
        destination_prefix="s3://oa-bolinas/issues/479/bico-lora-standard-rate/v1",
        relative_path="adapters/step-0000",
        client=client,
    )

    assert client.uploads == 2
    assert [record["s3_uri"] for record in records] == [
        (
            "s3://oa-bolinas/issues/479/bico-lora-standard-rate/v1/"
            "adapters/step-0000/adapter_config.json"
        ),
        (
            "s3://oa-bolinas/issues/479/bico-lora-standard-rate/v1/"
            "adapters/step-0000/adapter_model.safetensors"
        ),
    ]
    assert records[0]["sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert records[1]["size_bytes"] == len(weights.read_bytes())

    repeated = upload_issue_artifact(
        artifact,
        destination_prefix="s3://oa-bolinas/issues/479/bico-lora-standard-rate/v1",
        relative_path="adapters/step-0000",
        client=client,
    )
    assert repeated == records
    assert client.uploads == 2

    weights.write_bytes(b"different")
    with pytest.raises(FileExistsError, match="immutable S3 object"):
        upload_issue_artifact(
            artifact,
            destination_prefix="s3://oa-bolinas/issues/479/bico-lora-standard-rate/v1",
            relative_path="adapters/step-0000",
            client=client,
        )


def test_download_verified_issue_object_checks_remote_and_local_bytes(tmp_path: Path) -> None:
    body = b"retained-adapter"
    checksum = hashlib.sha256(body).hexdigest()
    client = _FakeS3()
    key = "issues/479/bico-lora-standard-rate/v1/adapters/step-1000/adapter.bin"
    client.objects[("oa-bolinas", key)] = {
        "body": body,
        "metadata": {"sha256": checksum},
    }
    destination = tmp_path / "adapter.bin"

    record = download_verified_issue_object(
        s3_uri=f"s3://oa-bolinas/{key}",
        destination=destination,
        expected_size_bytes=len(body),
        expected_sha256=checksum,
        client=client,
    )

    assert destination.read_bytes() == body
    assert record["sha256"] == checksum

    client.objects[("oa-bolinas", key)]["metadata"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="metadata changed"):
        download_verified_issue_object(
            s3_uri=f"s3://oa-bolinas/{key}",
            destination=destination,
            expected_size_bytes=len(body),
            expected_sha256=checksum,
            client=client,
        )
