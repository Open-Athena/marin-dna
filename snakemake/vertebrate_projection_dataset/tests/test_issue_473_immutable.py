from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from marin_dna_vertebrate_projection.issue_473.immutable import (
    ImmutableS3Object,
    read_artifact_inventory,
    stage_immutable_s3_object,
)


def test_stage_immutable_s3_object_records_verified_checksums(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"immutable-data"
    expected = ImmutableS3Object(
        name="artifact",
        s3_uri="s3://example/path/artifact.parquet",
        byte_size=len(payload),
        checksum_crc64nvme="AQIDBA==",
    )
    destination = tmp_path / "artifact.parquet"
    receipt = tmp_path / "receipt.json"
    calls: list[list[str]] = []

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[2] == "head-object":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(
                    {
                        "ContentLength": len(payload),
                        "ChecksumType": "FULL_OBJECT",
                        "ChecksumCRC64NVME": expected.checksum_crc64nvme,
                    }
                ),
            )
        partial = Path(command[-1])
        partial.write_bytes(payload)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"ChecksumCRC64NVME": expected.checksum_crc64nvme}),
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    stage_immutable_s3_object(expected, destination, receipt)

    assert destination.read_bytes() == payload
    recorded = json.loads(receipt.read_text())
    assert recorded["s3_uri"] == expected.s3_uri
    assert recorded["checksum_crc64nvme"] == expected.checksum_crc64nvme
    assert len(recorded["sha256"]) == 64
    assert len(calls) == 2


def test_read_artifact_inventory_rejects_unsafe_paths(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.tsv"
    inventory.write_text("anchors/scored/chr1.parquet\t42\n")
    assert read_artifact_inventory(inventory) == {"anchors/scored/chr1.parquet": 42}

    inventory.write_text("../outside.parquet\t42\n")
    with pytest.raises(AssertionError):
        read_artifact_inventory(inventory)
