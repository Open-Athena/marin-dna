from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest
from marin_dna_vertebrate_projection.mirror import (
    MirrorObject,
    stage_hal_object,
    stage_s3_object,
)


def test_stage_s3_object_atomically_installs_verified_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"maf"
    expected = MirrorObject(
        kind="primary_chromosome_maf",
        chrom="chr18",
        source_url="https://example.test/chr18.maf.gz",
        s3_uri="s3://example/chr18.maf.gz",
        byte_size=len(payload),
        md5=hashlib.md5(payload).hexdigest(),
    )
    destination = tmp_path / "chr18.maf.gz"

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        partial = Path(command[4])
        assert partial.name == ".chr18.maf.gz.partial"
        partial.write_bytes(payload)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    stage_s3_object(expected, destination)
    assert destination.read_bytes() == payload
    assert not (tmp_path / ".chr18.maf.gz.partial").exists()


def test_stage_hal_object_atomically_installs_readable_hal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "genomes.hal"

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if command[:2] == ["aws", "s3api"]:
            return subprocess.CompletedProcess(command, 0, stdout="8\n")
        if command[:3] == ["aws", "s3", "cp"]:
            partial = Path(command[4])
            assert partial.name == ".genomes.hal.partial"
            partial.write_bytes(b"test-hal")
            return subprocess.CompletedProcess(command, 0)
        assert command[:2] == ["halStats", "--genomes"]
        return subprocess.CompletedProcess(
            command, 0, stdout="Homo_sapiens Mus_musculus\n"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    stage_hal_object("s3://example/genomes.hal", destination)
    assert destination.read_bytes() == b"test-hal"
    assert not (tmp_path / ".genomes.hal.partial").exists()
