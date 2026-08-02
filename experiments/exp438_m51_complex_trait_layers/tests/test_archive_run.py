from __future__ import annotations

import json

from archive_run import build_archive_manifest
from common import ISSUE, sha256_file


def test_archive_manifest_verifies_nested_artifacts(tmp_path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "panel.parquet").write_bytes(b"panel")
    (inputs / "panel.manifest.json").write_text("{}")
    for directory in ("extraction", "associations", "summary"):
        root = tmp_path / directory
        root.mkdir()
        artifact = root / "artifact.txt"
        artifact.write_text(directory)
        expected = {"bytes": artifact.stat().st_size, "sha256": sha256_file(artifact)}
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "issue": ISSUE,
                    "run_id": "dna-exp438-test",
                    "artifacts": {"artifact.txt": expected},
                }
            )
        )
    manifest = build_archive_manifest(tmp_path)
    assert manifest["run_id"] == "dna-exp438-test"
    assert manifest["object_count_excluding_this_manifest"] == 8
    assert manifest["artifacts"]["inputs/panel.parquet"]["bytes"] == 5
