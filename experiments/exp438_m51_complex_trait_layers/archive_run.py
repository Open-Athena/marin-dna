"""Create a hash-complete manifest for an experiment 438 run directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import ISSUE, sha256_file, write_json

REQUIRED_DIRECTORIES = ("inputs", "extraction", "associations", "summary")


def verify_nested_manifest(root: Path, directory: str) -> None:
    nested_root = root / directory
    manifest_path = nested_root / "manifest.json"
    assert manifest_path.is_file(), manifest_path
    manifest = json.loads(manifest_path.read_text())
    assert manifest["issue"] == ISSUE
    for relative, expected in manifest["artifacts"].items():
        path = nested_root / relative
        assert path.is_file(), path
        assert path.stat().st_size == expected["bytes"]
        assert sha256_file(path) == expected["sha256"]


def build_archive_manifest(root: Path) -> dict[str, Any]:
    assert root.is_dir()
    assert (root / "inputs" / "panel.parquet").is_file()
    assert (root / "inputs" / "panel.manifest.json").is_file()
    for directory in REQUIRED_DIRECTORIES[1:]:
        verify_nested_manifest(root, directory)
    manifest_path = root / "archive_manifest.json"
    assert not manifest_path.exists()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    artifacts = {
        str(path.relative_to(root)): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    }
    assert artifacts
    run_ids = {
        json.loads((root / directory / "manifest.json").read_text())["run_id"]
        for directory in REQUIRED_DIRECTORIES[1:]
    }
    assert len(run_ids) == 1
    return {
        "issue": ISSUE,
        "run_id": run_ids.pop(),
        "object_count_excluding_this_manifest": len(artifacts),
        "bytes_excluding_this_manifest": sum(
            item["bytes"] for item in artifacts.values()
        ),
        "artifacts": artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_archive_manifest(args.run_root)
    write_json(args.run_root / "archive_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
