"""Create a hash-complete manifest for a compact post-hoc artifact directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import ISSUE, sha256_file, write_json


def build_manifest(
    root: Path, run_id: str, analysis_status: str = "post_hoc_descriptive"
) -> dict[str, object]:
    assert root.is_dir() and run_id and analysis_status
    manifest_path = root / "archive_manifest.json"
    assert not manifest_path.exists()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    assert files
    artifacts = {
        str(path.relative_to(root)): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    }
    return {
        "issue": ISSUE,
        "run_id": run_id,
        "analysis_status": analysis_status,
        "object_count_excluding_this_manifest": len(artifacts),
        "bytes_excluding_this_manifest": sum(
            int(item["bytes"]) for item in artifacts.values()
        ),
        "artifacts": artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--analysis-status", default="post_hoc_descriptive")
    args = parser.parse_args()
    write_json(
        args.root / "archive_manifest.json",
        build_manifest(args.root, args.run_id, args.analysis_status),
    )


if __name__ == "__main__":
    main()
