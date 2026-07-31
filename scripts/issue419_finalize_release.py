"""Regenerate the issue #419 dataset card and manifest before publication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from marin_dna.pipelines.chinchilla_logo import (
    write_dataset_readme,
    write_release_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--application-commit", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert len(args.application_commit) == 40
    results_root = args.results_root
    plan_paths = sorted((results_root / "plans").glob("*.coverage.json"))
    assert plan_paths, "no coverage plans found"
    coverage_rows = [json.loads(path.read_text())["coverage"] for path in plan_paths]
    scaffolds = [str(row["chrom"]) for row in coverage_rows]
    runtime_paths = [
        results_root / "shards" / f"{chrom}.runtime.json" for chrom in scaffolds
    ]
    assert all(path.is_file() for path in runtime_paths), runtime_paths
    artifact_runtime_path = results_root / "release" / "manifest" / "bigwig_build.json"
    assert artifact_runtime_path.is_file()
    chrom_sizes_path = results_root / "GCF_000276665.1.chrom.sizes.txt"
    assert chrom_sizes_path.is_file()
    release_root = results_root / "release"

    write_dataset_readme(
        release_root / "README.md",
        application_commit=args.application_commit,
        scaffolds=scaffolds,
    )
    manifest = write_release_manifest(
        release_root,
        chrom_sizes_path,
        plan_paths,
        runtime_paths,
        application_commit=args.application_commit,
        artifact_runtime_paths=[artifact_runtime_path],
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
