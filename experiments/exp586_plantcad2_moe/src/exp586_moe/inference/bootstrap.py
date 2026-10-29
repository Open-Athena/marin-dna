"""Restore reusable single-H100 benchmark artifacts from CoreWeave S3."""

import argparse
from pathlib import Path

import fsspec
from rigging.filesystem.s3_compat import configure_coreweave_s3

BOOTSTRAP_PREFIX = "marin-us-east-02a/tmp/ttl=7d/MarinDNA/exp586-inference/bootstrap"
EXPORT_PREFIX = (
    "marin-us-east-02a/MarinDNA/exp586_plantcad2_moe/inference-optimization/v1/hf-bf16"
)


def _download_prefix(fs: fsspec.AbstractFileSystem, prefix: str, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for item in fs.ls(prefix, detail=True):
        if item.get("type") == "directory":
            continue
        target = output / Path(item["name"]).name
        expected_size = int(item["size"])
        if target.exists() and target.stat().st_size == expected_size:
            print(f"reuse {target.name}", flush=True)
            continue
        fs.get_file(item["name"], str(target))
        if target.stat().st_size != expected_size:
            raise RuntimeError(f"Incomplete artifact: {target}")
        print(f"downloaded {target.name} {expected_size}", flush=True)


def bootstrap(root: Path) -> None:
    """Download immutable fixtures and the validated BF16 model export."""
    configure_coreweave_s3()
    fs = fsspec.filesystem("s3")
    _download_prefix(fs, BOOTSTRAP_PREFIX, root)
    _download_prefix(fs, EXPORT_PREFIX, root / "hf-bf16")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args()
    bootstrap(arguments.root)
