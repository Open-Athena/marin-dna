"""Operational pre-stage for #473 sources whose S3 objects lack checksum metadata.

The immutable direct manifest pins CRC64NVME values. The preserved #417
inventory pins object paths and byte sizes; this helper records a freshly
computed CRC64NVME and SHA-256 for every restored member. It is intentionally
separate from the existing Snakemake rules.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

import yaml

ISSUE_417_PREFIX = (
    "s3://oa-bolinas/staging/vertebrate_projection_dataset/v1/"
    "06549d8f7f3ba76151b9c54a5e52d3e3f4402a2d/full"
)
DIRECT_FILENAMES = {
    "exp351_noexon_bed": "exp351_noexon.bed.gz",
    "exp351_scored_anchors": "exp351_scored_anchors.parquet",
    "issue417_artifact_inventory": "issue417_artifact_inventory.tsv",
    "issue417_anchor_labels": "issue417_anchor_labels.parquet",
    "issue417_active_species": "issue417_active_species.tsv",
    "issue417_all_sequences": "issue417_all_sequences.parquet",
    "issue417_per_anchor_qc": "issue417_per_anchor_qc.parquet",
    "issue417_per_scope_qc": "issue417_per_scope_qc.parquet",
    "issue417_rejection_counts": "issue417_rejection_counts.parquet",
    "issue417_aggregates": "issue417_aggregates.parquet",
}


class Crc64Nvme(Protocol):
    def __call__(self, data: bytes, previous_crc64nvme: int = 0) -> int: ...


@dataclass(frozen=True)
class RestoreObject:
    name: str
    s3_uri: str
    byte_size: int
    destination: Path
    receipt: Path
    expected_crc64nvme: str | None


def required_inventory_paths(
    inventory_path: str | Path,
    species_path: str | Path,
    standard_chroms: list[str],
) -> dict[str, int]:
    """Select only scored anchors and rejection evidence consumed by #473."""
    with Path(inventory_path).open(newline="") as handle:
        inventory = {
            row[0]: int(row[1]) for row in csv.reader(handle, delimiter="\t") if row
        }
    with Path(species_path).open(newline="") as handle:
        species = list(csv.DictReader(handle, delimiter="\t"))
    selected = [row for row in species if row["selected"].lower() == "true"]
    required = {f"anchors/scored/{chrom}.parquet" for chrom in standard_chroms}
    for row in selected:
        alignment_name = row["alignment_name"]
        if row["backend"] == "zoonomia_cactus":
            backend = "hal"
        else:
            assert row["backend"] == "ucsc_multiz100way"
            backend = "multiz"
        required.update(
            {
                f"{backend}/rejected/{alignment_name}.parquet",
                f"{backend}/sequence_rejected/{alignment_name}.parquet",
            }
        )
    missing = required - set(inventory)
    assert not missing, f"#417 inventory is missing required paths: {sorted(missing)}"
    return {path: inventory[path] for path in sorted(required)}


def _bucket_key(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    assert parsed.scheme == "s3" and parsed.netloc and parsed.path
    return parsed.netloc, parsed.path.lstrip("/")


def _head_size(aws: str, uri: str) -> int:
    bucket, key = _bucket_key(uri)
    result = subprocess.run(
        [
            aws,
            "s3api",
            "head-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--query",
            "ContentLength",
            "--output",
            "text",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def _download(aws: str, uri: str, destination: Path) -> None:
    bucket, key = _bucket_key(uri)
    subprocess.run(
        [
            aws,
            "s3api",
            "get-object",
            "--bucket",
            bucket,
            "--key",
            key,
            str(destination),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def _hashes(path: Path, crc64nvme: Crc64Nvme) -> tuple[str, str]:
    sha256 = hashlib.sha256()
    crc64 = 0
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            sha256.update(chunk)
            crc64 = crc64nvme(chunk, crc64)
    crc64_b64 = base64.b64encode(crc64.to_bytes(8, "big")).decode()
    return sha256.hexdigest(), crc64_b64


def _restore(
    aws: str, source: RestoreObject, crc64nvme: Crc64Nvme
) -> dict[str, object]:
    remote_size = _head_size(aws, source.s3_uri)
    assert remote_size == source.byte_size, (
        f"remote size changed for {source.name}: {remote_size} != {source.byte_size}"
    )
    source.destination.parent.mkdir(parents=True, exist_ok=True)
    source.receipt.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if (
        not source.destination.is_file()
        or source.destination.stat().st_size != remote_size
    ):
        partial = source.destination.with_name(f".{source.destination.name}.partial")
        partial.unlink(missing_ok=True)
        try:
            _download(aws, source.s3_uri, partial)
            assert partial.stat().st_size == source.byte_size
            partial.replace(source.destination)
        finally:
            partial.unlink(missing_ok=True)
    sha256, crc64 = _hashes(source.destination, crc64nvme)
    if source.expected_crc64nvme is not None:
        assert crc64 == source.expected_crc64nvme, (
            f"CRC64NVME changed for {source.name}: "
            f"{crc64} != {source.expected_crc64nvme}"
        )
    finished = time.time()
    receipt = {
        "name": source.name,
        "s3_uri": source.s3_uri,
        "byte_size": source.byte_size,
        "checksum_crc64nvme": crc64,
        "sha256": sha256,
        "source_checksum_metadata": "absent; recomputed after exact-size restore",
        "started_unix": started,
        "finished_unix": finished,
        "elapsed_seconds": finished - started,
        "average_mib_per_second": (
            source.byte_size / (1024 * 1024) / max(finished - started, 1e-9)
        ),
    }
    temporary = source.receipt.with_name(f".{source.receipt.name}.partial")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(source.receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return receipt


def _direct_objects(project_root: Path, stage_root: Path) -> list[RestoreObject]:
    manifest_path = project_root / "config/issue_473_immutable_sources.tsv"
    with manifest_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert set(DIRECT_FILENAMES) == {row["name"] for row in rows}
    return [
        RestoreObject(
            name=row["name"],
            s3_uri=row["s3_uri"],
            byte_size=int(row["byte_size"]),
            destination=stage_root / "direct" / DIRECT_FILENAMES[row["name"]],
            receipt=(
                stage_root / "receipts/direct" / f"{DIRECT_FILENAMES[row['name']]}.json"
            ),
            expected_crc64nvme=row["checksum_crc64nvme"],
        )
        for row in rows
    ]


def _inventory_objects(project_root: Path, stage_root: Path) -> list[RestoreObject]:
    config = yaml.safe_load((project_root / "config/config.yaml").read_text())
    assert isinstance(config, dict)
    paths = required_inventory_paths(
        stage_root / "direct/issue417_artifact_inventory.tsv",
        project_root / "config/species_selected.tsv",
        [str(chrom) for chrom in config["standard_chroms"]],
    )
    objects: list[RestoreObject] = []
    for relative_path, byte_size in paths.items():
        parts = Path(relative_path).parts
        if parts[:2] == ("anchors", "scored"):
            destination = stage_root / "issue417/scored" / parts[-1]
            receipt = (
                stage_root / "receipts/issue417/scored" / f"{Path(parts[-1]).stem}.json"
            )
        else:
            assert len(parts) == 3 and parts[0] in {"hal", "multiz"}
            destination = stage_root / "issue417/rejections" / relative_path
            receipt = stage_root.joinpath(
                "receipts/issue417/rejections",
                parts[0],
                parts[1],
                f"{Path(parts[2]).stem}.json",
            )
        objects.append(
            RestoreObject(
                name=relative_path.replace("/", "__"),
                s3_uri=f"{ISSUE_417_PREFIX}/{relative_path}",
                byte_size=byte_size,
                destination=destination,
                receipt=receipt,
                expected_crc64nvme=None,
            )
        )
    return objects


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--stage-root",
        type=Path,
        default=Path("/mnt/nvme/vertebrate_projection/issue_473_immutable_sources"),
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    stage_root = args.stage_root.resolve()
    aws = shutil.which("aws")
    assert aws is not None, "aws CLI not found"
    from awscrt.checksums import crc64nvme

    overall_started = time.time()
    receipts = []
    for source in _direct_objects(project_root, stage_root):
        receipts.append(_restore(aws, source, crc64nvme))
    for source in _inventory_objects(project_root, stage_root):
        receipts.append(_restore(aws, source, crc64nvme))
    summary = {
        "objects": len(receipts),
        "bytes": sum(int(receipt["byte_size"]) for receipt in receipts),
        "started_unix": overall_started,
        "finished_unix": time.time(),
        "pid": os.getpid(),
    }
    summary_path = stage_root / "receipts/prestage_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
