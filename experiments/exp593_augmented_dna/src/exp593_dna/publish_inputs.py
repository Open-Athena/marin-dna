"""Publish immutable preparation inputs to the experiment's native S3 owner."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError

from exp593_dna.audit import file_sha256
from exp593_dna.pipeline import DATASET_REVISION
from exp593_dna.storage import coreweave_client

BUCKET = "marin-us-east-02a"
OWNER = "MarinDNA/exp593_augmented_dna"


def inventory(prep: Path, artifacts: Path) -> dict[str, Path]:
    """Include metadata and annotations, without uploading reference DNA to train."""
    entries = {
        "corpus/sources.json": prep / "corpus/sources.json",
        "annotations/train.parquet": prep / "frozen-annotations/train.parquet",
        "annotations/validation.parquet": prep
        / "frozen-annotations/validation.parquet",
        "annotations/manifest.json": prep / "frozen-annotations/manifest.json",
        "annotations/coverage.json": prep / "annotations/coverage.json",
        "validation/primary_validation_512.jsonl": prep
        / "primary_validation_512.jsonl",
        "provenance/dataset_api.json": prep / "sources/dataset_api.json",
    }
    for split, count in (("train", 44), ("validation", 6)):
        paths = sorted((prep / "metadata" / split).glob("*.parquet"))
        if len(paths) != count:
            raise ValueError(f"Missing original {split} metadata shards")
        for path in paths:
            entries[f"metadata/{split}/{path.name}"] = path
    for name in (
        "assembly_candidate_registry.json",
        "annotation_research.md",
        "maize_annotation_verification.json",
        "maize_reference_verification.json",
    ):
        entries[f"provenance/{name}"] = prep / "provenance" / name
    for path in sorted((prep / "references").glob("*/reference.json")):
        entries[f"references/{path.parent.name}.json"] = path
    for name in (
        "augmentation_report.json",
        "examples.jsonl",
        "template_catalog.json",
        "validation_primary_512.json",
        "validation_disjoint_512.json",
        "holdout_verification.json",
        "fim_validation_pairs.jsonl",
        "fim_validation_manifest.json",
    ):
        entries[f"audit/{name}"] = artifacts / name
    if any(not path.is_file() for path in entries.values()):
        raise ValueError("Preparation inventory has missing files")
    return entries


def put_immutable(
    client: Any, path: Path, key: str, digest: str, size: int
) -> dict[str, Any]:
    """Never replace a key; read back the entire uploaded object before accepting it."""
    try:
        head = client.head_object(Bucket=BUCKET, Key=key)
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in ("404", "NoSuchKey", "NotFound"):
            raise
        with path.open("rb") as source:
            client.put_object(
                Bucket=BUCKET,
                Key=key,
                Body=source,
                ContentLength=size,
                IfNoneMatch="*",
                Metadata={"sha256": digest, "dataset_revision": DATASET_REVISION},
            )
        head = client.head_object(Bucket=BUCKET, Key=key)
    if (
        head["ContentLength"] != size
        or head.get("Metadata", {}).get("sha256") != digest
    ):
        raise ValueError(f"Immutable object identity conflicts: {key}")
    response = client.get_object(Bucket=BUCKET, Key=key, IfMatch=head["ETag"])
    actual = hashlib.sha256()
    with response["Body"] as stream:
        while block := stream.read(8 * 1024 * 1024):
            actual.update(block)
    if actual.hexdigest() != digest:
        raise ValueError(f"Uploaded object failed complete readback: {key}")
    return {
        "uri": f"s3://{BUCKET}/{key}",
        "sha256": digest,
        "bytes": size,
        "etag": head["ETag"],
        "complete_readback_verified": True,
    }


def publish(
    prep: Path, artifacts: Path, output: Path, code_commit: str
) -> dict[str, Any]:
    """Use a content-addressed prefix and commit the manifest after all files verify."""
    if len(code_commit) != 40 or any(c not in "0123456789abcdef" for c in code_commit):
        raise ValueError("A full source commit SHA is required")
    paths = inventory(prep, artifacts)
    records = {
        key: {"sha256": file_sha256(path), "bytes": path.stat().st_size}
        for key, path in sorted(paths.items())
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            {"files": records, "source_code_commit": code_commit}, sort_keys=True
        ).encode()
    ).hexdigest()
    prefix = f"{OWNER}/inputs/prep-v2-{fingerprint}"
    client = coreweave_client(refresh_keys=True)
    uploaded = {}
    for name, record in records.items():
        uploaded[name] = put_immutable(
            client, paths[name], f"{prefix}/{name}", record["sha256"], record["bytes"]
        )
        print(json.dumps({"verified": name, "bytes": record["bytes"]}), flush=True)
    manifest = {
        "dataset_revision": DATASET_REVISION,
        "source_code_commit": code_commit,
        "fingerprint": fingerprint,
        "prefix": f"s3://{BUCKET}/{prefix}",
        "files": uploaded,
        "training_sequences": "Original exp586 train objects only; external reference FASTA is used for annotation mapping and is not part of this training inventory.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    # The uploaded marker appears only after all earlier objects passed readback.
    manifest["manifest_object"] = put_immutable(
        client,
        output,
        f"{prefix}/manifest.json",
        file_sha256(output),
        output.stat().st_size,
    )
    manifest["readback_completed_utc"] = datetime.now(UTC).isoformat()
    receipt = output.with_name(output.stem + "-receipt.json")
    receipt.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    result = publish(args.prep, args.artifacts, args.output, args.code_commit)
    print(json.dumps({"manifest": result["manifest_object"]}, indent=2))


if __name__ == "__main__":
    main()
