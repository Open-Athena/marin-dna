"""Archive a bounded research snapshot; never copy the upstream genome assets."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import boto3


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--selection-commit", required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    assert args.prefix.startswith("issues/568/") and not args.prefix.endswith("/")
    root = args.root
    files: dict[str, Path] = {}
    for relative_root in ["v2", "v2/synthetic", "v2/synthetic-paralogs10"]:
        base = root / relative_root
        for pattern in [
            "data/*.json",
            "data/*.jsonl.gz",
            "data/prior128.*",
            "results/*",
            "report/*",
        ]:
            for path in base.glob(pattern):
                if path.is_file():
                    files[str(path.relative_to(root))] = path
    for path in (root / "v2").glob("linclust-*/*"):
        if path.is_file() and (
            path.name == "assignments.tsv" or path.suffix in [".json", ".log", ".time"]
        ):
            files[str(path.relative_to(root))] = path
    for pattern in ["v2/*.log", "v2/*.json", "aws-price.json", "source.tar.gz"]:
        for path in root.glob(pattern):
            if path.is_file():
                files[str(path.relative_to(root))] = path
    # Preserve invalidated metrics as a clearly separate audit trail.
    for pattern in [
        "data/contexts.jsonl.gz",
        "data/fixture_manifest.json",
        "results/*",
    ]:
        for path in root.glob(pattern):
            if path.is_file():
                files["superseded-v1/" + str(path.relative_to(root))] = path
    assert files and sum(p.stat().st_size for p in files.values()) < 1024**3
    client = boto3.client("s3", region_name="us-east-2")
    bucket = "oa-bolinas"
    assert not client.list_objects_v2(
        Bucket=bucket, Prefix=args.prefix + "/", MaxKeys=1
    ).get("Contents"), "Refuse to overwrite a snapshot"
    manifest = {
        "issue": "https://github.com/Open-Athena/marin-dna/issues/568",
        "created_utc": datetime.now(UTC).isoformat(),
        "code_commit": args.code_commit,
        "selection_commit": args.selection_commit,
        "s3_uri": f"s3://{bucket}/{args.prefix}/",
        "invalidated_prefix": "superseded-v1/",
        "files": [],
    }
    for relative, path in sorted(files.items()):
        sha = digest(path)
        key = f"{args.prefix}/{relative}"
        client.upload_file(
            str(path), bucket, key, ExtraArgs={"Metadata": {"sha256": sha}}
        )
        head = client.head_object(Bucket=bucket, Key=key)
        assert head["ContentLength"] == path.stat().st_size
        assert head["Metadata"]["sha256"] == sha
        # Verify actual object bytes, not only caller-supplied checksum metadata.
        body = client.get_object(Bucket=bucket, Key=key)["Body"]
        remote = hashlib.sha256()
        try:
            while chunk := body.read(2**20):
                remote.update(chunk)
        finally:
            body.close()
        assert remote.hexdigest() == sha
        manifest["files"].append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha,
                "etag": head["ETag"],
                "version_id": head.get("VersionId"),
            }
        )
    output = root / "archive_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    client.upload_file(str(output), bucket, f"{args.prefix}/manifest.json")
    print(
        json.dumps(
            {
                "uri": manifest["s3_uri"],
                "files": len(files),
                "bytes": sum(p.stat().st_size for p in files.values()),
                "manifest_sha256": digest(output),
            }
        )
    )


if __name__ == "__main__":
    main()
