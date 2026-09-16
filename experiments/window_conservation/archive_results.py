"""Upload task-only result atoms to the existing private research owner."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for data in iter(lambda: handle.read(2**20), b""):
            value.update(data)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--extension", action="store_true")
    parser.add_argument("--repeatfree", action="store_true")
    args = parser.parse_args()
    assert not (args.extension and args.repeatfree)
    expanded = args.extension or args.repeatfree
    assert len(args.commit) == 40
    client = boto3.client("s3", region_name="us-east-2")
    bucket, owner = "oa-bolinas", "836683583872"
    client.head_bucket(Bucket=bucket, ExpectedBucketOwner=owner)
    assert all(
        client.get_public_access_block(Bucket=bucket, ExpectedBucketOwner=owner)[
            "PublicAccessBlockConfiguration"
        ].values()
    )
    version = "local100-repeatfree-v1" if args.repeatfree else "local100-extension-v1" if args.extension else "local100-v1"
    prefix = f"issues/577/{version}/{args.commit}/"
    assert not client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1).get(
        "KeyCount"
    ), "archive prefix already exists"
    paths = [args.root / "data/manifest.json", args.root / "source.tar.gz"]
    paths += list((args.root / "data").glob("labels-*.npz"))
    paths += [
        p
        for directory in ["report", "logs"]
        for p in (args.root / directory).rglob("*")
        if p.is_file()
    ]
    if expanded:
        if args.extension:
            paths += [args.root / "data/rmsk.txt.gz"]
        else:
            paths += [args.root / "data/protocol.json"]
        paths += [
            p
            for directory in ["memory", "spatial", "global3/selection", "profiles"]
            for p in (args.root / directory).rglob("*")
            if p.is_file()
            and p.suffix
            in [".json", ".csv", ".fa", ".bed", ".time", ".stdout", ".stderr", ".list"]
        ]
    paths += [
        p
        for directory in [
            "exploratory4096/report",
            "exploratory4096/logs",
            "slow-query-prototype",
        ]
        for p in (args.root / directory).rglob("*")
        if p.is_file()
    ]
    paths += [
        p
        for p in (args.root / "scaling").rglob("*")
        if p.is_file() and p.suffix in [".json", ".time", ".stdout", ".stderr", ".list"]
    ]
    compressed = args.root / "archive-scores"
    compressed.mkdir(exist_ok=True)
    score_paths = list((args.root / "scores").rglob("*.tsv"))
    if expanded:
        score_paths.append(args.root / "global3/scores.tsv")
    for path in score_paths:
        relative = str(path.relative_to(args.root)) if expanded else path.name
        output = compressed / (relative + ".gz")
        output.parent.mkdir(parents=True, exist_ok=True)
        with (
            path.open("rb") as src,
            output.open("wb") as handle,
            gzip.GzipFile(
                fileobj=handle,
                mode="wb",
                mtime=0,
                compresslevel=1 if expanded else 9,
            ) as dst,
        ):
            shutil.copyfileobj(src, dst, length=2**20)
        paths.append(output)
    manifest = {
        "producing_commit": args.commit,
        "prefix": f"s3://{bucket}/{prefix}",
        "files": [],
    }
    assert sum(p.stat().st_size for p in paths) < (8 if expanded else 2) * 2**30
    for path in sorted(paths):
        key = prefix + str(path.relative_to(args.root))
        checksum = digest(path)
        client.upload_file(
            str(path),
            bucket,
            key,
            ExtraArgs={"Metadata": {"sha256": checksum}},
            Config=TransferConfig(use_threads=False),
        )
        response = client.get_object(Bucket=bucket, Key=key, ExpectedBucketOwner=owner)
        actual = hashlib.sha256()
        while data := response["Body"].read(2**20):
            actual.update(data)
        assert actual.hexdigest() == checksum
        manifest["files"].append(
            {"key": key, "bytes": path.stat().st_size, "sha256": checksum}
        )
    output = args.root / "archive_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    client.upload_file(
        str(output),
        bucket,
        prefix + "manifest.json",
        Config=TransferConfig(use_threads=False),
    )
    assert (
        client.get_object(
            Bucket=bucket, Key=prefix + "manifest.json", ExpectedBucketOwner=owner
        )["Body"].read()
        == output.read_bytes()
    )
    print(
        json.dumps(
            {
                "prefix": manifest["prefix"],
                "files": len(paths),
                "bytes": sum(p.stat().st_size for p in paths),
                "manifest_sha256": digest(output),
            }
        )
    )


if __name__ == "__main__":
    main()
