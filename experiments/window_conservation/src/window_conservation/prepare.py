"""Fetch pinned genomes and evaluation labels; export bounded FASTA chunks."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import urllib.request
from pathlib import Path

import boto3
import py2bit
import yaml

LABEL_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/phastConsElements100way.txt.gz"
SCHEMA_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/phastConsElements100way.sql"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    data = args.root / "data"
    data.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path("config/genomes.yaml").read_text())
    client = boto3.client("s3", region_name="us-east-2")
    receipt: dict = {"sources": [], "labels": {}, "coordinates": "0-based half-open"}
    paths = []
    for source in cfg["homology_fixture"]["sources"]:
        name = source["label"]
        path = data / f"{name}.2bit"
        bucket, key = source["genome_uri"].removeprefix("s3://").split("/", 1)
        head = client.head_object(Bucket=bucket, Key=key, ExpectedBucketOwner="836683583872")
        assert head["ETag"].strip('"') == source["genome_etag"]
        assert head["ContentLength"] == source["genome_size_bytes"]
        if not path.exists():
            client.download_file(bucket, key, str(path))
        assert path.stat().st_size == source["genome_size_bytes"]
        genome = py2bit.open(str(path), True)
        chroms = genome.chroms()
        if name == "human":
            assert chroms["chr1"] == 248956422 and chroms["chr2"] == 242193529
        fasta = data / f"{name}.fa"
        with fasta.open("w") as handle:
            for chrom, length in chroms.items():
                handle.write(f">{chrom}\n")
                for start in range(0, length, 2**20):
                    sequence = genome.sequence(chrom, start, min(length, start + 2**20))
                    assert len(sequence) == min(length - start, 2**20)
                    handle.write(sequence + "\n")
        genome.close()
        receipt["sources"].append(
            {"species": name, "assembly": source["assembly"], "uri": source["genome_uri"],
             "etag": head["ETag"], "bytes": path.stat().st_size, "sha256": sha256(path),
             "fasta_sha256": sha256(fasta), "chromosomes": chroms,
             "bases": sum(chroms.values())}
        )
        paths.append(str(fasta))
        print("prepared", name, sum(chroms.values()), flush=True)
        (data / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (data / "genomes.list").write_text("\n".join(paths) + "\n")
    for url, name in [(LABEL_URL, "phastConsElements100way.txt.gz"), (SCHEMA_URL, "phastConsElements100way.sql")]:
        path = data / name
        urllib.request.urlretrieve(url, path)
        receipt["labels"][name] = {"url": url, "sha256": sha256(path), "bytes": path.stat().st_size}
    with gzip.open(data / "phastConsElements100way.txt.gz", "rt") as handle:
        first = handle.readline().rstrip().split("\t")
    assert first[1].startswith("chr") and int(first[2]) < int(first[3])
    receipt["labels"]["first_row_schema_check"] = first
    (data / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
