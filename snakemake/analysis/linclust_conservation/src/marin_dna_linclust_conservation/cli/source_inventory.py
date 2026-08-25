"""Resolve selected accessions against an existing S3 genome mirror."""

from __future__ import annotations

import argparse
from pathlib import Path

import boto3
import polars as pl

from marin_dna_linclust_conservation.sources import audit_existing_genome_mirror


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--suffix", default=".2bit")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--missing-output", type=Path, required=True)
    args = parser.parse_args()

    manifest = pl.read_csv(args.manifest, separator="\t")
    if "selected" in manifest.columns:
        manifest = manifest.filter(pl.col("selected"))
    accessions = manifest["accession"].to_list()
    assert accessions, "manifest contains no selected accessions"

    inventory, missing = audit_existing_genome_mirror(
        accessions,
        bucket=args.bucket,
        prefix=args.prefix,
        suffix=args.suffix,
        s3_client=boto3.client("s3"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.missing_output.parent.mkdir(parents=True, exist_ok=True)
    inventory.write_csv(args.output, separator="\t")
    missing.write_csv(args.missing_output, separator="\t")
    print(
        f"resolved {inventory.height} mirrored genome sources; "
        f"{missing.height} exact accessions require a fresh NCBI download"
    )


if __name__ == "__main__":
    main()
