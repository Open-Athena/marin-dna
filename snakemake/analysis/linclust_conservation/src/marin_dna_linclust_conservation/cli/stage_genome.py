"""Stage one immutable genome input and write a receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import boto3

from marin_dna_linclust_conservation.staging import (
    copy_reused_genome,
    download_convert_upload_genome,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accession", required=True)
    parser.add_argument("--destination-uri", required=True)
    parser.add_argument("--source-uri")
    parser.add_argument("--source-etag")
    parser.add_argument("--source-size-bytes", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    client = boto3.client("s3")
    if args.source_uri:
        assert args.source_etag
        assert args.source_size_bytes is not None
        receipt = copy_reused_genome(
            accession=args.accession,
            source_uri=args.source_uri,
            destination_uri=args.destination_uri,
            source_etag=args.source_etag,
            source_size_bytes=args.source_size_bytes,
            s3_client=client,
        )
    else:
        assert args.source_etag is None
        assert args.source_size_bytes is None
        receipt = download_convert_upload_genome(
            accession=args.accession,
            destination_uri=args.destination_uri,
            s3_client=client,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"staged {args.accession} at {args.destination_uri}")


if __name__ == "__main__":
    main()
