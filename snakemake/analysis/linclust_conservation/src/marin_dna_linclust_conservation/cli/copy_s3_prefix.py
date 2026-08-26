"""Copy an immutable S3 prefix with managed multipart transfers."""

from __future__ import annotations

import argparse
import json

import boto3
from boto3.s3.transfer import TransferConfig

from marin_dna_linclust_conservation.s3_prefix import copy_s3_prefix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-uri", required=True)
    parser.add_argument("--destination-uri", required=True)
    args = parser.parse_args()

    receipt = copy_s3_prefix(
        source_uri=args.source_uri,
        destination_uri=args.destination_uri,
        s3_client=boto3.client("s3"),
        transfer_config=TransferConfig(max_concurrency=8),
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
