"""Build the production input manifest from verified workflow publication receipts."""

import hashlib
import json
import re
from pathlib import Path

import boto3

ROOT = "snakemake/vertebrate_projection_dataset/results/rag-five-regions-publication-v1/54b6f936467bbc657ae88753a6f24b768bd3363d/e2af428648e45c1f7b23d938842b85325d9d864dc5335940da99d0cca3bbe0e0/full/rag/publication_provenance"
EXPECTED = {
    "cds": 581256,
    "tss_utr5": 112740,
    "utr3": 131550,
    "ncrna": 56396,
    "enhancer": 228064,
}
client = boto3.client("s3")
manifest = {}
for region, train_rows in EXPECTED.items():
    release_bytes = client.get_object(
        Bucket="oa-bolinas", Key=f"{ROOT}/{region}/release.json"
    )["Body"].read()
    receipt = json.loads(
        client.get_object(Bucket="oa-bolinas", Key=f"{ROOT}/{region}/hub_receipt.json")[
            "Body"
        ].read()
    )
    release = json.loads(release_bytes)
    assert (
        receipt["release_manifest_sha256"] == hashlib.sha256(release_bytes).hexdigest()
    )
    assert (
        receipt["repo_id"]
        == release["repo_id"]
        == f"marin-dna/rag-five-regions-v1-{region}"
    )
    assert re.fullmatch("[0-9a-f]{40}", receipt["revision"])
    assert receipt["anonymous_verified"] is True
    assert release["splits"]["train"]["rows"] == train_rows
    assert release["splits"]["validation"]["rows"] == 400
    manifest[region] = {
        "repo_id": receipt["repo_id"],
        "revision": receipt["revision"],
        "train_rows": train_rows,
        "validation_rows": 400,
        "anonymous_verified": True,
        "release_manifest_sha256": receipt["release_manifest_sha256"],
    }
destination = Path("/opt/issue550/verified-public-datasets.json")
destination.write_text(json.dumps(manifest, indent=2) + "\n")
print(destination)
print(json.dumps(manifest, indent=2))
