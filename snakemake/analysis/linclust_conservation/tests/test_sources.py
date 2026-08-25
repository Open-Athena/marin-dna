from __future__ import annotations

import pytest
from botocore.exceptions import ClientError
from marin_dna_linclust_conservation.sources import audit_existing_genome_mirror


class FakeS3Client:
    def head_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        assert Bucket == "bucket"
        if Key.endswith("GCF_missing.1.2bit"):
            raise ClientError(
                {
                    "Error": {"Code": "404", "Message": "missing"},
                    "ResponseMetadata": {"HTTPStatusCode": 404},
                },
                "HeadObject",
            )
        return {"ETag": '"abc123"', "ContentLength": 42}


class ForbiddenS3Client:
    def head_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        raise ClientError(
            {
                "Error": {"Code": "AccessDenied", "Message": "denied"},
                "ResponseMetadata": {"HTTPStatusCode": 403},
            },
            "HeadObject",
        )


def test_audit_existing_mirror_preserves_hits_and_missing_versions() -> None:
    inventory, missing = audit_existing_genome_mirror(
        ["GCF_present.1", "GCF_missing.1"],
        bucket="bucket",
        prefix="old/prefix/",
        suffix=".2bit",
        s3_client=FakeS3Client(),
    )
    assert inventory.to_dicts() == [
        {
            "accession": "GCF_present.1",
            "download_uri": "s3://bucket/old/prefix/GCF_present.1.2bit",
            "source_checksum_type": "s3_etag",
            "source_checksum": "abc123",
            "source_size_bytes": 42,
        }
    ]
    assert missing.to_dicts() == [
        {
            "accession": "GCF_missing.1",
            "reason": "exact version absent from existing S3 mirror",
        }
    ]


def test_audit_existing_mirror_rejects_empty_selection() -> None:
    with pytest.raises(AssertionError, match="no selected accessions"):
        audit_existing_genome_mirror(
            [],
            bucket="bucket",
            prefix="prefix",
            suffix=".2bit",
            s3_client=FakeS3Client(),
        )


def test_audit_existing_mirror_propagates_access_denial() -> None:
    with pytest.raises(ClientError):
        audit_existing_genome_mirror(
            ["GCF_1.1"],
            bucket="bucket",
            prefix="prefix",
            suffix=".2bit",
            s3_client=ForbiddenS3Client(),
        )
