from marin_dna_linclust_conservation.staging import (
    copy_reused_genome,
    parse_s3_uri,
)


class FakeS3Client:
    def __init__(self) -> None:
        self.copied = False

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        if Bucket == "source":
            return {"ETag": '"old-etag"', "ContentLength": 42}
        assert self.copied
        return {"ETag": '"new-etag"', "ContentLength": 42}

    def copy_object(self, **kwargs: object) -> dict[str, object]:
        assert kwargs["CopySourceIfMatch"] == '"old-etag"'
        self.copied = True
        return {"ResponseMetadata": {"RequestId": "request"}}


def test_parse_s3_uri() -> None:
    assert parse_s3_uri("s3://bucket/path/object") == ("bucket", "path/object")


def test_copy_reused_genome_checks_and_records_source_identity() -> None:
    receipt = copy_reused_genome(
        accession="GCF_1.1",
        source_uri="s3://source/old/GCF_1.1.2bit",
        destination_uri="s3://destination/new/GCF_1.1.2bit",
        source_etag="old-etag",
        source_size_bytes=42,
        s3_client=FakeS3Client(),
    )
    assert receipt["source_checksum"] == "old-etag"
    assert receipt["destination_etag"] == "new-etag"
    assert receipt["destination_size_bytes"] == 42
