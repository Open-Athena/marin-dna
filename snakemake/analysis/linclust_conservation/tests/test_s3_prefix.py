from botocore.exceptions import ClientError
from marin_dna_linclust_conservation.s3_prefix import copy_s3_prefix


class FakePaginator:
    def paginate(self, **kwargs: object) -> list[dict[str, object]]:
        assert kwargs == {"Bucket": "source", "Prefix": "old/"}
        return [
            {
                "Contents": [
                    {"Key": "old/a", "Size": 3},
                    {"Key": "old/nested/b", "Size": 5},
                ]
            }
        ]


class FakeS3Client:
    def __init__(self) -> None:
        self.destinations = {"new/a": 3}
        self.copies: list[dict[str, object]] = []

    def get_paginator(self, name: str) -> FakePaginator:
        assert name == "list_objects_v2"
        return FakePaginator()

    def head_object(self, **kwargs: object) -> dict[str, object]:
        assert kwargs["Bucket"] == "destination"
        key = str(kwargs["Key"])
        if key not in self.destinations:
            raise ClientError(
                {"Error": {"Code": "404", "Message": "not found"}},
                "HeadObject",
            )
        return {"ContentLength": self.destinations[key]}

    def copy(self, **kwargs: object) -> None:
        assert kwargs["CopySource"] == {
            "Bucket": "source",
            "Key": "old/nested/b",
        }
        assert kwargs["Config"] == "transfer-config"
        self.copies.append(kwargs)
        self.destinations[str(kwargs["Key"])] = 5


def test_copy_s3_prefix_preserves_relative_keys_and_reuses_complete_objects() -> None:
    client = FakeS3Client()
    receipt = copy_s3_prefix(
        source_uri="s3://source/old/",
        destination_uri="s3://destination/new/",
        s3_client=client,
        transfer_config="transfer-config",
    )

    assert receipt["objects_copied"] == 1
    assert receipt["objects_reused"] == 1
    assert receipt["total_bytes"] == 8
    assert len(client.copies) == 1
