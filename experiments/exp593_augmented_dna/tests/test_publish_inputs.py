import hashlib
import io

import pytest
from botocore.exceptions import ClientError
from botocore.response import StreamingBody

from exp593_dna.publish_inputs import put_immutable


class ObjectClient:
    def __init__(self):
        self.objects = {}
        self.uploads = 0
        self.corrupt_readback = False

    def head_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        payload, metadata = self.objects[Key]
        return {"ContentLength": len(payload), "Metadata": metadata, "ETag": '"etag"'}

    def put_object(self, *, Bucket, Key, Body, ContentLength, IfNoneMatch, Metadata):
        assert IfNoneMatch == "*" and Key not in self.objects
        payload = Body.read()
        assert len(payload) == ContentLength
        self.objects[Key] = (payload, Metadata)
        self.uploads += 1

    def get_object(self, *, Bucket, Key, IfMatch):
        assert IfMatch == '"etag"'
        payload = self.objects[Key][0]
        if self.corrupt_readback:
            payload = payload[::-1]
        return {"Body": StreamingBody(io.BytesIO(payload), len(payload))}


def test_publish_verifies_full_readback_and_cannot_replace_an_existing_key(tmp_path):
    path = tmp_path / "input.json"
    path.write_bytes(b"immutable scientific input")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    client = ObjectClient()
    result = put_immutable(client, path, "owner/key", digest, path.stat().st_size)
    assert result["complete_readback_verified"] and client.uploads == 1
    assert (
        put_immutable(client, path, "owner/key", digest, path.stat().st_size) == result
    )
    assert client.uploads == 1
    client.corrupt_readback = True
    with pytest.raises(ValueError, match="readback"):
        put_immutable(client, path, "owner/key", digest, path.stat().st_size)
    path.write_bytes(b"different scientific input")
    with pytest.raises(ValueError, match="conflicts"):
        put_immutable(
            client,
            path,
            "owner/key",
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_size,
        )
    assert client.uploads == 1
