import hashlib
import io

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from exp593_dna.anneal_task_data import fetch_source, sample_rows


def test_fixed_monitoring_membership_samples_all_shards(tmp_path):
    paths, records = [], []
    for number in range(3):
        path = tmp_path / f"part-{number}.parquet"
        pq.write_table(
            pa.table(
                {
                    "source_id": list(range(number * 10, (number + 1) * 10)),
                    "label": [number % 2] * 10,
                }
            ),
            path,
            row_group_size=3,
        )
        paths.append(path)
        records.append({"rows": 10})
    rows, membership = sample_rows(paths, records, 18, 593100)
    assert len(rows) == len(set(membership)) == 18
    assert membership == sorted(membership)
    assert {i // 10 for i in membership} == {0, 1, 2}
    assert [r["source_id"] for r in rows] == membership
    assert sample_rows(paths, records, 18, 593100) == (rows, membership)
    records[0]["rows"] = 9
    with pytest.raises(ValueError, match="row count differs"):
        sample_rows(paths, records, 18, 593100)


def test_remote_source_verification_uses_no_disk_cache(tmp_path):
    class Client:
        def get_object(self, **kwargs):
            assert kwargs == {"Bucket": "bucket", "Key": "immutable/source"}
            return {"Body": io.BytesIO(b"source bytes")}

    record = {
        "uri": "s3://bucket/immutable/source",
        "path": "source.parquet",
        "bytes": 12,
        "sha256": hashlib.sha256(b"source bytes").hexdigest(),
    }
    stream = fetch_source(Client(), record, tmp_path / "unused")
    assert stream.read() == b"source bytes"
    assert list(tmp_path.iterdir()) == []
    record["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Remote benchmark source differs"):
        fetch_source(Client(), record, tmp_path / "unused")
