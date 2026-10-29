import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from exp586_moe.inference.fixtures import load_fixture, materialize_fixture, sha256_file


def test_materialize_fixture_binds_source_and_sequences(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    sequence = "ACGTN" * 1_638 + "AC"
    pq.write_table(pa.table({"sequence": [sequence.lower(), sequence, "T" * 8192]}), source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / "fixture.json"

    payload = materialize_fixture(
        source,
        output,
        count=2,
        expected_source_sha256=digest,
        expected_rows=3,
    )

    assert load_fixture(output) == [sequence, "T" * 8192]
    assert payload["source_sha256"] == sha256_file(source)


def test_materialize_fixture_rejects_changed_source(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    pq.write_table(pa.table({"sequence": ["A" * 8192]}), source)
    with pytest.raises(ValueError, match="digest"):
        materialize_fixture(source, tmp_path / "fixture.json", expected_source_sha256="0" * 64)

