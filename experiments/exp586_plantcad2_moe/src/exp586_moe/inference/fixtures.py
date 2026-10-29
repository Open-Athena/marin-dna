"""Materialize a small immutable benchmark from real PlantCAD evaluation inputs."""

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from exp586_moe.inference.contract import (
    FIXTURE_MANIFEST_SHA256,
    FIXTURE_SOURCE,
    FIXTURE_SOURCE_ROWS,
    FIXTURE_SOURCE_SHA256,
    validate_sequences,
)


def sha256_file(path: Path, *, chunk_size: int = 8 << 20) -> str:
    """Hash a file without loading a real evaluation shard into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_fixture(
    source: Path,
    output: Path,
    *,
    count: int = 8,
    expected_source_sha256: str = FIXTURE_SOURCE_SHA256,
    expected_rows: int = FIXTURE_SOURCE_ROWS,
) -> dict[str, object]:
    """Select the first unique valid windows from a content-addressed shard."""
    if count < 1:
        raise ValueError("Fixture count must be positive")
    actual_sha256 = sha256_file(source)
    if actual_sha256 != expected_source_sha256:
        raise ValueError("Fixture source digest differs from the pinned shard")
    table = pq.read_table(source, columns=["sequence"])
    if table.num_rows != expected_rows:
        raise ValueError("Fixture source row count differs")
    selected: list[str] = []
    seen: set[str] = set()
    for value in table.column("sequence").to_pylist():
        sequence = str(value).upper()
        if sequence in seen:
            continue
        try:
            validate_sequences([sequence])
        except ValueError:
            continue
        selected.append(sequence)
        seen.add(sequence)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Only found {len(selected)} valid unique windows")
    validate_sequences(selected)
    payload: dict[str, object] = {
        "schema_version": 1,
        "source": FIXTURE_SOURCE,
        "source_sha256": actual_sha256,
        "source_rows": table.num_rows,
        "evaluation_manifest_sha256": FIXTURE_MANIFEST_SHA256,
        "sequences": selected,
        "sequence_sha256": [
            hashlib.sha256(sequence.encode()).hexdigest() for sequence in selected
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def load_fixture(path: Path) -> list[str]:
    """Load a materialized fixture and revalidate its content hashes."""
    payload = json.loads(path.read_text())
    sequences = payload["sequences"]
    if not isinstance(sequences, list) or not all(isinstance(x, str) for x in sequences):
        raise ValueError("Fixture sequences have an invalid schema")
    validate_sequences(sequences)
    hashes = [hashlib.sha256(sequence.encode()).hexdigest() for sequence in sequences]
    if hashes != payload.get("sequence_sha256"):
        raise ValueError("Fixture sequence digest differs")
    return sequences

