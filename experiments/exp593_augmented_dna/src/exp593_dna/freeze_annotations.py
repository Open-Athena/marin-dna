"""Freeze verified row sidecars into a compact, queryable annotation bank."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from exp593_dna.annotations import Landmark
from exp593_dna.audit import file_sha256
from exp593_dna.pipeline import DATASET_REVISION


def freeze_sidecar(source: Path, output: Path, split: str) -> dict[str, Any]:
    """Check revision/split, deduplicate row identity and preserve every feature."""
    maximum = {"train": 2_638_656, "validation": 329_832}[split]
    seen = bytearray(maximum)
    schema = pa.schema(
        [
            ("row", pa.int32()),
            ("sequence_sha256", pa.string()),
            (
                "landmarks",
                pa.list_(
                    pa.struct(
                        [
                            ("start", pa.int16()),
                            ("end", pa.int16()),
                            ("kind", pa.string()),
                            ("source", pa.string()),
                            ("feature_id", pa.string()),
                        ]
                    )
                ),
            ),
        ],
        metadata={
            b"dataset_revision": DATASET_REVISION.encode(),
            b"split": split.encode(),
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".part")
    counts: Counter[str] = Counter()
    rows = []
    with (
        source.open() as stream,
        pq.ParquetWriter(temporary, schema, compression="zstd") as writer,
    ):
        for line in stream:
            record = json.loads(line)
            if (
                record["dataset_revision"] != DATASET_REVISION
                or record["split"] != split
            ):
                raise ValueError("Annotation source revision or split differs")
            row = record["row"]
            if not isinstance(row, int) or not 0 <= row < maximum or seen[row]:
                raise ValueError("Invalid or duplicated annotation row")
            seen[row] = 1
            if len(record["sequence_sha256"]) != 64:
                raise ValueError("Missing original DNA hash binding")
            bytes.fromhex(record["sequence_sha256"])
            for value in record["landmarks"]:
                motif = Landmark(**value)
                counts[motif.kind] += 1
            if not record["landmarks"]:
                raise ValueError("An annotation row must have at least one landmark")
            rows.append({name: record[name] for name in schema.names})
            counts["annotated_rows"] += 1
            if len(rows) == 8192:
                writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                rows.clear()
        if rows:
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    temporary.replace(output)
    if pq.ParquetFile(output).metadata.num_rows != counts["annotated_rows"]:
        raise ValueError("Frozen annotation row count changed")
    return {
        "dataset_revision": DATASET_REVISION,
        "split": split,
        "source_sha256": file_sha256(source),
        "parquet_sha256": file_sha256(output),
        "parquet_bytes": output.stat().st_size,
        "counts": dict(counts),
        "source_rows": maximum,
        "fraction_source_rows_annotated": counts["annotated_rows"] / maximum,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = []
    for split in ("train", "validation"):
        report = freeze_sidecar(
            args.annotations / f"{split}.jsonl", args.output / f"{split}.parquet", split
        )
        reports.append(report)
        print(json.dumps(report), flush=True)
    (args.output / "manifest.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
