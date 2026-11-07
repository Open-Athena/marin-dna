"""Join original sequence-only objects to pinned metadata without changing row IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq


def partition_split(
    split: str,
    records: list[dict[str, Any]],
    corpus_root: Path,
    metadata_root: Path,
    output: Path,
    *,
    expected_rows: int,
) -> dict[str, Any]:
    """Preserve original sequence order and compare all frozen raw DNA samples."""
    writers: dict[str, pq.ParquetWriter] = {}
    counts: Counter[str] = Counter()
    first_row = verified = 0
    source_records = []
    try:
        for record in records:
            name = Path(record["uri"]).name
            path = corpus_root / split / name
            identity = json.loads(path.with_suffix(".source.json").read_text())
            if identity["uri"] != record["uri"] or identity["etag"] != record["etag"]:
                raise ValueError("Cached source identity differs")
            dna = pq.read_table(path, columns=["seq"])
            meta_path = metadata_root / split / name
            meta = pq.read_table(meta_path)
            if len(meta) != record["rows"] or len(dna) != len(meta):
                raise ValueError("Sequence and metadata row counts differ")
            if (
                not np.array_equal(
                    meta["row"].to_numpy(), np.arange(first_row, first_row + len(meta))
                )
                or pc.any(pc.not_equal(meta["split"], split)).as_py()
            ):
                raise ValueError("Source metadata order/split differs")
            sequences = dna["seq"]
            if pc.any(pc.not_equal(pc.utf8_length(sequences), 8192)).as_py():
                raise ValueError("Original DNA length changed")
            if pc.any(
                pc.match_substring_regex(sequences, "[^ACGTRYSWKMBDHVNacgtryswkmbdhvn]")
            ).as_py():
                raise ValueError("Unexpected original DNA alphabet")
            sample_path = meta_path.with_suffix(".sample.jsonl")
            for line in sample_path.read_text().splitlines():
                sample = json.loads(line)
                local = sample["row"] - first_row
                if sequences[local].as_py() != sample["seq"]:
                    raise ValueError("Pinned raw sequence/case differs at sampled row")
                verified += 1
            joined = meta.append_column("seq", sequences)
            codes = sorted(set(meta["assembly"].to_pylist()))
            for code in codes:
                selected = joined.filter(pc.equal(joined["assembly"], code))
                if code not in writers:
                    dest = output / split / f"{code}.parquet"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    writers[code] = pq.ParquetWriter(
                        dest, selected.schema, compression="zstd"
                    )
                writers[code].write_table(selected, row_group_size=4096)
                counts[code] += len(selected)
            source_records.append(
                {
                    "uri": record["uri"],
                    "etag": record["etag"],
                    "sequence_object_sha256": identity["sha256"],
                    "metadata_sha256": hashlib.sha256(
                        meta_path.read_bytes()
                    ).hexdigest(),
                }
            )
            first_row += len(meta)
            print(
                json.dumps(
                    {
                        "split": split,
                        "joined_rows": first_row,
                        "verified_raw_samples": verified,
                    }
                ),
                flush=True,
            )
    finally:
        for writer in writers.values():
            writer.close()
    if first_row != expected_rows:
        raise ValueError("Original split membership count differs")
    return {
        "split": split,
        "rows": first_row,
        "species": len(counts),
        "counts": dict(counts),
        "verified_raw_samples": verified,
        "sources": source_records,
    }


def main() -> None:
    """Build a local species-partitioned view solely for annotation processing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(
        files("exp593_dna").joinpath("assets/corpus.json").read_text()
    )
    reports = []
    for split, count in [("train", 2_638_656), ("validation", 329_832)]:
        reports.append(
            partition_split(
                split,
                inventory[split],
                args.corpus,
                args.metadata,
                args.output,
                expected_rows=count,
            )
        )
    (args.output / "partition_manifest.json").write_text(
        json.dumps(
            {"scope": "Existing original rows only", "splits": reports}, indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
