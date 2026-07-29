"""Build the tiny enhancer dataset used by examples/train_tiny_dna.

This script intentionally does not upload anything. It downloads one immutable
source shard, copies its first 256 records, validates coordinate and sequence
invariants, and writes a deterministic JSONL artifact for later review/upload.
"""

import argparse
import hashlib
import io
import json
from itertools import islice
from pathlib import Path

import zstandard
from huggingface_hub import hf_hub_download

SOURCE_DATASET_ID = "marin-dna/zoonomia-v1-v3_ccre_non_promoter"
SOURCE_REVISION = "862485aa18eed53a53e693ba4c2eb45e0afc5087"
SOURCE_SHARD = "data/train/shard_0000.jsonl.zst"
NUM_ROWS = 256

REQUIRED_FIELDS = {
    "query_name",
    "species",
    "t_chrom",
    "t_start",
    "t_end",
    "t_strand",
    "t_src_size",
    "sequence",
    "augmentation",
}


def _read_source_rows(path: Path) -> list[dict[str, object]]:
    with path.open("rb") as compressed:
        decompressor = zstandard.ZstdDecompressor()
        with decompressor.stream_reader(compressed) as reader:
            with io.TextIOWrapper(reader, encoding="utf-8") as text:
                rows = [json.loads(line) for line in islice(text, NUM_ROWS)]
    assert len(rows) == NUM_ROWS, f"expected {NUM_ROWS} source rows, found {len(rows)}"
    return rows


def _validate_rows(rows: list[dict[str, object]]) -> None:
    assert len(rows) == NUM_ROWS
    for index, row in enumerate(rows):
        missing = REQUIRED_FIELDS - row.keys()
        assert not missing, f"row {index} is missing {sorted(missing)}"
        assert isinstance(row["t_start"], int)
        assert isinstance(row["t_end"], int)
        assert isinstance(row["t_src_size"], int)
        assert 0 <= row["t_start"] < row["t_end"] <= row["t_src_size"]
        assert row["t_end"] - row["t_start"] == 255
        assert row["t_strand"] in {"+", "-"}
        assert row["augmentation"] in {"+", "-"}
        assert isinstance(row["sequence"], str)
        assert len(row["sequence"]) == 255
        assert set(row["sequence"].upper()) <= {"A", "C", "G", "T", "N"}


def _write_jsonl(rows: list[dict[str, object]], output: Path) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with output.open("wb") as f:
        for row in rows:
            line = (
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode()
            f.write(line)
            digest.update(line)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scratch/issue412_tiny_enhancers/data/train.jsonl"),
    )
    args = parser.parse_args()

    source_path = Path(
        hf_hub_download(
            repo_id=SOURCE_DATASET_ID,
            repo_type="dataset",
            revision=SOURCE_REVISION,
            filename=SOURCE_SHARD,
        )
    )
    rows = _read_source_rows(source_path)
    _validate_rows(rows)
    digest = _write_jsonl(rows, args.output)

    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
