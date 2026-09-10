"""Inspect completed publication payloads without uploading or reading credentials."""

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

EXPECTED = {
    "cds": 581256,
    "tss_utr5": 112740,
    "utr3": 131550,
    "ncrna": 56396,
    "enhancer": 228064,
}
PRODUCER = "6b1593c274a886d20f5c0ddf3712916d446f5fed"
PUBLISHER = "54b6f936467bbc657ae88753a6f24b768bd3363d"
GEOMETRY = re.compile(r"[ACGTNacgtn]{255}(?:\[SEQ\][ACGTNacgtn]{255}){0,39}\Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    base = Path("/opt/issue550/publication-assets")
    candidates = list(base.rglob("publication_provenance/*/release.json"))
    assert len(candidates) == 5, (
        f"Expected 5 finished manifests, found {len(candidates)}"
    )
    reports = {}
    for path in sorted(candidates):
        manifest = json.loads(path.read_text())
        region = manifest["region"]
        assert region in EXPECTED and region not in reports
        assert manifest["repo_id"] == f"marin-dna/rag-five-regions-v1-{region}"
        assert manifest["producer"]["pipeline_commit"] == PRODUCER
        assert manifest["publisher"]["pipeline_commit"] == PUBLISHER
        root = path.parents[2] / "publication" / region
        expected_files = {"README.md", "validation/00000-of-00001.parquet"}
        expected_files.update(f"train/{i:05d}-of-00016.parquet" for i in range(16))
        assert set(manifest["files"]) == expected_files
        assert {
            str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()
        } == expected_files
        split_rows = {"train": 0, "validation": 0}
        species_histograms = {split: {} for split in split_rows}
        segments_min, segments_max = 40, 0
        rows_with_n = 0
        for name, expected in manifest["files"].items():
            public_file = root / name
            assert public_file.stat().st_size == expected["bytes"]
            assert sha256(public_file) == expected["sha256"]
            if public_file.suffix != ".parquet":
                continue
            parquet = pq.ParquetFile(public_file)
            assert parquet.schema_arrow == pa.schema([("sequence", pa.string())])
            split = name.split("/")[0]
            split_rows[split] += parquet.metadata.num_rows
            for batch in parquet.iter_batches(batch_size=256):
                for sequence in batch.column(0).to_pylist():
                    assert isinstance(sequence, str) and GEOMETRY.fullmatch(sequence)
                    segments = sequence.count("[SEQ]") + 1
                    histogram = species_histograms[split]
                    histogram[segments] = histogram.get(segments, 0) + 1
                    segments_min = min(segments_min, segments)
                    segments_max = max(segments_max, segments)
                    rows_with_n += "N" in sequence or "n" in sequence
        assert split_rows == {"train": EXPECTED[region], "validation": 400}
        assert all(split_rows[s] == manifest["splits"][s]["rows"] for s in split_rows)
        token_statistics = {}
        for split, histogram in species_histograms.items():
            assert sum(histogram.values()) == split_rows[split]
            unpadded = sum(
                256 * species * count for species, count in histogram.items()
            )
            allocated = 10240 * split_rows[split]
            token_statistics[split] = {
                "species_count_histogram": dict(sorted(histogram.items())),
                "unpadded_positions": unpadded,
                "allocated_positions": allocated,
                "loss_bearing_positions": unpadded - split_rows[split],
                "mean_species_per_document": unpadded / (256 * split_rows[split]),
                "padding_fraction": 1 - unpadded / allocated,
            }
        reports[region] = {
            "repo_id": manifest["repo_id"],
            "destination": f"https://huggingface.co/datasets/{manifest['repo_id']}",
            "public": True,
            "columns": {"sequence": "string"},
            "rows": split_rows,
            "species_segments_min": segments_min,
            "species_segments_max": segments_max,
            "rows_with_n": rows_with_n,
            "token_statistics": token_statistics,
            "expected_training_unpadded_positions": 4_000_000
            * token_statistics["train"]["unpadded_positions"]
            / split_rows["train"],
            "exposure_note": "Dataset totals are exact; training exposure is the expectation under four million draws per region, not an observed count of sampled tokens.",
            "payload_bytes": sum(f["bytes"] for f in manifest["files"].values()),
            "manifest_sha256": sha256(path),
            "producer": manifest["producer"],
            "publisher": manifest["publisher"],
            "files": manifest["files"],
        }
    output = {"checked_at": datetime.now(UTC).isoformat(), "datasets": reports}
    Path("/opt/issue550/release-payload-audit.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                k: {
                    a: b
                    for a, b in v.items()
                    if a not in {"files", "producer", "publisher"}
                }
                for k, v in reports.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
