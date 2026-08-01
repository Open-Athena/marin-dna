"""Fetch and persist the current AlphaGenome human output-track metadata."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ASSAYS = {
    "ATAC": "atac",
    "DNASE": "dnase",
    "CHIP_TF": "chip_tf",
    "CHIP_HISTONE": "chip_histone",
    "CAGE": "cage",
    "PROCAP": "procap",
    "RNA_SEQ": "rna_seq",
}
EXPECTED_COUNTS = {
    "ATAC": 167,
    "DNASE": 305,
    "CHIP_TF": 1_617,
    "CHIP_HISTONE": 1_116,
    "CAGE": 546,
    "PROCAP": 12,
    "RNA_SEQ": 667,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _serializable(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in output.columns:
        if output[column].dtype == object:
            output[column] = output[column].map(
                lambda value: (
                    None
                    if pd.api.types.is_scalar(value) and pd.isna(value)
                    else str(value)
                )
            )
    return output


def fetch_metadata(*, output_dir: Path) -> dict[str, object]:
    assert not output_dir.exists()
    api_key = os.environ.get("ALPHA_GENOME_API_KEY")
    assert api_key, "ALPHA_GENOME_API_KEY is required"
    from alphagenome.models import dna_client

    model = dna_client.create(api_key)
    metadata = model.output_metadata(organism=dna_client.Organism.HOMO_SAPIENS)
    frames: list[pd.DataFrame] = []
    for assay, attribute in ASSAYS.items():
        frame = _serializable(getattr(metadata, attribute))
        assert len(frame) == EXPECTED_COUNTS[assay], (assay, len(frame))
        frame.insert(0, "assay_index", range(len(frame)))
        frame.insert(0, "assay", assay)
        frame.insert(
            0,
            "track_id",
            [f"{assay}_{index}" for index in range(len(frame))],
        )
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    assert len(combined) == sum(EXPECTED_COUNTS.values()) == 4_430
    assert combined["track_id"].is_unique
    output_dir.mkdir(parents=True, exist_ok=False)
    parquet_path = output_dir / "track_metadata.parquet"
    combined.to_parquet(parquet_path, index=False)
    result: dict[str, object] = {
        "created_at": datetime.now(UTC).isoformat(),
        "source": "AlphaGenome dna_model.output_metadata(HOMO_SAPIENS)",
        "alphagenome_version": importlib.metadata.version("alphagenome"),
        "counts": EXPECTED_COUNTS,
        "rows": len(combined),
        "artifact": {
            "path": parquet_path.name,
            "bytes": parquet_path.stat().st_size,
            "sha256": sha256(parquet_path),
        },
        "caveat": "Current metadata counts match the May 2026 score export, but the original export did not persist its metadata snapshot.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(fetch_metadata(output_dir=args.output_dir), indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
