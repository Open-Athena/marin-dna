"""Prepare a small UCSC region for the issue #419 GPU benchmark."""

from __future__ import annotations

import argparse
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from marin_dna.pipelines.chinchilla_logo import write_window_plan


UCSC_SEQUENCE_API = "https://api.genome.ucsc.edu/getData/sequence"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genome", default="GCF_000276665.1")
    parser.add_argument("--chrom", default="NW_004955402.1")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=2_000_000)
    parser.add_argument("--min-windows", type=int, default=8_192)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def fetch_ucsc_sequence(
    genome: str,
    chrom: str,
    start: int,
    end: int,
) -> tuple[str, str]:
    """Fetch one 0-based, half-open sequence interval from the UCSC API."""
    assert start == 0, "benchmark FASTA coordinates require a region starting at 0"
    assert end > start
    query = urllib.parse.urlencode(
        {"genome": genome, "chrom": chrom, "start": start, "end": end}
    )
    url = f"{UCSC_SEQUENCE_API}?{query}"
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
        payload: dict[str, Any] = json.load(response)

    assert payload["genome"] == genome
    assert payload["chrom"] == chrom
    assert int(payload["start"]) == start
    assert int(payload["end"]) == end
    sequence = str(payload["dna"])
    assert len(sequence) == end - start
    return sequence, url


def write_fasta(path: Path, chrom: str, sequence: str) -> None:
    """Write a single-record FASTA with deterministic 80-base lines."""
    assert sequence
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write(f">{chrom}\n")
        for offset in range(0, len(sequence), 80):
            handle.write(sequence[offset : offset + 80] + "\n")


def main() -> None:
    args = parse_args()
    assert args.start == 0
    assert args.end > args.start
    assert args.min_windows > 0
    assert re.fullmatch(r"[0-9a-f]{40}", args.commit_sha)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sequence, source_url = fetch_ucsc_sequence(
        args.genome,
        args.chrom,
        args.start,
        args.end,
    )
    fasta_path = args.output_dir / "region.fa"
    chrom_sizes_path = args.output_dir / "region.chrom.sizes"
    plan_path = args.output_dir / "plan.parquet"
    plan_metadata_path = args.output_dir / "plan.json"
    write_fasta(fasta_path, args.chrom, sequence)
    chrom_sizes_path.write_text(f"{args.chrom}\t{len(sequence)}\n")
    coverage = write_window_plan(
        fasta_path,
        chrom_sizes_path,
        args.chrom,
        plan_path,
        plan_metadata_path,
    )
    assert coverage.window_count >= args.min_windows, (
        f"region produced {coverage.window_count} windows, need {args.min_windows}"
    )

    manifest = {
        "coordinate_system": "0-based-half-open",
        "application_commit": args.commit_sha,
        "genome": args.genome,
        "chrom": args.chrom,
        "start": args.start,
        "end": args.end,
        "sequence_length": len(sequence),
        "source_url": source_url,
        "fasta": str(fasta_path),
        "plan": str(plan_path),
        "planned_windows": coverage.window_count,
        "minimum_benchmark_windows": args.min_windows,
    }
    (args.output_dir / "input.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
