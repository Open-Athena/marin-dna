"""Download one staged 2bit and extract deterministic smoke-test windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path

import boto3

from marin_dna_linclust_conservation.sequence_report import (
    read_sequence_report,
    sample_tiled_intervals,
)
from marin_dna_linclust_conservation.staging import parse_s3_uri, sha256_file
from marin_dna_linclust_conservation.windows import RejectedWindow, classify_window


def _read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    sequence: list[str] = []
    with path.open() as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith(">"):
                if header is not None:
                    records.append((header, "".join(sequence)))
                header = stripped[1:]
                sequence = []
            elif stripped:
                assert header is not None, "FASTA sequence precedes first header"
                sequence.append(stripped)
    if header is not None:
        records.append((header, "".join(sequence)))
    return records


def _sequence_report_counts(path: Path, accession: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["assembly_accession"] != accession:
                continue
            counts["reported_sequences"] += 1
            if record.get("assembly_unit") == "Primary Assembly":
                counts["primary_assembly_sequences"] += 1
            if record.get("role") in {"alt-scaffold", "fix-patch", "novel-patch"}:
                counts["excluded_alt_or_patch_sequences"] += 1
            if record.get("assigned_molecule_location_type") == "Mitochondrion":
                counts["excluded_mitochondrial_sequences"] += 1
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accession", required=True)
    parser.add_argument("--sequence-report", type=Path, required=True)
    parser.add_argument("--staging-receipt", type=Path, required=True)
    parser.add_argument("--window-length", type=int, required=True)
    parser.add_argument("--stride", type=int, required=True)
    parser.add_argument("--candidate-count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-repeat-fraction", type=float, required=True)
    parser.add_argument("--output-fasta", type=Path, required=True)
    parser.add_argument("--output-stats", type=Path, required=True)
    args = parser.parse_args()

    receipt = json.loads(args.staging_receipt.read_text())
    assert receipt["accession"] == args.accession
    bucket, key = parse_s3_uri(receipt["destination_uri"])
    sequences = [
        sequence
        for sequence in read_sequence_report(args.sequence_report)
        if sequence.assembly_accession == args.accession
    ]
    assert sequences, f"{args.accession}: no eligible primary-assembly sequences"
    seed = int.from_bytes(
        hashlib.sha256(f"{args.seed}:{args.accession}".encode()).digest()[:8],
        "big",
    )
    intervals = sample_tiled_intervals(
        sequences,
        window_length=args.window_length,
        stride=args.stride,
        sample_size=args.candidate_count,
        seed=seed,
    )

    started = time.monotonic()
    with tempfile.TemporaryDirectory(
        prefix=f"linclust_sample_{args.accession}_"
    ) as directory:
        temporary = Path(directory)
        twobit = temporary / f"{args.accession}.2bit"
        download_started = time.monotonic()
        boto3.client("s3").download_file(bucket, key, str(twobit))
        download_seconds = time.monotonic() - download_started
        sequence_sha256 = sha256_file(twobit)
        sizes_result = subprocess.run(
            ["twoBitInfo", str(twobit), "stdout"],
            check=True,
            capture_output=True,
            text=True,
        )
        observed_sizes = {
            name: int(length)
            for name, length in (
                line.split("\t") for line in sizes_result.stdout.splitlines()
            )
        }
        for sequence in sequences:
            assert observed_sizes.get(sequence.sequence_accession) == sequence.length, (
                f"{args.accession}:{sequence.sequence_accession}: sequence-report "
                f"length {sequence.length} != 2bit length "
                f"{observed_sizes.get(sequence.sequence_accession)}"
            )

        bed = temporary / "sample.bed"
        with bed.open("w") as handle:
            for interval in intervals:
                handle.write(
                    f"{interval.sequence_accession}\t{interval.start}\t"
                    f"{interval.end}\t.\n"
                )
        extracted_fasta = temporary / "sample.raw.fasta"
        subprocess.run(
            [
                "twoBitToFa",
                str(twobit),
                str(extracted_fasta),
                f"-bed={bed}",
                "-bedPos",
            ],
            check=True,
        )
        extracted = _read_fasta(extracted_fasta)
        assert len(extracted) == len(intervals)

        rejections: Counter[str] = Counter()
        retained_repeat_fractions: list[float] = []
        args.output_fasta.parent.mkdir(parents=True, exist_ok=True)
        with args.output_fasta.open("w") as output:
            for (header, sequence), interval in zip(extracted, intervals, strict=True):
                expected_header = (
                    f"{interval.sequence_accession}:{interval.start}-{interval.end}"
                )
                assert header.split()[0] == expected_header, (header, expected_header)
                assert len(sequence) == args.window_length
                classified = classify_window(
                    sequence,
                    accession=args.accession,
                    sequence_name=interval.sequence_accession,
                    start=interval.start,
                    max_repeat_fraction=args.max_repeat_fraction,
                )
                if isinstance(classified, RejectedWindow):
                    rejections[classified.reason.value] += 1
                    continue
                retained_repeat_fractions.append(classified.repeat_fraction)
                output.write(f">{classified.record_id}\n{classified.sequence}\n")

        retained = len(retained_repeat_fractions)
        assert retained > 0, f"{args.accession}: smoke sampling retained no windows"
        stats: dict[str, object] = {
            "accession": args.accession,
            "candidate_windows": len(intervals),
            "eligible_sequences": len(sequences),
            "excluded_sequences_in_2bit": len(observed_sizes) - len(sequences),
            "rejections": dict(sorted(rejections.items())),
            "retained_bases": retained * args.window_length,
            "retained_fraction": retained / len(intervals),
            "retained_windows": retained,
            "repeat_fraction_max": max(retained_repeat_fractions),
            "repeat_fraction_mean": sum(retained_repeat_fractions) / retained,
            "sequence_sha256": sequence_sha256,
            "source_size_bytes": twobit.stat().st_size,
            "download_seconds": download_seconds,
            "wall_seconds": time.monotonic() - started,
            **_sequence_report_counts(args.sequence_report, args.accession),
        }
        args.output_stats.parent.mkdir(parents=True, exist_ok=True)
        args.output_stats.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
        print(f"{args.accession}: retained {retained}/{len(intervals)} smoke windows")


if __name__ == "__main__":
    main()
