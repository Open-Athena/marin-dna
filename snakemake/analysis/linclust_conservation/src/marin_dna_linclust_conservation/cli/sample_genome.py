"""Download one staged 2bit and extract deterministic sampled or exhaustive windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from itertools import batched
from pathlib import Path

import boto3

from marin_dna_linclust_conservation.sequence_report import (
    SampledInterval,
    is_primary_nuclear_record,
    iter_tiled_intervals,
    read_sequence_report,
    sample_tiled_intervals,
    tiled_interval_count,
)
from marin_dna_linclust_conservation.staging import download_staged_genome
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
            if is_primary_nuclear_record(record):
                counts["primary_assembly_sequences"] += 1
            if record.get("role") in {"alt-scaffold", "fix-patch", "novel-patch"}:
                counts["excluded_alt_or_patch_sequences"] += 1
            if record.get("assigned_molecule_location_type") == "Mitochondrion":
                counts["excluded_mitochondrial_sequences"] += 1
    return dict(counts)


def _interval_batches(
    intervals: Iterable[SampledInterval],
    *,
    batch_size: int,
) -> Iterator[tuple[SampledInterval, ...]]:
    assert batch_size > 0
    yield from batched(intervals, batch_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accession", required=True)
    parser.add_argument("--sequence-report", type=Path, required=True)
    parser.add_argument("--staging-receipt", type=Path, required=True)
    parser.add_argument("--window-length", type=int, required=True)
    parser.add_argument("--stride", type=int, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--candidate-count", type=int)
    selection.add_argument("--all-windows", action="store_true")
    parser.add_argument("--batch-size", type=int, default=250_000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-repeat-fraction", type=float, required=True)
    parser.add_argument("--output-fasta", type=Path, required=True)
    parser.add_argument("--output-stats", type=Path, required=True)
    args = parser.parse_args()

    receipt = json.loads(args.staging_receipt.read_text())
    assert receipt["accession"] == args.accession
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
    assert args.batch_size > 0
    if args.all_windows:
        candidate_count = tiled_interval_count(
            sequences,
            window_length=args.window_length,
            stride=args.stride,
        )
        intervals = iter_tiled_intervals(
            sequences,
            window_length=args.window_length,
            stride=args.stride,
        )
        selection_mode = "all"
    else:
        assert args.candidate_count is not None and args.candidate_count > 0
        candidate_count = args.candidate_count
        intervals = iter(
            sample_tiled_intervals(
                sequences,
                window_length=args.window_length,
                stride=args.stride,
                sample_size=candidate_count,
                seed=seed,
            )
        )
        selection_mode = "sample"

    started = time.monotonic()
    with tempfile.TemporaryDirectory(
        prefix=f"linclust_sample_{args.accession}_"
    ) as directory:
        temporary = Path(directory)
        twobit = temporary / f"{args.accession}.2bit"
        download_started = time.monotonic()
        sequence_sha256, source_size_bytes = download_staged_genome(
            receipt=receipt,
            destination_path=twobit,
            s3_client=boto3.client("s3"),
        )
        download_seconds = time.monotonic() - download_started
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

        rejections: Counter[str] = Counter()
        retained = 0
        repeat_fraction_max = 0.0
        repeat_fraction_sum = 0.0
        processed = 0
        bed = temporary / "windows.bed"
        extracted_fasta = temporary / "windows.raw.fasta"
        args.output_fasta.parent.mkdir(parents=True, exist_ok=True)
        with args.output_fasta.open("w") as output:
            for interval_batch in _interval_batches(
                intervals,
                batch_size=args.batch_size,
            ):
                with bed.open("w") as handle:
                    for interval in interval_batch:
                        handle.write(
                            f"{interval.sequence_accession}\t{interval.start}\t"
                            f"{interval.end}\t.\n"
                        )
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
                assert len(extracted) == len(interval_batch)
                for (header, sequence), interval in zip(
                    extracted,
                    interval_batch,
                    strict=True,
                ):
                    expected_header = (
                        f"{interval.sequence_accession}:{interval.start}-{interval.end}"
                    )
                    assert header.split()[0] == expected_header, (
                        header,
                        expected_header,
                    )
                    assert len(sequence) == args.window_length
                    classified = classify_window(
                        sequence,
                        accession=args.accession,
                        sequence_name=interval.sequence_accession,
                        start=interval.start,
                        max_repeat_fraction=args.max_repeat_fraction,
                    )
                    processed += 1
                    if isinstance(classified, RejectedWindow):
                        rejections[classified.reason.value] += 1
                        continue
                    retained += 1
                    repeat_fraction_max = max(
                        repeat_fraction_max,
                        classified.repeat_fraction,
                    )
                    repeat_fraction_sum += classified.repeat_fraction
                    output.write(f">{classified.record_id}\n{classified.sequence}\n")

        assert processed == candidate_count
        assert retained > 0, f"{args.accession}: extraction retained no windows"
        stats: dict[str, object] = {
            "accession": args.accession,
            "candidate_windows": candidate_count,
            "eligible_sequences": len(sequences),
            "excluded_sequences_in_2bit": len(observed_sizes) - len(sequences),
            "rejections": dict(sorted(rejections.items())),
            "retained_bases": retained * args.window_length,
            "retained_fraction": retained / candidate_count,
            "retained_windows": retained,
            "repeat_fraction_max": repeat_fraction_max,
            "repeat_fraction_mean": repeat_fraction_sum / retained,
            "selection_mode": selection_mode,
            "sequence_sha256": sequence_sha256,
            "source_size_bytes": source_size_bytes,
            "download_seconds": download_seconds,
            "wall_seconds": time.monotonic() - started,
            **_sequence_report_counts(args.sequence_report, args.accession),
        }
        args.output_stats.parent.mkdir(parents=True, exist_ok=True)
        args.output_stats.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
        print(f"{args.accession}: retained {retained}/{candidate_count} windows")


if __name__ == "__main__":
    main()
