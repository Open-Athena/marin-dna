"""Aggregate bounded real-data smoke metrics."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

from marin_dna_linclust_conservation.mmseqs import parse_cluster_assignments


def _parse_elapsed(value: str) -> float:
    fields = value.strip().split(":")
    if len(fields) == 2:
        minutes, seconds = fields
        return int(minutes) * 60 + float(seconds)
    assert len(fields) == 3, value
    hours, minutes, seconds = fields
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def parse_time_report(path: str | Path) -> list[dict[str, object]]:
    """Parse appended GNU time verbose records for MMseqs stages."""
    text = Path(path).read_text()
    records: list[dict[str, object]] = []
    for block in text.split("Command being timed:")[1:]:
        command_match = re.search(r'^\s*"([^"]+)"', block)
        rss_match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", block)
        elapsed_match = re.search(
            r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([^\n]+)",
            block,
        )
        user_match = re.search(r"User time \(seconds\):\s*([0-9.]+)", block)
        system_match = re.search(r"System time \(seconds\):\s*([0-9.]+)", block)
        assert (
            command_match
            and rss_match
            and elapsed_match
            and user_match
            and system_match
        )
        records.append(
            {
                "command": command_match.group(1),
                "elapsed_seconds": _parse_elapsed(elapsed_match.group(1)),
                "maximum_rss_kib": int(rss_match.group(1)),
                "system_seconds": float(system_match.group(1)),
                "user_seconds": float(user_match.group(1)),
            }
        )
    assert records, f"{path}: no GNU time records"
    return records


def summarize_smoke(
    *,
    stats_paths: Iterable[str | Path],
    assignments_path: str | Path,
    alignments_path: str | Path,
    fasta_path: str | Path,
    mmseqs_version_path: str | Path,
    resources_path: str | Path,
    temporary_bytes_path: str | Path,
) -> dict[str, object]:
    """Return the Phase 0 smoke receipt from per-assembly and MMseqs outputs."""
    per_assembly = [json.loads(Path(path).read_text()) for path in stats_paths]
    assert per_assembly
    accessions = [row["accession"] for row in per_assembly]
    assert len(accessions) == len(set(accessions))
    assignments = parse_cluster_assignments(assignments_path)
    cluster_sizes = assignments.group_by("representative").len()["len"]
    time_records = parse_time_report(resources_path)
    alignment_edges = sum(
        1 for line in Path(alignments_path).read_text().splitlines() if line
    )
    retained_windows = sum(int(row["retained_windows"]) for row in per_assembly)
    return {
        "accession_count": len(per_assembly),
        "alignment_edges": alignment_edges,
        "candidate_windows": sum(int(row["candidate_windows"]) for row in per_assembly),
        "cluster_count": assignments["representative"].n_unique(),
        "input_fasta_bytes": Path(fasta_path).stat().st_size,
        "mmseqs_cpu_seconds": sum(
            float(record["user_seconds"]) + float(record["system_seconds"])
            for record in time_records
        ),
        "mmseqs_peak_rss_kib": max(
            int(record["maximum_rss_kib"]) for record in time_records
        ),
        "mmseqs_stage_resources": time_records,
        "mmseqs_version": Path(mmseqs_version_path).read_text().strip(),
        "mmseqs_wall_seconds": sum(
            float(record["elapsed_seconds"]) for record in time_records
        ),
        "per_assembly": sorted(per_assembly, key=lambda row: row["accession"]),
        "release_gate_passed": True,
        "retained_bases": sum(int(row["retained_bases"]) for row in per_assembly),
        "retained_windows": retained_windows,
        "singleton_clusters": int((cluster_sizes == 1).sum()),
        "singleton_window_fraction": int((cluster_sizes == 1).sum()) / retained_windows,
        "temporary_bytes_after_mmseqs": int(
            Path(temporary_bytes_path).read_text().strip()
        ),
    }
