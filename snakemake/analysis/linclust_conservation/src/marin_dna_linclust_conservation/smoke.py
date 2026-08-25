"""Aggregate bounded real-data smoke metrics."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from marin_dna_linclust_conservation.mmseqs import (
    parse_alignments,
    parse_cluster_assignments,
    validate_alignment_coverage,
)


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


def fasta_record_ids(path: str | Path) -> list[str]:
    """Return unique record identifiers from a FASTA without loading sequences."""
    identifiers: list[str] = []
    with Path(path).open() as handle:
        for line in handle:
            if line.startswith(">"):
                identifier = line[1:].split(maxsplit=1)[0]
                assert identifier, f"{path}: empty FASTA identifier"
                identifiers.append(identifier)
    assert identifiers, f"{path}: no FASTA records"
    assert len(identifiers) == len(set(identifiers)), f"{path}: duplicate FASTA IDs"
    return identifiers


def validate_release_gate(
    *,
    path: str | Path,
    expected_mmseqs_version: str,
    expected_configuration: dict[str, Any],
    pipeline_commit: str,
    pipeline_config_sha256: str,
) -> dict[str, object]:
    """Validate the complete synthetic gate used by a real-data smoke."""
    receipt = json.loads(Path(path).read_text())
    assert receipt["release_gate_passed"] is True
    assert receipt["assignment_partition_stable"] is True
    assert receipt["mmseqs_version"] == expected_mmseqs_version
    assert receipt["configuration"] == expected_configuration
    assert receipt["pipeline_commit"] == pipeline_commit
    assert receipt["pipeline_config_sha256"] == pipeline_config_sha256
    runs = receipt["runs"]
    assert runs
    assert all(run["reverse_complement_alignment_verified"] is True for run in runs)
    assert all(run["alignment_count"] == run["record_count"] for run in runs)
    return receipt


def summarize_smoke(
    *,
    stats_paths: Iterable[str | Path],
    assignments_path: str | Path,
    alignments_path: str | Path,
    fasta_path: str | Path,
    mmseqs_version_path: str | Path,
    resources_path: str | Path,
    peak_temporary_bytes_path: str | Path,
    release_gate_path: str | Path,
    expected_mmseqs_version: str,
    expected_configuration: dict[str, Any],
    pipeline_commit: str,
    pipeline_config_sha256: str,
) -> dict[str, object]:
    """Return the Phase 0 smoke receipt from per-assembly and MMseqs outputs."""
    per_assembly = [json.loads(Path(path).read_text()) for path in stats_paths]
    assert per_assembly
    accessions = [row["accession"] for row in per_assembly]
    assert len(accessions) == len(set(accessions))
    assignments = parse_cluster_assignments(assignments_path)
    alignments = parse_alignments(alignments_path)
    validate_alignment_coverage(assignments, alignments)
    record_ids = fasta_record_ids(fasta_path)
    assignment_members = assignments["member"].to_list()
    retained_windows = sum(int(row["retained_windows"]) for row in per_assembly)
    assert retained_windows == len(record_ids) == assignments.height
    assert set(record_ids) == set(assignment_members), (
        "FASTA identifiers do not exactly match cluster assignment members"
    )
    mmseqs_version = Path(mmseqs_version_path).read_text().strip()
    assert mmseqs_version == expected_mmseqs_version
    validate_release_gate(
        path=release_gate_path,
        expected_mmseqs_version=expected_mmseqs_version,
        expected_configuration=expected_configuration,
        pipeline_commit=pipeline_commit,
        pipeline_config_sha256=pipeline_config_sha256,
    )
    cluster_sizes = assignments.group_by("representative").len()["len"]
    time_records = parse_time_report(resources_path)
    alignment_edges = alignments.height
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
        "mmseqs_configuration": expected_configuration,
        "mmseqs_version": mmseqs_version,
        "mmseqs_wall_seconds": sum(
            float(record["elapsed_seconds"]) for record in time_records
        ),
        "per_assembly": sorted(per_assembly, key=lambda row: row["accession"]),
        "pipeline_commit": pipeline_commit,
        "pipeline_config_sha256": pipeline_config_sha256,
        "release_gate_passed": True,
        "retained_bases": sum(int(row["retained_bases"]) for row in per_assembly),
        "retained_windows": retained_windows,
        "singleton_clusters": int((cluster_sizes == 1).sum()),
        "singleton_window_fraction": int((cluster_sizes == 1).sum()) / retained_windows,
        "peak_temporary_bytes": int(
            Path(peak_temporary_bytes_path).read_text().strip()
        ),
    }
