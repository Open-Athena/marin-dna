"""Build a receipt for an exhaustive clustering-only sensitivity run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from marin_dna_linclust_conservation.cluster_summary import (
    summarize_assignment_handle,
)
from marin_dna_linclust_conservation.smoke import (
    parse_time_report,
    validate_release_gate,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignments", required=True)
    parser.add_argument("--compressed-assignments", type=Path, required=True)
    parser.add_argument("--stats", type=Path, nargs="+", required=True)
    parser.add_argument("--fasta", type=Path, nargs="+", required=True)
    parser.add_argument("--mmseqs-version", type=Path, required=True)
    parser.add_argument("--expected-mmseqs-version", required=True)
    parser.add_argument("--configuration", type=json.loads, required=True)
    parser.add_argument("--release-gate-configuration", type=json.loads)
    parser.add_argument("--release-gate", type=Path)
    parser.add_argument("--pipeline-commit", required=True)
    parser.add_argument("--pipeline-config-sha256", required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--peak-temporary-bytes", type=Path, required=True)
    parser.add_argument("--run-kind", default="exhaustive_clustering_sensitivity")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    per_assembly = [json.loads(path.read_text()) for path in args.stats]
    accessions = [str(row["accession"]) for row in per_assembly]
    assert len(accessions) == len(set(accessions))
    assert all(row["selection_mode"] == "all" for row in per_assembly)
    expected_member_counts = {
        str(row["accession"]): int(row["retained_windows"]) for row in per_assembly
    }
    if args.assignments == "-":
        summary = summarize_assignment_handle(
            sys.stdin,
            expected_accessions=set(accessions),
        )
    else:
        with Path(args.assignments).open() as handle:
            summary = summarize_assignment_handle(
                handle,
                expected_accessions=set(accessions),
            )
    assert summary.member_count_by_accession == dict(
        sorted(expected_member_counts.items())
    )
    retained_windows = sum(expected_member_counts.values())
    assert summary.assignment_count == retained_windows

    mmseqs_version = args.mmseqs_version.read_text().strip()
    assert mmseqs_version == args.expected_mmseqs_version
    expected_configuration: dict[str, Any] = args.configuration
    release_gate_configuration = None
    release_gate_passed = None
    if args.release_gate is not None:
        release_gate_configuration = (
            args.release_gate_configuration or expected_configuration
        )
        validate_release_gate(
            path=args.release_gate,
            expected_mmseqs_version=args.expected_mmseqs_version,
            expected_configuration=release_gate_configuration,
            pipeline_commit=args.pipeline_commit,
            pipeline_config_sha256=args.pipeline_config_sha256,
        )
        release_gate_passed = True
    time_records = parse_time_report(args.resources)
    receipt = {
        "accession_count": len(accessions),
        "alignment_edges": None,
        "assignment_count": summary.assignment_count,
        "candidate_windows": sum(int(row["candidate_windows"]) for row in per_assembly),
        "cluster_count": summary.cluster_count,
        "cluster_size_bucket_histogram": summary.size_bucket_histogram,
        "compressed_assignments_bytes": args.compressed_assignments.stat().st_size,
        "cross_genome_cluster_count": summary.cross_genome_cluster_count,
        "cross_genome_member_count": summary.cross_genome_member_count,
        "distinct_genome_histogram": summary.distinct_genome_histogram,
        "input_fasta_bytes": sum(path.stat().st_size for path in args.fasta),
        "max_cluster_size": summary.max_cluster_size,
        "max_distinct_genomes": summary.max_distinct_genomes,
        "member_count_by_accession": summary.member_count_by_accession,
        "mmseqs_configuration": expected_configuration,
        "mmseqs_cpu_seconds": sum(
            float(record["user_seconds"]) + float(record["system_seconds"])
            for record in time_records
        ),
        "mmseqs_peak_rss_kib": max(
            int(record["maximum_rss_kib"]) for record in time_records
        ),
        "mmseqs_stage_resources": time_records,
        "mmseqs_version": mmseqs_version,
        "mmseqs_wall_seconds": sum(
            float(record["elapsed_seconds"]) for record in time_records
        ),
        "peak_temporary_bytes": int(args.peak_temporary_bytes.read_text().strip()),
        "per_assembly": sorted(per_assembly, key=lambda row: row["accession"]),
        "pipeline_commit": args.pipeline_commit,
        "pipeline_config_sha256": args.pipeline_config_sha256,
        "release_gate_passed": release_gate_passed,
        "release_gate_configuration": release_gate_configuration,
        "retained_bases": sum(int(row["retained_bases"]) for row in per_assembly),
        "retained_windows": retained_windows,
        "run_kind": args.run_kind,
        "singleton_cluster_count": summary.singleton_cluster_count,
        "singleton_window_fraction": summary.singleton_cluster_count / retained_windows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
