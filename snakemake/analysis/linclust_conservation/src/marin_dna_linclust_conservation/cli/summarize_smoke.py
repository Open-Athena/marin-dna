"""Write the bounded real-data smoke receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from marin_dna_linclust_conservation.smoke import summarize_smoke


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", type=Path, nargs="+", required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--alignments", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--mmseqs-version", type=Path, required=True)
    parser.add_argument("--expected-mmseqs-version", required=True)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--release-gate", type=Path, required=True)
    parser.add_argument("--pipeline-commit", required=True)
    parser.add_argument("--pipeline-config-sha256", required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--peak-temporary-bytes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = summarize_smoke(
        stats_paths=args.stats,
        assignments_path=args.assignments,
        alignments_path=args.alignments,
        fasta_path=args.fasta,
        mmseqs_version_path=args.mmseqs_version,
        resources_path=args.resources,
        peak_temporary_bytes_path=args.peak_temporary_bytes,
        release_gate_path=args.release_gate,
        expected_mmseqs_version=args.expected_mmseqs_version,
        expected_configuration=json.loads(args.configuration),
        pipeline_commit=args.pipeline_commit,
        pipeline_config_sha256=args.pipeline_config_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        f"smoke retained {receipt['retained_windows']} windows in "
        f"{receipt['cluster_count']} clusters"
    )


if __name__ == "__main__":
    main()
