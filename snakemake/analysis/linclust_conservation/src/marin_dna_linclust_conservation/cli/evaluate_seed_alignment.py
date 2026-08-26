"""Evaluate alignment gates on graph pairs in truth-containing components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from marin_dna_linclust_conservation.seed_alignment import evaluate_seed_alignments
from marin_dna_linclust_conservation.smoke import parse_time_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--alignments", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--mmseqs-version", type=Path, required=True)
    parser.add_argument("--thresholds", type=json.loads, required=True)
    parser.add_argument("--pipeline-commit", required=True)
    parser.add_argument("--pipeline-config-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = evaluate_seed_alignments(
        truth_path=args.truth,
        pairs_path=args.pairs,
        alignments_path=args.alignments,
        thresholds=args.thresholds,
    )
    time_records = parse_time_report(args.resources)
    receipt.update(
        {
            "mmseqs_cpu_seconds": sum(
                float(record["user_seconds"]) + float(record["system_seconds"])
                for record in time_records
            ),
            "mmseqs_peak_rss_kib": max(
                int(record["maximum_rss_kib"]) for record in time_records
            ),
            "mmseqs_stage_resources": time_records,
            "mmseqs_version": args.mmseqs_version.read_text().strip(),
            "mmseqs_wall_seconds": sum(
                float(record["elapsed_seconds"]) for record in time_records
            ),
            "pipeline_commit": args.pipeline_commit,
            "pipeline_config_sha256": args.pipeline_config_sha256,
            "preparation": json.loads(args.preparation.read_text()),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
