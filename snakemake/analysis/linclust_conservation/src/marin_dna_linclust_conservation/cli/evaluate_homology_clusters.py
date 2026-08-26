"""Write truth-set metrics for one bounded Linclust configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from marin_dna_linclust_conservation.homology_evaluation import (
    evaluate_homology_clusters,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--alignments", type=Path, required=True)
    parser.add_argument("--configuration", type=json.loads, required=True)
    parser.add_argument("--mmseqs-version", type=Path, required=True)
    parser.add_argument("--pipeline-commit", required=True)
    parser.add_argument("--pipeline-config-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    receipt = evaluate_homology_clusters(
        truth_path=args.truth,
        assignments_path=args.assignments,
        alignments_path=args.alignments,
    )
    receipt.update(
        {
            "mmseqs_configuration": args.configuration,
            "mmseqs_version": args.mmseqs_version.read_text().strip(),
            "pipeline_commit": args.pipeline_commit,
            "pipeline_config_sha256": args.pipeline_config_sha256,
            "run_kind": "projected_homology_linclust_tuning",
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
