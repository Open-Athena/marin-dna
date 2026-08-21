"""Generate the isolated issue #417 exact-validation control config."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import yaml

FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
VALIDATION_FILENAME = "data/validation/shard_0000.jsonl.zst"
SOURCES = {
    "issue417_mammals_only": {
        "repo": "marin-dna/vertebrate-v1-cds_mammals_only",
        "revision": "d2bea760f6416775772699b821b266d3ae87245e",
        "filename": VALIDATION_FILENAME,
        "expected_rows": 16_384,
    },
    "issue417_combined": {
        "repo": "marin-dna/vertebrate-v1-cds",
        "revision": "bfab878078c4ee6c0f47b760f1e5e0577549dc9d",
        "filename": VALIDATION_FILENAME,
        "expected_rows": 16_384,
    },
}
CHECKPOINT_ROOTS = {
    "issue417_mammals_only": (
        "gs://marin-us-east5/checkpoints/"
        "dna-exp417-cds-mammals-only-p255m-b2m-5k/2026.08.01/hf"
    ),
    "issue417_combined": (
        "gs://marin-us-east5/checkpoints/"
        "dna-exp417-cds-combined-vertebrates-p255m-b2m-5k/2026.08.01/hf"
    ),
}
STEPS = {
    "issue417_mammals_only": (
        500,
        1000,
        1500,
        2000,
        2500,
        3000,
        3500,
        4000,
        4500,
        4999,
    ),
    "issue417_combined": (1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 4999),
}


def build_issue417_validation_control_config(
    *, diagnostic_commit: str
) -> dict[str, Any]:
    commit = diagnostic_commit.strip()
    if not FULL_SHA_RE.fullmatch(commit):
        raise ValueError("diagnostic_commit must be a full lowercase commit SHA")
    models = [
        {
            "name": f"{arm}-step-{step}",
            "arm": arm,
            "step": step,
            "gcs_path": f"{CHECKPOINT_ROOTS[arm]}/step-{step}",
        }
        for arm, steps in STEPS.items()
        for step in steps
    ]
    assert len(models) == 19
    return {
        "purpose": "damage_control_issue417_validation_control",
        "interpretation_allowed": False,
        "vep_held_out_access": False,
        "diagnostic_commit": commit,
        "sources": SOURCES,
        "models": models,
        "inference": {
            "batch_size": 128,
            "num_workers": 4,
            "torch_compile": True,
        },
        "results_root": (f"results/issue473/{commit}/issue417_validation_control"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--diagnostic-commit", required=True)
    args = parser.parse_args()
    config = build_issue417_validation_control_config(
        diagnostic_commit=args.diagnostic_commit
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
