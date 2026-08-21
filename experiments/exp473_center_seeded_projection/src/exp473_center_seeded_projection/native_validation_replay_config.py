"""Generate the isolated issue #473 native-validation replay config."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from exp473_center_seeded_projection.eval_config import validate_experiment_commit

VALIDATION_FILENAME = "data/validation/shard_0000.jsonl.zst"
VALIDATION_ROWS = 16_384
SOURCES: dict[str, dict[str, Any]] = {
    "cds_center_1": {
        "region": "cds",
        "repo": "marin-dna/vertebrate-v1-issue473-center1-cds",
        "revision": "4d9a04ab6c4a6e445345fe35fbe2be41b43e7938",
    },
    "enhancer_full_window": {
        "region": "ccre_enhancer_centered",
        "repo": "marin-dna/vertebrate-v1-issue473-fullwindow-ccre-enhancer-centered",
        "revision": "ffb9c63fae72311fb457640af9c8365b84f0edf8",
    },
    "enhancer_center_1": {
        "region": "ccre_enhancer_centered",
        "repo": "marin-dna/vertebrate-v1-issue473-center1-ccre-enhancer-centered",
        "revision": "23d1531f63998b5716e7895a74437e0568186bd1",
    },
}
CHECKPOINT_ROOTS = {
    "cds_center_1": (
        "gs://marin-us-east1/MarinDNA/exp473_center_seeded_projection/checkpoints/"
        "dna-exp473-0p25b-cds_center_1-v1/2026.08.20/hf"
    ),
    "enhancer_full_window": (
        "gs://marin-us-east1/MarinDNA/exp473_center_seeded_projection/checkpoints/"
        "dna-exp473-0p25b-enhancer_full_window-v1/2026.08.20/hf"
    ),
    "enhancer_center_1": (
        "gs://marin-us-east5/MarinDNA/exp473_center_seeded_projection/checkpoints/"
        "dna-exp473-0p25b-enhancer_center_1-v1/2026.08.20/hf"
    ),
}
NATIVE_WANDB_LOSSES = {
    ("cds_center_1", 2_000): 1.2911146879196167,
    ("cds_center_1", 4_999): 1.3029967546463013,
    ("enhancer_full_window", 1_500): 1.3206731081008911,
    ("enhancer_full_window", 4_999): 1.3403956890106201,
    ("enhancer_center_1", 1_500): 1.3208948373794556,
    ("enhancer_center_1", 4_999): 1.3436254262924194,
}


def build_native_validation_replay_config(*, diagnostic_commit: str) -> dict[str, Any]:
    """Build the six-cell minimum-vs-terminal replay matrix."""
    commit = validate_experiment_commit(diagnostic_commit)
    models: list[dict[str, Any]] = []
    for (arm, step), native_loss in NATIVE_WANDB_LOSSES.items():
        source = SOURCES[arm]
        models.append(
            {
                "name": f"{arm}-step-{step}",
                "arm": arm,
                "region": source["region"],
                "step": step,
                "gcs_path": f"{CHECKPOINT_ROOTS[arm]}/step-{step}",
                "native_wandb_loss": native_loss,
            }
        )
    return {
        "diagnostic_commit": commit,
        "purpose": "damage_control_native_validation_replay",
        "interpretation_allowed": False,
        "vep_held_out_access": False,
        "models": models,
        "sources": {
            arm: {
                **source,
                "filename": VALIDATION_FILENAME,
                "expected_rows": VALIDATION_ROWS,
            }
            for arm, source in SOURCES.items()
        },
        "results_root": f"results/issue473/{commit}/native_validation_replay",
        "inference": {
            "batch_size": 128,
            "num_workers": 4,
            "torch_compile": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic-commit", required=True)
    args = parser.parse_args()
    config = build_native_validation_replay_config(
        diagnostic_commit=args.diagnostic_commit
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"wrote {args.output}: {len(config['models'])} replay cells")


if __name__ == "__main__":
    main()
