"""Build the additive terminal VEP config for the random-validation control."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from exp473_center_seeded_projection.eval_config import (
    DATASETS,
    GENOME_PATH,
    validate_experiment_commit,
)

TERMINAL_STEP = 4_999
RANDOM_CHECKPOINT_ROOT = (
    "gs://marin-us-east5/MarinDNA/exp473_center_seeded_projection/checkpoints/"
    "dna-exp473-0p25b-cds-fullwindow-random-val-v1/2026.08.21"
)
BASELINE_EXPERIMENT_COMMIT = "ae90f6d9e4b23ebe8fb1bd2314baa66cb82b37c1"
BASELINE_MODEL = (
    "exp473-ae90f6d9e4b23ebe8fb1bd2314baa66cb82b37c1-cds-full-window-step-4999"
)
BASELINE_RESULTS_ROOT = (
    f"results/issue473/{BASELINE_EXPERIMENT_COMMIT}/development_eval"
)
DATASET_NAMES = tuple(dataset["name"] for dataset in DATASETS)
RELEVANT_SUBSETS = {
    "mendelian_traits": (
        "missense_variant",
        "splicing",
        "synonymous_variant",
    ),
    "complex_traits": ("missense_variant",),
    "sge": ("missense_variant", "splicing"),
}


def model_name(snapshot_commit: str) -> str:
    """Return the commit-keyed name of the single new evaluation model."""
    commit = validate_experiment_commit(snapshot_commit)
    return f"exp473-{commit}-cds-full-window-random-validation-step-{TERMINAL_STEP}"


def build_random_validation_vep_config(snapshot_commit: str) -> dict[str, Any]:
    """Build one terminal-only development VEP comparison config."""
    commit = validate_experiment_commit(snapshot_commit)
    model = model_name(commit)
    results_root = f"results/issue473/{commit}/random_validation_vep"
    return {
        "input_hf_prefix": "bolinas-dna/evals",
        "genome_path": GENOME_PATH,
        "split": "train",
        "datasets": [dict(dataset) for dataset in DATASETS],
        "models": [
            {
                "name": model,
                "gcs_path": f"{RANDOM_CHECKPOINT_ROOT}/hf/step-{TERMINAL_STEP}",
                "window_size": 255,
                "datasets": list(DATASET_NAMES),
            }
        ],
        "inference": {
            "batch_size": 128,
            "num_workers": 4,
            "data_transform_on_the_fly": True,
            "torch_compile": True,
            "rc": True,
            "return_embeddings": False,
            "eval_accumulation_steps": None,
            "n_bootstrap": 1_000,
            "bootstrap_seed": 0,
        },
        "nuc_dep": {"models": []},
        "umap_embeddings": {"models": []},
        "ll_gap": {"models": []},
        "probe": {"models": []},
        "issue_473_random_validation_vep": {
            "snapshot_commit": commit,
            "held_out_access": False,
            "dataset_file": "train.parquet",
            "results_root": results_root,
            "new_model": model,
            "baseline_experiment_commit": BASELINE_EXPERIMENT_COMMIT,
            "baseline_model": BASELINE_MODEL,
            "baseline_results_root": BASELINE_RESULTS_ROOT,
            "relevant_subsets": {
                name: list(subsets) for name, subsets in RELEVANT_SUBSETS.items()
            },
            "paired_bootstrap": {
                "n_bootstrap": 1_000,
                "seed": 473,
                "unit": "match_group",
            },
        },
    }


def write_config(config: dict[str, Any], output: Path) -> None:
    """Write a deterministic YAML config."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot-commit", required=True)
    args = parser.parse_args()
    config = build_random_validation_vep_config(args.snapshot_commit)
    write_config(config, args.output)
    print(
        f"wrote {args.output}: {len(config['models'])} terminal checkpoint, "
        f"{len(DATASET_NAMES)} development score cells"
    )


if __name__ == "__main__":
    main()
