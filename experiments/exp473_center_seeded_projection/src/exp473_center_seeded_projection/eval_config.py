"""Generate the additive isolated-evals_v2 config for issue #473.

The generated config contains the reused #417 CDS full-window baseline plus
the three new issue #473 model families and uses the official evaluator's
development split. It is intentionally emitted at run time because Marin
appends an immutable identity to each new checkpoint artifact root.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml

CHECKPOINT_STEPS = (*range(1_000, 5_000, 500), 4_999)
CDS_FULL_WINDOW_CHECKPOINT_ROOT = (
    "gs://marin-us-east5/checkpoints/"
    "dna-exp417-cds-combined-vertebrates-p255m-b2m-5k/2026.08.01"
)
GENOME_PATH = (
    "s3://oa-bolinas/data/genomes/homo_sapiens/GRCh38/ensembl-release-115/"
    "Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz"
)
DATASETS: tuple[dict[str, Any], ...] = (
    {
        "name": "mendelian_traits",
        "hf_revision": "4aed58e50c5dea0b878a665007af2ef9e5108e9f",
        "score_protocol": "minus_llr",
    },
    {
        "name": "complex_traits",
        "hf_revision": "22f86a89c65cb8f3007ac3cc2739f40efefa4340",
        "score_protocol": "abs_llr",
    },
    {
        "name": "sge",
        "hf_revision": "225d3d1ea32a4af547891b13c33b5e92a5aae849",
        "score_protocol": "minus_llr",
        "eval_protocol": "sge",
    },
)

ARM_DATASETS = {
    "cds_full_window": ("mendelian_traits", "sge"),
    "cds_center_1": ("mendelian_traits", "sge"),
    "enhancer_full_window": ("mendelian_traits", "complex_traits"),
    "enhancer_center_1": ("mendelian_traits", "complex_traits"),
}
REUSED_CHECKPOINT_ROOTS = {
    "cds_full_window": CDS_FULL_WINDOW_CHECKPOINT_ROOT,
}
ARM_ROOT_ENV = {
    arm: f"EXP473_{arm.upper()}_CHECKPOINT_ROOT"
    for arm in ARM_DATASETS
    if arm not in REUSED_CHECKPOINT_ROOTS
}


def validate_experiment_commit(experiment_commit: str) -> str:
    """Require the exact lowercase commit that owns evaluation outputs."""
    if len(experiment_commit) != 40 or any(
        character not in "0123456789abcdef" for character in experiment_commit
    ):
        raise ValueError("experiment_commit must be a full lowercase hexadecimal SHA")
    return experiment_commit


def model_name(arm: str, step: int, *, experiment_commit: str) -> str:
    """Commit-keyed evaluator name for one issue #473 checkpoint."""
    assert arm in ARM_DATASETS
    assert step in CHECKPOINT_STEPS
    commit = validate_experiment_commit(experiment_commit)
    return f"exp473-{commit}-{arm.replace('_', '-')}-step-{step}"


def validate_checkpoint_root(root: str) -> str:
    """Require one immutable GCS artifact root, before its ``hf`` directory."""
    normalized = root.strip().rstrip("/")
    if not normalized.startswith("gs://"):
        raise ValueError(f"checkpoint root must be a gs:// URI, got {root!r}")
    if normalized.endswith("/hf") or "/hf/step-" in normalized:
        raise ValueError(
            f"checkpoint root must stop before /hf; got checkpoint-like path {root!r}"
        )
    return normalized


def checkpoint_roots_from_env() -> dict[str, str]:
    """Combine the pinned #417 baseline with three new artifact roots."""
    roots = dict(REUSED_CHECKPOINT_ROOTS)
    for arm, variable in ARM_ROOT_ENV.items():
        value = os.environ.get(variable, "")
        if not value.strip():
            raise ValueError(f"missing required environment variable {variable}")
        roots[arm] = validate_checkpoint_root(value)
    return roots


def build_eval_config(
    checkpoint_roots: dict[str, str], *, experiment_commit: str
) -> dict[str, Any]:
    """Build a complete config accepted by the unchanged evals_v2 Snakefile."""
    if set(checkpoint_roots) != set(ARM_DATASETS):
        raise ValueError(
            f"checkpoint roots must be exactly {sorted(ARM_DATASETS)}, "
            f"got {sorted(checkpoint_roots)}"
        )
    baseline_root = validate_checkpoint_root(checkpoint_roots["cds_full_window"])
    if baseline_root != CDS_FULL_WINDOW_CHECKPOINT_ROOT:
        raise ValueError(
            "cds_full_window must reuse the exact #417 checkpoint root "
            f"{CDS_FULL_WINDOW_CHECKPOINT_ROOT}, got {baseline_root}"
        )
    experiment_commit = validate_experiment_commit(experiment_commit)

    models: list[dict[str, Any]] = []
    for arm, datasets in ARM_DATASETS.items():
        root = validate_checkpoint_root(checkpoint_roots[arm])
        for step in CHECKPOINT_STEPS:
            models.append(
                {
                    "name": model_name(arm, step, experiment_commit=experiment_commit),
                    "gcs_path": f"{root}/hf/step-{step}",
                    "window_size": 255,
                    "datasets": list(datasets),
                }
            )

    return {
        "input_hf_prefix": "bolinas-dna/evals",
        "genome_path": GENOME_PATH,
        # This is the invariant that prevents held-out access.
        "split": "train",
        "datasets": [dict(dataset) for dataset in DATASETS],
        "models": models,
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
        # The workflow's built-in config is loaded before command-line config.
        # Explicitly empty every off-rule model registry so unrelated base
        # models cannot leak into this experiment overlay under recursive merge.
        "nuc_dep": {"models": []},
        "umap_embeddings": {"models": []},
        "ll_gap": {"models": []},
        "probe": {
            "models": [],
            "min_variants": 300,
            "min_chroms": 3,
            "c_grid": [-12, 4, 17],
            "inner_splits": 5,
            "n_bootstrap": 1_000,
            "n_min": 30,
            "n_jobs": 4,
        },
        "issue_473": {
            "experiment_commit": experiment_commit,
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "held_out_access": False,
            "dataset_file": "train.parquet",
            "results_root": (
                f"results/issue473/{experiment_commit}/development_eval"
            ),
            "policy_comparison": "center_1_minus_full_window",
            "reused_checkpoint_roots": dict(REUSED_CHECKPOINT_ROOTS),
        },
    }


def write_eval_config(config: dict[str, Any], output: Path) -> None:
    """Write a deterministic YAML config for the official evaluator."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-commit", required=True)
    args = parser.parse_args()
    config = build_eval_config(
        checkpoint_roots_from_env(), experiment_commit=args.experiment_commit
    )
    write_eval_config(config, args.output)
    print(
        f"wrote {args.output}: {len(config['models'])} checkpoints, "
        f"split={config['split']}"
    )


if __name__ == "__main__":
    main()
