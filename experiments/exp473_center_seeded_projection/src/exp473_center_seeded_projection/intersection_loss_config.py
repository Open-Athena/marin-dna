"""Generate the separate additive issue #473 intersection-loss config."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from exp473_center_seeded_projection.eval_config import (
    ARM_DATASETS,
    CHECKPOINT_STEPS,
    checkpoint_roots_from_env,
    model_name,
    validate_checkpoint_root,
    validate_experiment_commit,
)
from exp473_center_seeded_projection.intersection_loss import (
    ARM_CONTEXT,
    REGION_LABELS,
    SPLIT,
)

PRODUCER_COMMIT = "f764b7f1fa34ea730842117239dd179a7e3be572"
PRODUCER_CONFIG_SHA256 = (
    "bf8367c285f955407cfb2dba6102661b2e528261b64fe52095d52a688cd6d039"
)
PRODUCER_ROOT = (
    "s3://oa-bolinas/snakemake/vertebrate_projection_dataset/results/v1/"
    f"{PRODUCER_COMMIT}/{PRODUCER_CONFIG_SHA256}/full/"
    "experiments/473/fixed/full_scale/intersection"
)


def source_uri(region: str, policy: str) -> str:
    """Return one producer-pinned unlabeled chromosome-18 intersection view."""
    if region not in REGION_LABELS:
        raise ValueError(f"unknown region {region!r}")
    if policy not in {"full_window", "center_1"}:
        raise ValueError(f"unknown projection policy {policy!r}")
    return f"{PRODUCER_ROOT}/{REGION_LABELS[region]}/{policy}_validation.parquet"


def build_intersection_loss_config(
    checkpoint_roots: dict[str, str], *, experiment_commit: str
) -> dict[str, Any]:
    """Build the complete config for the isolated 40-cell loss workflow."""
    if set(checkpoint_roots) != set(ARM_DATASETS):
        raise ValueError(
            f"checkpoint roots must be exactly {sorted(ARM_DATASETS)}, "
            f"got {sorted(checkpoint_roots)}"
        )
    experiment_commit = validate_experiment_commit(experiment_commit)
    models: list[dict[str, Any]] = []
    for arm in ARM_DATASETS:
        root = validate_checkpoint_root(checkpoint_roots[arm])
        context = ARM_CONTEXT[arm]
        for step in CHECKPOINT_STEPS:
            models.append(
                {
                    "name": model_name(arm, step, experiment_commit=experiment_commit),
                    "arm": arm,
                    "region": context["region"],
                    "policy": context["policy"],
                    "step": step,
                    "gcs_path": f"{root}/hf/step-{step}",
                }
            )
    sources = {
        f"{region}_{policy}": {
            "region": region,
            "policy": policy,
            "uri": source_uri(region, policy),
        }
        for region in REGION_LABELS
        for policy in ("full_window", "center_1")
    }
    return {
        "experiment_commit": experiment_commit,
        "producer_commit": PRODUCER_COMMIT,
        "producer_config_sha256": PRODUCER_CONFIG_SHA256,
        "split": SPLIT,
        "vep_held_out_access": False,
        "models": models,
        "sources": sources,
        "results_root": f"results/issue473/{experiment_commit}/intersection_loss",
        "inference": {
            "batch_size": 128,
            "num_workers": 4,
            "torch_compile": True,
        },
        "analysis": {"n_bootstrap": 1_000, "seed": 473},
    }


def write_intersection_loss_config(config: dict[str, Any], output: Path) -> None:
    """Write deterministic YAML for the isolated Snakemake workflow."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-commit", required=True)
    args = parser.parse_args()
    config = build_intersection_loss_config(
        checkpoint_roots_from_env(), experiment_commit=args.experiment_commit
    )
    write_intersection_loss_config(config, args.output)
    print(
        f"wrote {args.output}: {len(config['models'])} paired-loss cells, "
        f"split={config['split']}, VEP held-out access=false"
    )


if __name__ == "__main__":
    main()
