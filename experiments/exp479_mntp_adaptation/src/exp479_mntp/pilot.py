"""Resumable sequential execution of the three registered exp479 arms."""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from exp479_mntp.config import (
    BUDGET_USD,
    DATA_COMPONENTS,
    LAMBDA_GH200_PRICE_PER_HOUR_USD,
    TRAIN_STEPS,
)
from exp479_mntp.data import build_sequence_plan, plan_sha256
from exp479_mntp.module import Arm
from exp479_mntp.publishing import (
    assert_budget_reserve,
    download_latest_remote_checkpoint,
    initialize_model_repo,
    remote_arm_is_complete,
    remote_files,
    upload_run_file,
)
from exp479_mntp.train import train_arm

ARMS: tuple[Arm, ...] = ("transferred_mntp", "scratch_mntp", "clm_continuation")
EVALUATION_RESERVE_USD = 10.0


def selected_batch_size(preflight_path: Path, maximum: int | None = None) -> int:
    """Read the passing preflight batch, optionally applying a post-OOM cap."""

    payload: dict[str, Any] = json.loads(preflight_path.read_text(encoding="utf-8"))
    if payload.get("status") != "passed":
        raise RuntimeError(f"preflight did not pass: {payload.get('status')!r}")
    batch_size = int(payload["memory_and_throughput"]["selected"]["batch_size"])
    if batch_size <= 0:
        raise ValueError(f"preflight selected invalid batch size {batch_size}")
    if maximum is not None:
        if maximum <= 0:
            raise ValueError(f"maximum batch size must be positive, got {maximum}")
        batch_size = min(batch_size, maximum)
    return batch_size


def latest_local_checkpoint(output_dir: Path) -> Path | None:
    """Return the newest numbered full checkpoint, if this arm was interrupted."""

    checkpoints = sorted((output_dir / "checkpoints").glob("step-*.ckpt"))
    return checkpoints[-1] if checkpoints else None


def arm_is_complete(output_dir: Path) -> bool:
    """Check the compact completion manifest before deciding to skip an arm."""

    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return int(manifest.get("global_step", -1)) == TRAIN_STEPS


def assert_observed_budget_projection(artifact_dir: Path) -> dict[str, float | int] | None:
    """Reproject remaining trained arms from completed-arm wall times."""

    durations: list[float] = []
    remaining_arms = 0
    for arm in ARMS:
        runtime_path = artifact_dir / arm / "runtime.json"
        if runtime_path.exists():
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
            durations.append(float(runtime["elapsed_seconds"]))
        else:
            remaining_arms += 1
    if not durations:
        return None
    start = float(os.environ["EXP479_INSTANCE_START_UNIX"])
    accrued_hours = max(0.0, (time.time() - start) / 3600)
    projected_remaining_hours = 1.10 * (sum(durations) / len(durations)) * remaining_arms / 3600
    projected_total_usd = (
        accrued_hours + projected_remaining_hours
    ) * LAMBDA_GH200_PRICE_PER_HOUR_USD + EVALUATION_RESERVE_USD
    result: dict[str, float | int] = {
        "completed_arms": len(durations),
        "remaining_arms": remaining_arms,
        "accrued_hours": accrued_hours,
        "projected_remaining_training_hours": projected_remaining_hours,
        "evaluation_reserve_usd": EVALUATION_RESERVE_USD,
        "projected_total_usd": projected_total_usd,
    }
    budget_path = artifact_dir / "budget-projection.json"
    budget_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if projected_total_usd >= BUDGET_USD:
        raise RuntimeError(
            f"observed exp479 projection ${projected_total_usd:.2f} exceeds ${BUDGET_USD:.2f} cap"
        )
    return result


def run_pilot(
    *,
    preflight_path: Path,
    artifact_dir: Path,
    hf_repo_id: str,
    model_card: Path,
    experiment_commit: str,
    seed: int,
    num_workers: int,
    offline_wandb: bool,
    maximum_batch_size: int | None = None,
) -> None:
    """Prepare matched plans, train each arm, and publish resumable artifacts."""

    assert_budget_reserve()
    batch_size = selected_batch_size(preflight_path, maximum_batch_size)
    initialize_model_repo(
        repo_id=hf_repo_id,
        card_template=model_card,
        experiment_commit=experiment_commit,
        private=True,
    )
    published_files = remote_files(hf_repo_id)

    data_dir = artifact_dir / "data"
    train_plan = data_dir / "train.jsonl"
    validation_plan = data_dir / "validation.jsonl"
    if not train_plan.exists():
        total_samples = TRAIN_STEPS * batch_size
        if total_samples % len(DATA_COMPONENTS) != 0:
            raise ValueError("training exposure must divide evenly across five components")
        build_sequence_plan(
            train_plan,
            samples_per_component=total_samples // len(DATA_COMPONENTS),
            seed=seed,
            validation=False,
        )
    if not validation_plan.exists():
        build_sequence_plan(
            validation_plan,
            samples_per_component=128,
            seed=seed + 10_000,
            validation=True,
        )
    data_manifest = data_dir / "manifest.json"
    data_manifest.write_text(
        json.dumps(
            {
                "batch_size": batch_size,
                "train_rows": TRAIN_STEPS * batch_size,
                "validation_rows": 128 * len(DATA_COMPONENTS),
                "train_sha256": plan_sha256(train_plan),
                "validation_sha256": plan_sha256(validation_plan),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    upload_run_file(
        local_path=preflight_path,
        path_in_repo="runs/preflight.json",
        repo_id=hf_repo_id,
        commit_message="Upload exp479 GH200 preflight",
    )
    upload_run_file(
        local_path=data_manifest,
        path_in_repo="runs/data-manifest.json",
        repo_id=hf_repo_id,
        commit_message="Upload exp479 sequence-plan hashes",
    )

    for arm in ARMS:
        output_dir = artifact_dir / arm
        if arm_is_complete(output_dir) or remote_arm_is_complete(published_files, arm):
            continue
        assert_observed_budget_projection(artifact_dir)
        assert_budget_reserve()
        resume_from = latest_local_checkpoint(output_dir)
        if resume_from is None:
            resume_from = download_latest_remote_checkpoint(
                repo_id=hf_repo_id,
                arm=arm,
                destination_dir=artifact_dir / "downloaded-checkpoints",
                files=published_files,
            )
        train_arm(
            arm=arm,
            batch_size=batch_size,
            train_plan=train_plan,
            validation_plan=validation_plan,
            output_dir=output_dir,
            seed=seed,
            num_workers=num_workers,
            resume_from=resume_from,
            offline_wandb=offline_wandb,
            accelerator="gpu",
            precision="bf16-mixed",
            hf_repo_id=hf_repo_id,
        )
        gc.collect()
        torch.cuda.empty_cache()
        published_files = remote_files(hf_repo_id)
        projection = assert_observed_budget_projection(artifact_dir)
        if projection is not None:
            upload_run_file(
                local_path=artifact_dir / "budget-projection.json",
                path_in_repo="runs/budget-projection.json",
                repo_id=hf_repo_id,
                commit_message="Update exp479 observed budget projection",
            )
