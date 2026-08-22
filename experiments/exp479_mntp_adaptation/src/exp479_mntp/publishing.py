"""Resumable Hugging Face publication for exp479 checkpoints and exports."""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lightning as L
from huggingface_hub import HfApi, hf_hub_download

from exp479_mntp.config import BUDGET_USD, LAMBDA_GH200_PRICE_PER_HOUR_USD


def assert_budget_reserve(reserve_usd: float = 2.0) -> None:
    """Reject a new upload when instance charges have reached the budget reserve."""

    raw_start = os.getenv("EXP479_INSTANCE_START_UNIX")
    prior = float(os.getenv("EXP479_PRIOR_COST_USD", "0"))
    price_per_hour = float(
        os.getenv("EXP479_INSTANCE_PRICE_PER_HOUR_USD", str(LAMBDA_GH200_PRICE_PER_HOUR_USD))
    )
    if raw_start is None:
        return
    accrued = prior + (time.time() - float(raw_start)) / 3600 * price_per_hour
    if accrued >= BUDGET_USD - reserve_usd:
        raise RuntimeError(
            f"refusing exp479 upload at accrued charge ${accrued:.2f}; "
            f"cap=${BUDGET_USD:.2f}, reserve=${reserve_usd:.2f}"
        )


def initialize_model_repo(
    *,
    repo_id: str,
    card_template: Path,
    experiment_commit: str,
    private: bool,
) -> None:
    """Create the model repository and upload the human-reviewed card."""

    if len(experiment_commit) != 40:
        raise ValueError("experiment_commit must be a full SHA")
    card = card_template.read_text(encoding="utf-8").replace(
        "<commit-pinned experiments/exp479_mntp_adaptation link>",
        "https://github.com/Open-Athena/marin-dna/tree/"
        f"{experiment_commit}/experiments/exp479_mntp_adaptation",
    )
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    if api.file_exists(repo_id=repo_id, filename="README.md", repo_type="model"):
        return
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add the exp479 model-card draft",
    )


class CheckpointUploadCallback(L.Callback):
    """Upload each newly completed full Lightning checkpoint once."""

    def __init__(
        self,
        *,
        checkpoint_dir: Path,
        repo_id: str,
        arm: str,
        upload_steps: tuple[int, ...] | None = None,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.repo_id = repo_id
        self.arm = arm
        self.upload_steps = None if upload_steps is None else frozenset(upload_steps)
        self.uploaded_names: set[str] = set()

    def _upload_new(self) -> None:
        api = HfApi()
        for checkpoint in sorted(self.checkpoint_dir.glob("step-*.ckpt")):
            if checkpoint.name in self.uploaded_names:
                continue
            step = int(checkpoint.stem.rsplit("-", maxsplit=1)[1])
            if self.upload_steps is not None and step not in self.upload_steps:
                self.uploaded_names.add(checkpoint.name)
                continue
            path_in_repo = f"lightning/{self.arm}/{checkpoint.name}"
            if api.file_exists(
                repo_id=self.repo_id,
                filename=path_in_repo,
                repo_type="model",
            ):
                self.uploaded_names.add(checkpoint.name)
                continue
            assert_budget_reserve()
            api.upload_file(
                path_or_fileobj=checkpoint,
                path_in_repo=path_in_repo,
                repo_id=self.repo_id,
                repo_type="model",
                commit_message=f"Upload {self.arm} {checkpoint.stem}",
            )
            self.uploaded_names.add(checkpoint.name)

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del trainer, pl_module, outputs, batch, batch_idx
        self._upload_new()

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        del trainer, pl_module
        self._upload_new()

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        del trainer, pl_module
        self._upload_new()

    def on_exception(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        exception: BaseException,
    ) -> None:
        del trainer, pl_module, exception
        self._upload_new()

    def state_dict(self) -> dict[str, Any]:
        return {"uploaded_names": sorted(self.uploaded_names)}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.uploaded_names = set(state_dict.get("uploaded_names", ()))


def upload_final_arm(*, output_dir: Path, repo_id: str, arm: str) -> None:
    """Upload the cooled HF export and compact manifests for one arm."""

    assert_budget_reserve()
    api = HfApi()
    api.upload_folder(
        folder_path=output_dir / "hf" / "step-1000",
        path_in_repo=f"hf/{arm}/step-1000",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload cooled {arm} export",
    )
    for name in ("manifest.json", "runtime.json"):
        path = output_dir / name
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=f"runs/{arm}/{name}",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload {arm} {name}",
        )


def remote_files(repo_id: str) -> set[str]:
    """List current private staging files for restart decisions."""

    return set(HfApi().list_repo_files(repo_id=repo_id, repo_type="model"))


def upload_run_file(
    *,
    local_path: Path,
    path_in_repo: str,
    repo_id: str,
    commit_message: str,
) -> None:
    """Upload one compact run-contract file to the private staging repo."""

    assert_budget_reserve()
    HfApi().upload_file(
        path_or_fileobj=local_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )


def write_cost_estimate(*, artifact_dir: Path) -> Path:
    """Write the listed-price estimate immediately before Sky autodown."""

    artifact_dir.mkdir(parents=True, exist_ok=True)
    start_unix = float(os.environ["EXP479_INSTANCE_START_UNIX"])
    finish_unix = time.time()
    elapsed_hours = (finish_unix - start_unix) / 3600
    prior_cost = float(os.getenv("EXP479_PRIOR_COST_USD", "0"))
    price_per_hour = float(
        os.getenv("EXP479_INSTANCE_PRICE_PER_HOUR_USD", str(LAMBDA_GH200_PRICE_PER_HOUR_USD))
    )
    current_cost = elapsed_hours * price_per_hour
    payload = {
        "instance_start_utc": datetime.fromtimestamp(start_unix, UTC).isoformat(),
        "recorded_before_autodown_utc": datetime.fromtimestamp(finish_unix, UTC).isoformat(),
        "elapsed_hours": elapsed_hours,
        "listed_price_per_hour_usd": price_per_hour,
        "prior_estimated_list_cost_usd": prior_cost,
        "current_estimated_list_cost_usd": current_cost,
        "estimated_list_cost_usd": prior_cost + current_cost,
        "budget_cap_usd": BUDGET_USD,
        "note": "Pre-autodown listed-price estimate; reconcile against the provider bill.",
    }
    path = artifact_dir / "cost-estimate.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def publish_cost_estimate(*, artifact_dir: Path, repo_id: str) -> Path:
    """Publish the listed-price estimate immediately before Sky autodown."""

    path = write_cost_estimate(artifact_dir=artifact_dir)
    upload_run_file(
        local_path=path,
        path_in_repo="runs/cost-estimate.json",
        repo_id=repo_id,
        commit_message="Record exp479 pre-autodown cost estimate",
    )
    return path


def remote_arm_is_complete(files: set[str], arm: str) -> bool:
    """Recognize an arm only after both its manifest and final export exist."""

    return {
        f"runs/{arm}/manifest.json",
        f"hf/{arm}/step-1000/config.json",
    }.issubset(files)


def download_latest_remote_checkpoint(
    *,
    repo_id: str,
    arm: str,
    destination_dir: Path,
    files: set[str],
) -> Path | None:
    """Download the newest uploaded full checkpoint for cross-instance resume."""

    prefix = f"lightning/{arm}/step-"
    candidates = sorted(
        filename for filename in files if filename.startswith(prefix) and filename.endswith(".ckpt")
    )
    if not candidates:
        return None
    destination_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=repo_id,
        filename=candidates[-1],
        repo_type="model",
        local_dir=destination_dir,
    )
    return Path(path)
