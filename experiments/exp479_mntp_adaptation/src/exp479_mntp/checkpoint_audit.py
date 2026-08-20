"""Checkpoint trajectories and contract audits for issue 479."""

from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from pyfaidx import Fasta
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from exp479_mntp.callbacks import BudgetGuardCallback
from exp479_mntp.config import NUCLEOTIDE_LENGTH
from exp479_mntp.data import SequenceCollator, SequencePlanDataset, plan_sha256
from exp479_mntp.datamodule import ExperimentDataModule
from exp479_mntp.loss import IGNORE_INDEX, per_sequence_weighted_loss
from exp479_mntp.masking import sample_seed
from exp479_mntp.modeling import canonical_token_ids, load_model_bundle, model_logits
from exp479_mntp.module import AdaptationModule
from exp479_mntp.nucleotide_dependency import (
    LOCI,
    locus_window,
    orientation_dependency,
)
from exp479_mntp.publishing import assert_budget_reserve
from exp479_mntp.vep import (
    DATASETS,
    LoadedArm,
    _protocol_scores,
    attach_reference_windows,
    download_reference,
    load_variant_frame,
    reverse_complement,
    score_strand,
)
from exp479_mntp.vep_metrics import GLOBAL, MACRO, matched_metrics, sge_metrics

PRIMARY_REPO = "marin-dna/marin-dna-exp479-mntp-m5.1"
SPILLOVER_REPO = "gonzalobenegas/marin-dna-exp479-mntp-m5.1-spillover"
CHECKPOINT_REPOS = (PRIMARY_REPO, SPILLOVER_REPO)
EXPECTED_TRAIN_HASH = "9c715b08dad078c8ae5cf06325d4917051f52453f048674f6507ef6563130b91"
EXPECTED_VALIDATION_HASH = "35542611d71102479f3d07dc6565350120d1d89944e5a93f88efb641ece7e3ba"
REPLAY_STEPS = (0, 1, 5, 10, 25, 50, 100, 200, 400)
SCORE_PARITY_TOLERANCE = 2e-3
LOSS_PARITY_TOLERANCE = 2e-3

Objective = Literal["mntp", "clm"]
PointKind = Literal["source", "no_adaptation", "scratch_initial", "replay", "lightning", "hf"]


@dataclass(frozen=True)
class ModelPoint:
    """One independently loadable trajectory point."""

    point_id: str
    arm: str
    step: int
    objective: Objective
    kind: PointKind
    plot_series: str
    path: Path | None = None


class StepExportCallback(L.Callback):
    """Save lightweight Hugging Face exports at exact post-update steps."""

    def __init__(self, output_dir: Path, tokenizer: Any, steps: tuple[int, ...]) -> None:
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.steps = frozenset(steps)
        self.saved: set[int] = set()

    def _save(self, trainer: L.Trainer, pl_module: AdaptationModule) -> None:
        step = int(trainer.global_step)
        if step not in self.steps or step in self.saved:
            return
        destination = self.output_dir / f"step-{step:04d}"
        destination.mkdir(parents=True, exist_ok=True)
        pl_module.model.save_pretrained(destination, safe_serialization=True)
        self.tokenizer.save_pretrained(destination)
        (destination / "audit-export.json").write_text(
            json.dumps({"global_step": step, "objective": "clm"}, indent=2) + "\n",
            encoding="utf-8",
        )
        self.saved.add(step)

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: AdaptationModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del outputs, batch, batch_idx
        self._save(trainer, pl_module)


def assert_plan_contract(train_plan: Path, validation_plan: Path) -> None:
    """Require the exact plans used by the original three arms."""

    observed = {
        "train": plan_sha256(train_plan),
        "validation": plan_sha256(validation_plan),
    }
    expected = {"train": EXPECTED_TRAIN_HASH, "validation": EXPECTED_VALIDATION_HASH}
    if observed != expected:
        raise RuntimeError(
            f"sequence-plan hashes changed: observed={observed}, expected={expected}"
        )


def save_zero_update_export(output_dir: Path) -> None:
    """Serialize the source checkpoint without taking an optimizer step."""

    L.seed_everything(0, workers=True)
    bundle = load_model_bundle(
        initialization="transferred",
        add_mask=False,
        attention_implementation="sdpa",
    )
    destination = output_dir / "step-0000"
    destination.mkdir(parents=True, exist_ok=True)
    bundle.model.save_pretrained(destination, safe_serialization=True)
    bundle.tokenizer.save_pretrained(destination)
    (destination / "audit-export.json").write_text(
        json.dumps({"global_step": 0, "objective": "clm", "updates": 0}, indent=2) + "\n",
        encoding="utf-8",
    )
    del bundle
    gc.collect()


def replay_early_clm(
    *,
    train_plan: Path,
    validation_plan: Path,
    output_dir: Path,
    batch_size: int,
    num_workers: int,
) -> dict[str, object]:
    """Replay the first 400 causal updates with the original 1,000-step schedule."""

    assert_plan_contract(train_plan, validation_plan)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_zero_update_export(output_dir)
    L.seed_everything(0, workers=True)
    torch.set_float32_matmul_precision("high")
    bundle = load_model_bundle(
        initialization="transferred",
        add_mask=False,
        attention_implementation="sdpa",
    )
    bundle.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    bundle.model.config.use_cache = False
    module = AdaptationModule(
        model=bundle.model,
        arm="clm_continuation",
        batch_size=batch_size,
        record_gradient_norms=True,
    )
    data = ExperimentDataModule(
        train_plan=train_plan,
        validation_plan=validation_plan,
        tokenizer=bundle.tokenizer,
        objective="clm",
        canonical_token_ids=bundle.canonical_token_ids,
        mask_token_id=None,
        batch_size=batch_size,
        seed=0,
        num_workers=num_workers,
    )
    export = StepExportCallback(output_dir, bundle.tokenizer, REPLAY_STEPS[1:])
    started = time.perf_counter()
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        max_steps=400,
        max_epochs=-1,
        accumulate_grad_batches=1,
        val_check_interval=100,
        check_val_every_n_epoch=None,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        deterministic=True,
        default_root_dir=str(output_dir),
        logger=False,
        callbacks=[
            export,
            BudgetGuardCallback(
                instance_start_unix=(
                    None
                    if os.getenv("EXP479_INSTANCE_START_UNIX") is None
                    else float(os.environ["EXP479_INSTANCE_START_UNIX"])
                ),
                prior_cost_usd=float(os.getenv("EXP479_PRIOR_COST_USD", "0")),
            ),
        ],
        enable_checkpointing=False,
    )
    trainer.fit(module, datamodule=data)
    elapsed = time.perf_counter() - started
    if trainer.global_step != 400:
        raise RuntimeError(f"early CLM replay stopped at {trainer.global_step}, expected 400")
    missing = set(REPLAY_STEPS) - ({0} | export.saved)
    if missing:
        raise RuntimeError(f"early CLM replay omitted exports at steps {sorted(missing)}")
    trace = pd.DataFrame(module.gradient_norm_trace)
    expected_steps = np.arange(400)
    if len(trace) != 400 or not np.array_equal(trace["step"].to_numpy(), expected_steps):
        raise RuntimeError("early CLM gradient trace does not contain steps 0 through 399")
    numeric_columns = (
        "train_loss",
        "pre_clip_gradient_norm",
        "adamh_learning_rate",
        "adam_learning_rate",
    )
    if not np.isfinite(trace[list(numeric_columns)].to_numpy()).all():
        raise RuntimeError("early CLM gradient trace contains non-finite values")
    trace.to_csv(output_dir / "gradient-norm-trace.csv", index=False)
    norms = trace["pre_clip_gradient_norm"].to_numpy()
    losses = trace["train_loss"].to_numpy()
    post_warmup = losses[100:]
    post_median = float(np.median(post_warmup))
    post_mad = float(np.median(np.abs(post_warmup - post_median)))
    spike_threshold = post_median + 6 * 1.4826 * post_mad
    stability = {
        "n_steps": len(trace),
        "maximum_pre_clip_gradient_norm": float(norms.max()),
        "median_pre_clip_gradient_norm": float(np.median(norms)),
        "p95_pre_clip_gradient_norm": float(np.quantile(norms, 0.95)),
        "clipped_steps": int(trace["clipped"].sum()),
        "clip_fraction": float(trace["clipped"].mean()),
        "maximum_train_loss": float(losses.max()),
        "largest_positive_loss_delta": float(np.diff(losses).max()),
        "post_warmup_spike_threshold": spike_threshold,
        "post_warmup_spike_count": int((post_warmup > spike_threshold).sum()),
    }
    payload: dict[str, object] = {
        "global_step": int(trainer.global_step),
        "elapsed_seconds": elapsed,
        "export_steps": list(REPLAY_STEPS),
        "train_plan_sha256": plan_sha256(train_plan),
        "validation_plan_sha256": plan_sha256(validation_plan),
        "optimizer": module.optimizer_values.to_dict(),
        "stability": stability,
    }
    (output_dir / "replay-runtime.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    del trainer, data, module, bundle
    gc.collect()
    torch.cuda.empty_cache()
    return payload


def plot_training_stability(
    trace: pd.DataFrame,
    output_path: Path,
    *,
    clip_threshold: float,
) -> None:
    """Plot exact replay loss and pre-clipping global gradient norms."""

    figure, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    axes[0].plot(
        trace["step"],
        trace["train_loss"],
        color="#4C78A8",
        alpha=0.45,
        linewidth=0.8,
        label="Per-step loss",
    )
    axes[0].plot(
        trace["step"],
        trace["train_loss"].rolling(20, min_periods=1).mean(),
        color="#1F4E79",
        linewidth=1.6,
        label="20-step mean",
    )
    axes[0].set_ylabel("Training cross-entropy")
    axes[0].set_title("Exact continued-CLM replay")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        trace["step"],
        trace["pre_clip_gradient_norm"],
        color="#F58518",
        linewidth=1,
        label="Pre-clip global L2 norm",
    )
    axes[1].axhline(
        clip_threshold,
        color="#B22222",
        linestyle="--",
        linewidth=1,
        label=f"Clip threshold ({clip_threshold:.4f})",
    )
    axes[1].set_xlabel("Optimizer step")
    axes[1].set_ylabel("Gradient norm")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.savefig(output_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


TRAIN_STABILITY_RUN_IDS = {
    "transferred_mntp": "6iqcmdm7",
    "scratch_mntp": "4nstge1d",
    "clm_continuation": "yod8l3mb",
}
TRAIN_STABILITY_ARMS = (
    "transferred_mntp",
    "scratch_mntp",
    "clm_continuation",
)


def _original_training_loss(arm: str) -> pd.DataFrame:
    run_id = TRAIN_STABILITY_RUN_IDS[arm]
    run = wandb.Api().run(f"gonzalobenegas/marin/{run_id}")
    history = pd.DataFrame(
        run.scan_history(
            keys=["trainer/global_step", "train/loss"],
            page_size=1_000,
        )
    ).dropna()
    return history.rename(
        columns={
            "trainer/global_step": "step",
            "train/loss": "logged_train_loss",
        }
    )


def replay_training_stability_arm(
    *,
    arm: Literal["transferred_mntp", "scratch_mntp", "clm_continuation"],
    train_plan: Path,
    validation_plan: Path,
    root_dir: Path,
    batch_size: int,
    num_workers: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Replay one arm for 400 steps and record pre-clip norms."""

    assert_plan_contract(train_plan, validation_plan)
    L.seed_everything(0, workers=True)
    torch.set_float32_matmul_precision("high")
    objective = "clm" if arm == "clm_continuation" else "mntp"
    bundle = load_model_bundle(
        initialization="scratch" if arm == "scratch_mntp" else "transferred",
        add_mask=objective == "mntp",
        attention_implementation="sdpa",
    )
    bundle.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    bundle.model.config.use_cache = False
    module = AdaptationModule(
        model=bundle.model,
        arm=arm,
        batch_size=batch_size,
        record_gradient_norms=True,
    )
    data = ExperimentDataModule(
        train_plan=train_plan,
        validation_plan=validation_plan,
        tokenizer=bundle.tokenizer,
        objective=objective,
        canonical_token_ids=bundle.canonical_token_ids,
        mask_token_id=bundle.mask_token_id if objective == "mntp" else None,
        batch_size=batch_size,
        seed=0,
        num_workers=num_workers,
    )
    started = time.perf_counter()
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        max_steps=400,
        max_epochs=-1,
        accumulate_grad_batches=1,
        val_check_interval=100,
        check_val_every_n_epoch=None,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        deterministic=True,
        default_root_dir=str(root_dir / arm),
        logger=False,
        callbacks=[
            BudgetGuardCallback(
                instance_start_unix=(
                    None
                    if os.getenv("EXP479_INSTANCE_START_UNIX") is None
                    else float(os.environ["EXP479_INSTANCE_START_UNIX"])
                ),
                prior_cost_usd=float(os.getenv("EXP479_PRIOR_COST_USD", "0")),
            )
        ],
        enable_checkpointing=False,
    )
    trainer.fit(module, datamodule=data)
    elapsed = time.perf_counter() - started
    if trainer.global_step != 400:
        raise RuntimeError(f"{arm} stability replay stopped at {trainer.global_step}")
    trace = pd.DataFrame(module.gradient_norm_trace)
    expected_steps = np.arange(400)
    if len(trace) != 400 or not np.array_equal(trace["step"].to_numpy(), expected_steps):
        raise RuntimeError(f"{arm} gradient trace does not contain steps 0 through 399")
    numeric = trace[
        [
            "train_loss",
            "pre_clip_gradient_norm",
            "adamh_learning_rate",
            "adam_learning_rate",
        ]
    ].to_numpy()
    if not np.isfinite(numeric).all():
        raise RuntimeError(f"{arm} gradient trace contains non-finite values")
    trace.insert(0, "arm", arm)

    logged = _original_training_loss(arm)
    logged["step"] = logged["step"].astype(int)
    trace = trace.merge(logged, on="step", how="left", validate="one_to_one")
    if trace["logged_train_loss"].isna().any():
        raise RuntimeError(f"{arm} W&B history omits replayed training steps")
    trace["logged_abs_error"] = (trace["train_loss"] - trace["logged_train_loss"]).abs()
    maximum_loss_error = float(trace["logged_abs_error"].max())
    if maximum_loss_error > LOSS_PARITY_TOLERANCE:
        raise RuntimeError(f"{arm} replay loss differs from W&B by {maximum_loss_error}")

    norms = trace["pre_clip_gradient_norm"].to_numpy()
    losses = trace["train_loss"].to_numpy()
    post_warmup = losses[100:]
    post_median = float(np.median(post_warmup))
    post_mad = float(np.median(np.abs(post_warmup - post_median)))
    spike_threshold = post_median + 6 * 1.4826 * post_mad
    summary: dict[str, object] = {
        "arm": arm,
        "wandb_run_id": TRAIN_STABILITY_RUN_IDS[arm],
        "n_steps": len(trace),
        "elapsed_seconds": elapsed,
        "maximum_pre_clip_gradient_norm": float(norms.max()),
        "median_pre_clip_gradient_norm": float(np.median(norms)),
        "p95_pre_clip_gradient_norm": float(np.quantile(norms, 0.95)),
        "clipped_steps": int(trace["clipped"].sum()),
        "clip_fraction": float(trace["clipped"].mean()),
        "maximum_train_loss": float(losses.max()),
        "largest_positive_loss_delta": float(np.diff(losses).max()),
        "post_warmup_spike_threshold": spike_threshold,
        "post_warmup_spike_count": int((post_warmup > spike_threshold).sum()),
        "maximum_wandb_loss_abs_error": maximum_loss_error,
        "optimizer": module.optimizer_values.to_dict(),
    }
    del trainer, data, module, bundle
    gc.collect()
    torch.cuda.empty_cache()
    return trace, summary


def plot_training_stability_panel(
    trace: pd.DataFrame,
    output_path: Path,
    *,
    clip_threshold: float,
) -> None:
    """Plot loss and gradient norm for all three deterministic replays."""

    labels = {
        "transferred_mntp": "Transferred MNTP",
        "scratch_mntp": "Scratch MNTP",
        "clm_continuation": "Continued CLM",
    }
    figure, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    for row, arm in enumerate(TRAIN_STABILITY_ARMS):
        cell = trace[trace["arm"] == arm].sort_values("step")
        loss_axis = axes[row, 0]
        loss_axis.plot(
            cell["step"],
            cell["train_loss"],
            color="#4C78A8",
            alpha=0.4,
            linewidth=0.7,
            label="Per-step loss",
        )
        loss_axis.plot(
            cell["step"],
            cell["train_loss"].rolling(20, min_periods=1).mean(),
            color="#1F4E79",
            linewidth=1.5,
            label="20-step mean",
        )
        loss_axis.set_title(f"{labels[arm]} loss")
        loss_axis.set_ylabel("Cross-entropy")
        loss_axis.grid(alpha=0.25)
        loss_axis.legend(fontsize=7)

        gradient_axis = axes[row, 1]
        gradient_axis.plot(
            cell["step"],
            cell["pre_clip_gradient_norm"],
            color="#F58518",
            linewidth=0.9,
            label="Pre-clip global L2 norm",
        )
        gradient_axis.axhline(
            clip_threshold,
            color="#B22222",
            linestyle="--",
            linewidth=1,
            label=f"Clip threshold ({clip_threshold:.4f})",
        )
        gradient_axis.set_title(f"{labels[arm]} gradient norm")
        gradient_axis.set_ylabel("Global L2 norm")
        gradient_axis.grid(alpha=0.25)
        gradient_axis.legend(fontsize=7)
    axes[-1, 0].set_xlabel("Optimizer step")
    axes[-1, 1].set_xlabel("Optimizer step")
    figure.suptitle("Exact 400-step training-stability replays")
    figure.savefig(output_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_training_stability_audit(
    *,
    artifact_dir: Path,
    output_dir: Path,
    train_plan: Path,
    validation_plan: Path,
    hf_repo_id: str,
    batch_size: int,
    num_workers: int,
) -> None:
    """Replay all three arms and publish compact gradient/loss stability evidence."""

    if not torch.cuda.is_available():
        raise RuntimeError("training stability audit requires the Lambda GH200")
    if hf_repo_id not in CHECKPOINT_REPOS:
        raise ValueError(f"unexpected publication repository {hf_repo_id}")
    assert_plan_contract(train_plan, validation_plan)
    output_dir.mkdir(parents=True, exist_ok=True)
    traces: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for arm in TRAIN_STABILITY_ARMS:
        trace, summary = replay_training_stability_arm(
            arm=arm,
            train_plan=train_plan,
            validation_plan=validation_plan,
            root_dir=artifact_dir / "training-stability-replay",
            batch_size=batch_size,
            num_workers=num_workers,
        )
        traces.append(trace)
        summaries.append(summary)
    combined = pd.concat(traces, ignore_index=True)
    combined.to_csv(output_dir / "gradient-norm-trace.csv", index=False)
    summary_frame = pd.DataFrame(
        [{key: value for key, value in row.items() if key != "optimizer"} for row in summaries]
    )
    summary_frame.to_csv(output_dir / "stability-summary.csv", index=False)
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    clip_threshold = float(summaries[0]["optimizer"]["max_grad_norm"])  # type: ignore[index]
    plot_training_stability_panel(
        combined,
        figures / "training-stability",
        clip_threshold=clip_threshold,
    )

    wandb_url = None
    if os.getenv("WANDB_API_KEY"):
        run = wandb.init(
            project="marin",
            group="dna-exp479",
            name="dna-exp479-training-stability-audit",
            tags=["MNTP-479", "issue-479", "stability-audit", "gradient-norm"],
            config={
                "development_data": "unlabeled exact training and validation plans",
                "replay_steps_per_arm": 400,
                "clip_threshold": clip_threshold,
            },
        )
        run.log(
            {
                "gradient_norm_trace": wandb.Table(dataframe=combined),
                "stability_summary": wandb.Table(dataframe=summary_frame),
                "training_stability": wandb.Image(str(figures / "training-stability.png")),
            }
        )
        for summary in summaries:
            arm = str(summary["arm"])
            for key, value in summary.items():
                if key not in {"arm", "optimizer"}:
                    run.summary[f"{arm}/{key}"] = value
        wandb_url = run.get_url()
        run.finish(exit_code=0)

    manifest = {
        "train_plan_sha256": plan_sha256(train_plan),
        "validation_plan_sha256": plan_sha256(validation_plan),
        "arms": summaries,
        "wandb_url": wandb_url,
        "publication_path": "evaluation/training-stability-audit",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    assert_budget_reserve()
    HfApi().upload_folder(
        folder_path=output_dir,
        path_in_repo="evaluation/training-stability-audit",
        repo_id=hf_repo_id,
        repo_type="model",
        commit_message="Upload exp479 three-arm training stability audit",
    )


def _repo_files() -> dict[str, set[str]]:
    api = HfApi()
    return {
        repo: set(api.list_repo_files(repo_id=repo, repo_type="model")) for repo in CHECKPOINT_REPOS
    }


def _find_remote(files: dict[str, set[str]], filename: str) -> str:
    matches = [repo for repo, names in files.items() if filename in names]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one owner for {filename}, found {matches}")
    return matches[0]


def download_lightning_checkpoint(
    *,
    arm: str,
    step: int,
    cache_dir: Path,
    files: dict[str, set[str]],
) -> Path:
    """Download one original full checkpoint into a bounded local cache."""

    filename = f"lightning/{arm}/step-{step:04d}.ckpt"
    repo = _find_remote(files, filename)
    return Path(
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            repo_type="model",
            local_dir=cache_dir / repo.replace("/", "--"),
        )
    )


def download_final_export(
    *,
    arm: str,
    cache_dir: Path,
    files: dict[str, set[str]],
) -> Path:
    """Download one final Hugging Face export from its unique private owner."""

    config_name = f"hf/{arm}/step-1000/config.json"
    repo = _find_remote(files, config_name)
    root = cache_dir / repo.replace("/", "--")
    snapshot_download(
        repo_id=repo,
        repo_type="model",
        allow_patterns=[f"hf/{arm}/step-1000/*"],
        local_dir=root,
    )
    return root / "hf" / arm / "step-1000"


def _loaded_from_hf(path: Path, objective: Objective) -> LoadedArm:
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
    )
    return LoadedArm(
        model=model,
        tokenizer=tokenizer,
        canonical_ids=canonical_token_ids(tokenizer),
        mask_token_id=(int(tokenizer.mask_token_id) if objective == "mntp" else None),
    )


def _loaded_from_lightning(path: Path, arm: str, objective: Objective) -> LoadedArm:
    L.seed_everything(0, workers=True)
    bundle = load_model_bundle(
        initialization="scratch" if arm == "scratch_mntp" else "transferred",
        add_mask=objective == "mntp",
        attention_implementation="sdpa",
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    model_state = {
        key.removeprefix("model."): value
        for key, value in state_dict.items()
        if key.startswith("model.")
    }
    if len(model_state) != len(state_dict):
        unexpected = sorted(set(state_dict) - {f"model.{key}" for key in model_state})
        raise RuntimeError(f"checkpoint contains non-model state keys: {unexpected[:5]}")
    bundle.model.load_state_dict(model_state, strict=True)
    del checkpoint, state_dict, model_state
    return LoadedArm(
        model=bundle.model,
        tokenizer=bundle.tokenizer,
        canonical_ids=bundle.canonical_token_ids,
        mask_token_id=bundle.mask_token_id,
    )


def load_point(
    point: ModelPoint,
    *,
    cache_dir: Path,
    files: dict[str, set[str]],
) -> LoadedArm:
    """Load a point without retaining any prior point in accelerator memory."""

    if point.kind == "source":
        bundle = load_model_bundle(
            initialization="transferred",
            add_mask=False,
            attention_implementation="sdpa",
            dtype=torch.bfloat16,
        )
        return LoadedArm(bundle.model, bundle.tokenizer, bundle.canonical_token_ids, None)
    if point.kind == "no_adaptation":
        bundle = load_model_bundle(
            initialization="transferred",
            add_mask=True,
            attention_implementation="sdpa",
        )
        return LoadedArm(
            bundle.model,
            bundle.tokenizer,
            bundle.canonical_token_ids,
            bundle.mask_token_id,
        )
    if point.kind == "scratch_initial":
        L.seed_everything(0, workers=True)
        bundle = load_model_bundle(
            initialization="scratch",
            add_mask=True,
            attention_implementation="sdpa",
        )
        return LoadedArm(
            bundle.model,
            bundle.tokenizer,
            bundle.canonical_token_ids,
            bundle.mask_token_id,
        )
    if point.kind == "replay":
        if point.path is None:
            raise ValueError("replay point lacks a path")
        return _loaded_from_hf(point.path, point.objective)
    if point.kind == "lightning":
        checkpoint = download_lightning_checkpoint(
            arm=point.arm,
            step=point.step,
            cache_dir=cache_dir,
            files=files,
        )
        loaded = _loaded_from_lightning(checkpoint, point.arm, point.objective)
        checkpoint.unlink()
        return loaded
    if point.kind == "hf":
        path = download_final_export(arm=point.arm, cache_dir=cache_dir, files=files)
        return _loaded_from_hf(path, point.objective)
    raise ValueError(f"unknown point kind {point.kind}")


def _find_sample_id_for_position(position: int) -> int:
    """Find a deterministic single-mask sample ID selecting a nucleotide index."""

    if not 0 <= position < NUCLEOTIDE_LENGTH:
        raise ValueError(f"position outside 255-bp window: {position}")
    for sample_id in range(100_000):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(sample_seed(0, sample_id, stream=1))
        if int(torch.randint(NUCLEOTIDE_LENGTH, (), generator=generator)) == position:
            return sample_id
    raise RuntimeError(f"could not select nucleotide position {position}")


@torch.inference_mode()
def audit_alignment(point: ModelPoint, arm: LoadedArm) -> dict[str, object]:
    """Cross-check tokenizer, shifted labels, strands, and inference scores."""

    tokenizer = arm.tokenizer
    expected_special = {0, 1, 2} | ({7} if point.objective == "mntp" else set())
    if {int(value) for value in tokenizer.all_special_ids} != expected_special:
        raise RuntimeError(f"{point.point_id} special IDs changed: {tokenizer.all_special_ids}")
    if (tokenizer.pad_token_id, tokenizer.unk_token_id, tokenizer.bos_token_id) != (0, 1, 2):
        raise RuntimeError(f"{point.point_id} PAD/UNK/BOS IDs changed")
    if tokenizer.eos_token_id is not None:
        raise RuntimeError(f"{point.point_id} unexpectedly adds EOS")
    expected_vocab = 8 if point.objective == "mntp" else 7
    if len(tokenizer) != expected_vocab:
        raise RuntimeError(f"{point.point_id} tokenizer vocab is {len(tokenizer)}")
    if int(arm.model.get_input_embeddings().num_embeddings) != expected_vocab:
        raise RuntimeError(f"{point.point_id} input embedding vocab differs from tokenizer")
    output = arm.model.get_output_embeddings()
    if output is None or int(output.weight.shape[0]) != expected_vocab:
        raise RuntimeError(f"{point.point_id} output vocab differs from tokenizer")

    sequence = ("ACGT" * 64)[:NUCLEOTIDE_LENGTH]
    encoded = tokenizer(sequence, add_special_tokens=True, return_tensors="pt")
    ids = encoded["input_ids"]
    if ids.shape != (1, NUCLEOTIDE_LENGTH + 1):
        raise RuntimeError(f"{point.point_id} tokenized shape changed: {tuple(ids.shape)}")
    if int(ids[0, 0]) != 2 or int(encoded["attention_mask"].sum()) != 256:
        raise RuntimeError(f"{point.point_id} BOS or attention-mask contract changed")
    if tuple(sorted({int(value) for value in ids[0, 1:].tolist()})) != tuple(
        sorted(arm.canonical_ids)
    ):
        raise RuntimeError(f"{point.point_id} base-token contract changed")

    device = next(arm.model.parameters()).device
    positions = (0, 63, 127, 191, 254)
    max_score_error = 0.0
    label_checks = 0
    module_arm = "clm_continuation" if point.objective == "clm" else "transferred_mntp"
    module = AdaptationModule(model=arm.model, arm=module_arm, batch_size=2).eval()
    nucleotide_lookup = dict(zip("ACGT", arm.canonical_ids, strict=True))

    for variant_index in positions:
        ref = sequence[variant_index]
        alt = next(base for base in "ACGT" if base != ref)
        frame = pd.DataFrame({"sequence": [sequence], "ref": [ref], "alt": [alt]})
        for strand in ("fwd", "rc"):
            oriented = sequence if strand == "fwd" else reverse_complement(sequence)
            oriented_index = (
                variant_index if strand == "fwd" else NUCLEOTIDE_LENGTH - 1 - variant_index
            )
            oriented_ref = ref if strand == "fwd" else reverse_complement(ref)
            oriented_alt = alt if strand == "fwd" else reverse_complement(alt)
            if point.objective == "mntp":
                sample_id = _find_sample_id_for_position(oriented_index)
                collator = SequenceCollator(
                    tokenizer=tokenizer,
                    objective="mntp",
                    canonical_token_ids=arm.canonical_ids,
                    mask_token_id=arm.mask_token_id,
                    seed=0,
                    validation_mode="single",
                )
                batch = collator(
                    [{"sample_id": sample_id, "component": "audit", "sequence": oriented}]
                )
                selected = (batch["labels"][0] != IGNORE_INDEX).nonzero().flatten().tolist()
                if selected != [oriented_index]:
                    raise RuntimeError(
                        f"{point.point_id} {strand} target {oriented_index} aligned to {selected}"
                    )
                if int(batch["labels"][0, oriented_index]) != nucleotide_lookup[oriented_ref]:
                    raise RuntimeError(f"{point.point_id} {strand} true label changed")
                if int(batch["input_ids"][0, oriented_index + 1]) != arm.mask_token_id:
                    raise RuntimeError(f"{point.point_id} {strand} mask is off by one")
                moved = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = module(moved)[0, oriented_index, list(arm.canonical_ids)]
                logp = torch.log_softmax(logits.float(), dim=-1)
                index = {token: idx for idx, token in enumerate(arm.canonical_ids)}
                manual = float(
                    logp[index[nucleotide_lookup[oriented_alt]]]
                    - logp[index[nucleotide_lookup[oriented_ref]]]
                )
            else:
                alt_oriented = (
                    oriented[:oriented_index] + oriented_alt + oriented[oriented_index + 1 :]
                )
                collator = SequenceCollator(
                    tokenizer=tokenizer,
                    objective="clm",
                    canonical_token_ids=arm.canonical_ids,
                    mask_token_id=None,
                    seed=0,
                )
                batch = collator(
                    [
                        {"sample_id": 0, "component": "audit", "sequence": oriented},
                        {"sample_id": 1, "component": "audit", "sequence": alt_oriented},
                    ]
                )
                if not torch.equal(batch["labels"][:, :-1], batch["input_ids"][:, 1:]):
                    raise RuntimeError(f"{point.point_id} CLM labels are not shifted once")
                if not torch.all(batch["labels"][:, -1] == IGNORE_INDEX):
                    raise RuntimeError(f"{point.point_id} CLM final output is supervised")
                moved = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = module(moved)
                logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
                gathered = logp.gather(2, moved["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)
                manual = float(gathered[1].sum() - gathered[0].sum())
            inferred = float(
                score_strand(
                    arm,
                    frame,
                    objective=point.objective,
                    strand=strand,  # type: ignore[arg-type]
                    batch_size=1,
                    variant_index=variant_index,
                )[0]
            )
            max_score_error = max(max_score_error, abs(manual - inferred))
            label_checks += 1

    masked_base_invariance = None
    if point.objective == "mntp":
        center = NUCLEOTIDE_LENGTH // 2
        changed = (
            sequence[:center]
            + next(base for base in "ACGT" if base != sequence[center])
            + sequence[center + 1 :]
        )
        frame = pd.DataFrame(
            {
                "sequence": [sequence, changed],
                "ref": [sequence[center], sequence[center]],
                "alt": ["T" if sequence[center] != "T" else "A"] * 2,
            }
        )
        invariance_scores = score_strand(
            arm,
            frame,
            objective="mntp",
            strand="fwd",
            batch_size=2,
        )
        masked_base_invariance = float(abs(invariance_scores[0] - invariance_scores[1]))
        if masked_base_invariance > 1e-6:
            raise RuntimeError(f"{point.point_id} can see the true base under MASK")
    if max_score_error > 2e-4:
        raise RuntimeError(f"{point.point_id} training/inference alignment error {max_score_error}")
    del module
    return {
        "point_id": point.point_id,
        "objective": point.objective,
        "tokenized_length": int(ids.shape[1]),
        "vocab_size": expected_vocab,
        "special_ids": sorted(expected_special),
        "positions_per_strand_checked": list(positions),
        "label_checks": label_checks,
        "max_training_inference_score_abs_error": max_score_error,
        "masked_true_base_score_abs_error": masked_base_invariance,
        "passed": True,
    }


@torch.inference_mode()
def evaluate_validation_loss(
    point: ModelPoint,
    arm: LoadedArm,
    validation_plan: Path,
    *,
    batch_size: int,
) -> list[dict[str, object]]:
    """Recompute fixed-plan validation loss through the training collator."""

    dataset = SequencePlanDataset(validation_plan)
    modes = ("diffusion", "single") if point.objective == "mntp" else ("diffusion",)
    device = next(arm.model.parameters()).device
    rows: list[dict[str, object]] = []
    for mode in modes:
        collator = SequenceCollator(
            tokenizer=arm.tokenizer,
            objective=point.objective,
            canonical_token_ids=arm.canonical_ids,
            mask_token_id=arm.mask_token_id,
            seed=0,
            validation_mode=mode,  # type: ignore[arg-type]
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collator,
        )
        weighted_loss = 0.0
        weighted_accuracy = 0.0
        count = 0
        for batch in loader:
            assert_budget_reserve()
            moved = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model_logits(
                    arm.model,
                    input_ids=moved["input_ids"],
                    attention_mask=moved["attention_mask"],
                    attention_mode="full" if point.objective == "mntp" else "causal",
                )
                metrics = per_sequence_weighted_loss(
                    logits,
                    moved["labels"],
                    moved["loss_weights"],
                )
            size = len(batch["sample_ids"])
            weighted_loss += float(metrics.loss) * size
            weighted_accuracy += float(metrics.accuracy) * size
            count += size
        rows.append(
            {
                "point_id": point.point_id,
                "arm": point.arm,
                "plot_series": point.plot_series,
                "kind": point.kind,
                "step": point.step,
                "objective": point.objective,
                "validation_mode": "causal" if point.objective == "clm" else mode,
                "loss": weighted_loss / count,
                "accuracy": weighted_accuracy / count,
                "n_rows": count,
            }
        )
    return rows


def _endpoint_rows(
    point: ModelPoint,
    dataset_name: str,
    variants: pd.DataFrame,
    protocol_scores: pd.DataFrame,
    *,
    n_bootstrap: int,
) -> list[dict[str, object]]:
    if dataset_name in {"mendelian_traits", "complex_traits"}:
        metrics = matched_metrics(
            variants,
            protocol_scores,
            n_bootstrap=n_bootstrap,
            seed=0,
        )
        endpoint = MACRO if dataset_name == "mendelian_traits" else GLOBAL
        selected = metrics[metrics["subset"] == endpoint]
    else:
        metrics = sge_metrics(
            variants,
            protocol_scores,
            n_bootstrap=n_bootstrap,
            seed=0,
        )
        selected = metrics[(metrics["subset"] == MACRO) & (metrics["accession"] == MACRO)]
        endpoint = "accession_consequence_macro"
    rows: list[dict[str, object]] = []
    for metric in selected.itertuples(index=False):
        rows.append(
            {
                "point_id": point.point_id,
                "arm": point.arm,
                "plot_series": point.plot_series,
                "kind": point.kind,
                "step": point.step,
                "objective": point.objective,
                "dataset": dataset_name,
                "endpoint": endpoint,
                "orientation": metric.score_type,
                "auprc": float(metric.value),
                "se": float(metric.se),
                "n_rows": int(metric.n_rows if hasattr(metric, "n_rows") else metric.n),
            }
        )
    return rows


def evaluate_point(
    point: ModelPoint,
    arm: LoadedArm,
    frames: dict[str, pd.DataFrame],
    output_dir: Path,
    *,
    batch_size: int,
    n_bootstrap: int,
) -> list[dict[str, object]]:
    """Score both orientations and write private per-variant trajectory scores."""

    point_dir = output_dir / "scores" / point.point_id
    point_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, object]] = []
    for spec in DATASETS:
        frame = frames[spec.name]
        raw_fwd = score_strand(
            arm,
            frame,
            objective=point.objective,
            strand="fwd",
            batch_size=batch_size,
        )
        raw_rc = score_strand(
            arm,
            frame,
            objective=point.objective,
            strand="rc",
            batch_size=batch_size,
        )
        protocol_fwd = _protocol_scores(raw_fwd, spec.protocol)
        protocol_rc = _protocol_scores(raw_rc, spec.protocol)
        protocol_avg = _protocol_scores((raw_fwd + raw_rc) / 2, spec.protocol)
        private = pd.DataFrame(
            {
                "row_id": np.arange(len(frame)),
                "raw_llr_fwd": raw_fwd,
                "raw_llr_rc": raw_rc,
                "protocol_fwd": protocol_fwd,
                "protocol_rc": protocol_rc,
                "protocol_fwd_rc": protocol_avg,
            }
        )
        private.to_parquet(point_dir / f"{spec.name}.parquet", index=False)
        metric_rows.extend(
            _endpoint_rows(
                point,
                spec.name,
                frame,
                private[["protocol_fwd", "protocol_fwd_rc"]],
                n_bootstrap=n_bootstrap,
            )
        )
    return metric_rows


def coordinate_audit(
    frames: dict[str, pd.DataFrame],
    reference_path: Path,
) -> pd.DataFrame:
    """Independently verify 1-based variants against 0-based half-open slices."""

    rows: list[dict[str, object]] = []
    with Fasta(reference_path, as_raw=True, rebuild=False) as genome:
        for dataset, frame in frames.items():
            indices = sorted({0, len(frame) // 2, len(frame) - 1})
            selected = frame.iloc[indices].drop(columns=["sequence"])
            for variant_index in (63, 127, 191):
                attached = attach_reference_windows(
                    selected,
                    reference_path,
                    variant_index=variant_index,
                )
                for row in attached.itertuples(index=False):
                    center_zero_based = int(row.pos) - 1
                    start = center_zero_based - variant_index
                    end = start + NUCLEOTIDE_LENGTH
                    independently_sliced = str(genome[str(row.chrom)][start:end]).upper()
                    passed = (
                        independently_sliced == row.sequence
                        and row.sequence[variant_index] == row.ref
                        and end - start == NUCLEOTIDE_LENGTH
                    )
                    if not passed:
                        raise RuntimeError(
                            f"coordinate audit failed at {dataset} {row.chrom}:{row.pos}"
                        )
                    rows.append(
                        {
                            "dataset": dataset,
                            "chrom": str(row.chrom),
                            "position_1_based": int(row.pos),
                            "center_zero_based": center_zero_based,
                            "window_start_zero_based": start,
                            "window_end_zero_based_exclusive": end,
                            "variant_index_zero_based": variant_index,
                            "passed": True,
                        }
                    )
    return pd.DataFrame(rows)


def _score_file(output_dir: Path, point_id: str, dataset: str) -> Path:
    return output_dir / "scores" / point_id / f"{dataset}.parquet"


def compare_score_points(
    output_dir: Path,
    first_id: str,
    second_id: str,
    comparison: str,
) -> list[dict[str, object]]:
    """Compare raw checkpoint scores row-for-row."""

    rows: list[dict[str, object]] = []
    for spec in DATASETS:
        first = pd.read_parquet(_score_file(output_dir, first_id, spec.name))
        second = pd.read_parquet(_score_file(output_dir, second_id, spec.name))
        for column in ("raw_llr_fwd", "raw_llr_rc"):
            difference = np.abs(first[column].to_numpy() - second[column].to_numpy())
            maximum = float(difference.max())
            rows.append(
                {
                    "comparison": comparison,
                    "dataset": spec.name,
                    "score": column,
                    "n_rows": len(first),
                    "max_abs_error": maximum,
                    "mean_abs_error": float(difference.mean()),
                    "tolerance": SCORE_PARITY_TOLERANCE,
                    "passed": maximum <= SCORE_PARITY_TOLERANCE,
                }
            )
    return rows


def compare_existing_scores(
    *,
    output_dir: Path,
    point_id: str,
    existing_prefix: str,
    cache_dir: Path,
    files: dict[str, set[str]],
) -> list[dict[str, object]]:
    """Compare newly scored anchors with the original private VEP artifacts."""

    rows: list[dict[str, object]] = []
    for spec in DATASETS:
        filename = f"evaluation/vep/{spec.name}.scores.parquet"
        repo = _find_remote(files, filename)
        existing_path = hf_hub_download(
            repo_id=repo,
            filename=filename,
            repo_type="model",
            local_dir=cache_dir / repo.replace("/", "--"),
        )
        existing = pd.read_parquet(existing_path)
        current = pd.read_parquet(_score_file(output_dir, point_id, spec.name))
        for strand, current_column in (
            ("fwd", "protocol_fwd"),
            ("rc", "protocol_rc"),
        ):
            existing_column = f"{existing_prefix}_{strand}"
            difference = np.abs(
                current[current_column].to_numpy() - existing[existing_column].to_numpy()
            )
            maximum = float(difference.max())
            rows.append(
                {
                    "comparison": f"{point_id}_vs_original_evaluation",
                    "dataset": spec.name,
                    "score": strand,
                    "n_rows": len(current),
                    "max_abs_error": maximum,
                    "mean_abs_error": float(difference.mean()),
                    "tolerance": SCORE_PARITY_TOLERANCE,
                    "passed": maximum <= SCORE_PARITY_TOLERANCE,
                }
            )
    return rows


def attach_logged_loss_parity(losses: pd.DataFrame, logged_csv: Path) -> pd.DataFrame:
    """Join original checkpoints to their W&B history values and gate parity."""

    logged = pd.read_csv(logged_csv).set_index("step")
    columns = {
        ("transferred_mntp", "diffusion"): "transferred_diffusion_loss",
        ("transferred_mntp", "single"): "transferred_single_mask_loss",
        ("scratch_mntp", "diffusion"): "scratch_diffusion_loss",
        ("scratch_mntp", "single"): "scratch_single_mask_loss",
        ("clm_continuation", "causal"): "clm_diffusion_loss",
    }
    result = losses.copy()
    result["logged_loss"] = np.nan
    for index, row in result.iterrows():
        key = (str(row["arm"]), str(row["validation_mode"]))
        if row["kind"] not in {"lightning", "hf"} or key not in columns:
            continue
        if int(row["step"]) not in logged.index:
            continue
        result.loc[index, "logged_loss"] = float(logged.loc[int(row["step"]), columns[key]])
    result["logged_abs_error"] = (result["loss"] - result["logged_loss"]).abs()
    result["logged_parity_passed"] = result["logged_loss"].isna() | (
        result["logged_abs_error"] <= LOSS_PARITY_TOLERANCE
    )
    return result


def triangle_summary(matrix: np.ndarray, name: str) -> dict[str, object]:
    """Summarize dependency mass on either side of the diagonal."""

    upper = np.triu(matrix, k=1)
    lower = np.tril(matrix, k=-1)
    total = float(matrix.sum())
    return {
        "map": name,
        "upper_mass": float(upper.sum()),
        "lower_mass": float(lower.sum()),
        "upper_fraction": float(upper.sum() / total) if total else 0.0,
        "lower_fraction": float(lower.sum() / total) if total else 0.0,
        "upper_nonzero": int(np.count_nonzero(upper)),
        "lower_nonzero": int(np.count_nonzero(lower)),
        "maximum": float(matrix.max()),
    }


def plot_dependency_panel(
    maps: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    """Plot raw forward, raw registered RC, and registered FWD+RC maps."""

    column_titles = (
        "Raw reference-directed",
        "Raw RC-directed, aligned",
        "Registered FWD+RC",
    )
    figure, axes = plt.subplots(
        len(maps),
        3,
        figsize=(12, 3.1 * len(maps)),
        constrained_layout=True,
    )
    for row, (name, forward, reverse, combined) in enumerate(maps):
        maximum = float(max(forward.max(), reverse.max(), combined.max()))
        for column, matrix in enumerate((forward, reverse, combined)):
            image = axes[row, column].imshow(
                matrix,
                origin="lower",
                cmap="viridis",
                vmin=0,
                vmax=maximum,
                interpolation="nearest",
                rasterized=True,
            )
            if row == 0:
                axes[row, column].set_title(column_titles[column])
            if row == len(maps) - 1:
                axes[row, column].set_xlabel("Readout position")
            if column == 0:
                axes[row, column].set_ylabel(f"{name}\nSubstitution position")
        figure.colorbar(image, ax=axes[row, :], fraction=0.015, pad=0.01)
    figure.suptitle("Nucleotide dependency before and after orientation registration")
    figure.savefig(output_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(figure)


def run_dependency_audit(
    *,
    artifact_dir: Path,
    output_dir: Path,
    cache_dir: Path,
    files: dict[str, set[str]],
    batch_size: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Render all existing maps and recompute a causal-attention control."""

    maps: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    summary: list[dict[str, object]] = []
    existing_by_locus: dict[str, dict[str, np.ndarray]] = {}
    for locus in LOCI:
        filename = f"evaluation/nucleotide-dependency/{locus.name}.npz"
        repo = _find_remote(files, filename)
        path = hf_hub_download(
            repo_id=repo,
            filename=filename,
            repo_type="model",
            local_dir=cache_dir / repo.replace("/", "--"),
        )
        with np.load(path) as archive:
            forward = archive["forward_directed"].copy()
            reverse = archive["reverse_directed_forward_coordinates"].copy()
            combined = archive["fwd_rc"].copy()
        existing_by_locus[locus.name] = {
            "forward": forward,
            "reverse": reverse,
            "combined": combined,
        }
        maps.append((locus.name, forward, reverse, combined))
        for label, matrix in (
            ("forward_directed", forward),
            ("reverse_directed_forward_coordinates", reverse),
            ("registered_fwd_rc", combined),
        ):
            summary.append({"locus": locus.name, **triangle_summary(matrix, label)})
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_dependency_panel(maps, figures / "nucleotide-dependency-panel")

    final = ModelPoint(
        "transferred-final-nucdep",
        "transferred_mntp",
        1000,
        "mntp",
        "hf",
        "Transferred MNTP",
    )
    arm = load_point(final, cache_dir=cache_dir, files=files)
    arm.model.to(device="cuda", dtype=torch.bfloat16).eval()
    reference = download_reference(artifact_dir / "reference")
    locus = next(value for value in LOCI if value.name == "tRNA_Arg_TCT")
    with Fasta(reference, as_raw=True, rebuild=False) as genome:
        sequence, _ = locus_window(genome, locus)
    recomputed = orientation_dependency(
        arm,
        sequence,
        batch_size=batch_size,
        attention_mode="full",
    )
    causal = orientation_dependency(
        arm,
        sequence,
        batch_size=batch_size,
        attention_mode="causal",
    )
    existing = existing_by_locus[locus.name]["forward"]
    maximum_error = float(np.abs(recomputed - existing).max())
    causal_forbidden_max = float(np.tril(causal, k=-1).max())
    checks = [
        {
            "comparison": "tRNA_full_recompute_vs_original",
            "dataset": "tRNA_Arg_TCT",
            "score": "dependency",
            "n_rows": int(recomputed.size),
            "max_abs_error": maximum_error,
            "mean_abs_error": float(np.abs(recomputed - existing).mean()),
            "tolerance": SCORE_PARITY_TOLERANCE,
            "passed": maximum_error <= SCORE_PARITY_TOLERANCE,
        },
        {
            "comparison": "tRNA_causal_forbidden_triangle",
            "dataset": "tRNA_Arg_TCT",
            "score": "dependency",
            "n_rows": int(np.tril(causal, k=-1).size),
            "max_abs_error": causal_forbidden_max,
            "mean_abs_error": float(np.abs(np.tril(causal, k=-1)).mean()),
            "tolerance": 1e-6,
            "passed": causal_forbidden_max <= 1e-6,
        },
    ]
    np.savez_compressed(
        output_dir / "nucleotide-dependency-causal-control.npz",
        full_attention=recomputed,
        causal_attention=causal,
    )
    maximum = float(max(recomputed.max(), causal.max()))
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for axis, matrix, title in zip(
        axes,
        (recomputed, causal),
        ("Actual full attention", "Forced causal-attention control"),
        strict=True,
    ):
        image = axis.imshow(
            matrix,
            origin="lower",
            cmap="viridis",
            vmin=0,
            vmax=maximum,
            interpolation="nearest",
            rasterized=True,
        )
        axis.set_title(title)
        axis.set_xlabel("Readout position")
        axis.set_ylabel("Substitution position")
    figure.colorbar(image, ax=axes, label="L∞ change in A/C/G/T log probability")
    figure.suptitle("tRNA-Arg-TCT: bidirectional dependency and causal negative control")
    figure.savefig(
        figures / "nucleotide-dependency-causal-control.svg",
        format="svg",
        bbox_inches="tight",
    )
    figure.savefig(
        figures / "nucleotide-dependency-causal-control.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)
    summary.extend(
        [
            {"locus": locus.name, **triangle_summary(recomputed, "recomputed_full")},
            {"locus": locus.name, **triangle_summary(causal, "forced_causal")},
        ]
    )
    del arm
    gc.collect()
    torch.cuda.empty_cache()
    return pd.DataFrame(summary), checks


def plot_auprc_trajectories(metrics: pd.DataFrame, output_path: Path) -> None:
    """Plot primary AUPRC trajectories with uncertainty."""

    datasets = ("mendelian_traits", "complex_traits", "sge")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    selected = metrics[metrics["orientation"] == "protocol_fwd_rc"]
    styles = {
        "Continued CLM replay": ("#4C78A8", "o", "-"),
        "Continued CLM original": ("#4C78A8", "s", "--"),
        "Transferred MNTP": ("#E45756", "o", "-"),
        "Scratch MNTP": ("#72B7B2", "^", "-"),
        "Source CLM direct": ("#222222", "D", "None"),
        "Source CLM save/reload": ("#888888", "x", "None"),
    }
    for axis, dataset in zip(axes, datasets, strict=True):
        cell = selected[selected["dataset"] == dataset]
        for series, group in cell.groupby("plot_series", sort=False):
            color, marker, linestyle = styles.get(series, ("#999999", "o", "-"))
            group = group.sort_values("step")
            axis.errorbar(
                group["step"],
                group["auprc"],
                yerr=group["se"],
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.5,
                markersize=4,
                capsize=2,
                label=series,
            )
        axis.set_xscale("symlog", linthresh=10)
        axis.set_xlabel("Optimizer steps (symlog; linear through 10)")
        axis.set_ylabel("AUPRC")
        axis.set_title(dataset.replace("_", " ").title())
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
    figure.suptitle("Issue 479 primary odd-autosome/X VEP trajectories")
    figure.savefig(output_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_loss_trajectories(
    losses: pd.DataFrame,
    logged_csv: Path,
    output_path: Path,
) -> None:
    """Overlay W&B histories with independently recomputed checkpoint losses."""

    logged = pd.read_csv(logged_csv)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    panels = (
        (
            "MNTP diffusion validation",
            "diffusion",
            (
                ("transferred_mntp", "transferred_diffusion_loss", "#E45756"),
                ("scratch_mntp", "scratch_diffusion_loss", "#72B7B2"),
            ),
        ),
        (
            "MNTP single-mask validation",
            "single",
            (
                ("transferred_mntp", "transferred_single_mask_loss", "#E45756"),
                ("scratch_mntp", "scratch_single_mask_loss", "#72B7B2"),
            ),
        ),
        (
            "Causal validation",
            "causal",
            (("clm_continuation", "clm_diffusion_loss", "#4C78A8"),),
        ),
    )
    for axis, (title, mode, series) in zip(axes, panels, strict=True):
        for arm, column, color in series:
            axis.plot(
                logged["step"],
                logged[column],
                color=color,
                linewidth=1.5,
                label=f"{arm} W&B",
            )
            audit = losses[
                (losses["arm"] == arm) & (losses["validation_mode"] == mode)
            ].sort_values("step")
            axis.scatter(
                audit["step"],
                audit["loss"],
                facecolors="none",
                edgecolors=color,
                marker="o",
                s=35,
                label=f"{arm} recomputed",
            )
        axis.set_xscale("symlog", linthresh=10)
        axis.set_xlabel("Optimizer steps (symlog; linear through 10)")
        axis.set_ylabel("Cross-entropy loss")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    figure.suptitle("Training histories with independent checkpoint recomputation")
    figure.savefig(output_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def trajectory_points(replay_dir: Path) -> list[ModelPoint]:
    """Enumerate every available original checkpoint plus early CLM replay."""

    points = [
        ModelPoint(
            "source-clm-direct-step0000",
            "source_clm",
            0,
            "clm",
            "source",
            "Source CLM direct",
        ),
        ModelPoint(
            "clm-replay-step0000",
            "clm_continuation",
            0,
            "clm",
            "replay",
            "Source CLM save/reload",
            replay_dir / "step-0000",
        ),
    ]
    points.extend(
        ModelPoint(
            f"clm-replay-step{step:04d}",
            "clm_continuation",
            step,
            "clm",
            "replay",
            "Continued CLM replay",
            replay_dir / f"step-{step:04d}",
        )
        for step in REPLAY_STEPS[1:]
    )
    points.extend(
        ModelPoint(
            f"clm-original-step{step:04d}",
            "clm_continuation",
            step,
            "clm",
            "hf" if step == 1000 else "lightning",
            "Continued CLM original",
        )
        for step in (400, 800, 1000)
    )
    points.append(
        ModelPoint(
            "transferred-step0000",
            "full_attention_no_adaptation",
            0,
            "mntp",
            "no_adaptation",
            "Transferred MNTP",
        )
    )
    points.extend(
        ModelPoint(
            f"transferred-step{step:04d}",
            "transferred_mntp",
            step,
            "mntp",
            "hf" if step == 1000 else "lightning",
            "Transferred MNTP",
        )
        for step in (*range(100, 900, 100), 1000)
    )
    points.append(
        ModelPoint(
            "scratch-step0000",
            "scratch_mntp",
            0,
            "mntp",
            "scratch_initial",
            "Scratch MNTP",
        )
    )
    points.extend(
        ModelPoint(
            f"scratch-step{step:04d}",
            "scratch_mntp",
            step,
            "mntp",
            "hf" if step == 1000 else "lightning",
            "Scratch MNTP",
        )
        for step in (400, 800, 1000)
    )
    return points


def publish_wandb(
    output_dir: Path,
    metrics: pd.DataFrame,
    losses: pd.DataFrame,
    checks: pd.DataFrame,
) -> str | None:
    """Publish compact numeric tables and figures to a dedicated W&B run."""

    if not os.getenv("WANDB_API_KEY"):
        return None
    run = wandb.init(
        project="marin",
        group="dna-exp479",
        name="dna-exp479-checkpoint-trajectory-audit",
        tags=["MNTP-479", "issue-479", "checkpoint-audit", "alignment-audit"],
        config={
            "development_split": "odd autosomes and X",
            "replay_steps": list(REPLAY_STEPS),
            "score_parity_tolerance": SCORE_PARITY_TOLERANCE,
            "loss_parity_tolerance": LOSS_PARITY_TOLERANCE,
        },
    )
    run.log(
        {
            "checkpoint_auprc": wandb.Table(dataframe=metrics),
            "checkpoint_loss": wandb.Table(dataframe=losses),
            "parity_checks": wandb.Table(dataframe=checks),
            "auprc_trajectories": wandb.Image(
                str(output_dir / "figures" / "auprc-trajectories.png")
            ),
            "loss_trajectories": wandb.Image(str(output_dir / "figures" / "loss-trajectories.png")),
            "nucleotide_dependency": wandb.Image(
                str(output_dir / "figures" / "nucleotide-dependency-panel.png")
            ),
        }
    )
    run.summary["all_contract_checks_passed"] = bool(checks["passed"].all())
    run.summary["maximum_score_parity_abs_error"] = float(checks["max_abs_error"].max())
    url = run.get_url()
    run.finish(exit_code=0)
    return url


def run_checkpoint_audit(
    *,
    artifact_dir: Path,
    output_dir: Path,
    train_plan: Path,
    validation_plan: Path,
    logged_loss_csv: Path,
    hf_repo_id: str,
    batch_size: int,
    vep_batch_size: int,
    dependency_batch_size: int,
    n_bootstrap: int,
    num_workers: int,
) -> None:
    """Replay, score, gate, plot, and privately publish the complete audit."""

    if not torch.cuda.is_available():
        raise RuntimeError("checkpoint audit requires the Lambda GH200")
    if hf_repo_id not in CHECKPOINT_REPOS:
        raise ValueError(f"unexpected publication repository {hf_repo_id}")
    assert_plan_contract(train_plan, validation_plan)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = artifact_dir / "checkpoint-audit-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = _repo_files()
    reference = download_reference(artifact_dir / "reference")
    frames = {
        spec.name: attach_reference_windows(load_variant_frame(spec), reference)
        for spec in DATASETS
    }
    coordinates = coordinate_audit(frames, reference)
    coordinates.to_csv(output_dir / "coordinate-audit.csv", index=False)

    replay_dir = artifact_dir / "clm-replay"
    replay_runtime = replay_early_clm(
        train_plan=train_plan,
        validation_plan=validation_plan,
        output_dir=replay_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    metric_rows: list[dict[str, object]] = []
    loss_rows: list[dict[str, object]] = []
    alignment_rows: list[dict[str, object]] = []
    points = trajectory_points(replay_dir)
    for point in points:
        assert_budget_reserve()
        arm = load_point(point, cache_dir=cache_dir, files=files)
        arm.model.to(device="cuda", dtype=torch.bfloat16).eval()
        alignment_rows.append(audit_alignment(point, arm))
        loss_rows.extend(
            evaluate_validation_loss(
                point,
                arm,
                validation_plan,
                batch_size=batch_size,
            )
        )
        metric_rows.extend(
            evaluate_point(
                point,
                arm,
                frames,
                output_dir,
                batch_size=vep_batch_size,
                n_bootstrap=n_bootstrap,
            )
        )
        del arm
        gc.collect()
        torch.cuda.empty_cache()

    metrics = pd.DataFrame(metric_rows)
    losses = attach_logged_loss_parity(pd.DataFrame(loss_rows), logged_loss_csv)
    alignment = pd.DataFrame(alignment_rows)
    metrics.to_csv(output_dir / "checkpoint-auprc.csv", index=False)
    losses.to_csv(output_dir / "checkpoint-loss-audit.csv", index=False)
    alignment.to_json(output_dir / "alignment-audit.json", orient="records", indent=2)

    parity_rows = compare_score_points(
        output_dir,
        "source-clm-direct-step0000",
        "clm-replay-step0000",
        "source_direct_vs_zero_update_save_reload",
    )
    parity_rows.extend(
        compare_score_points(
            output_dir,
            "clm-replay-step0400",
            "clm-original-step0400",
            "replayed_step400_vs_original_step400",
        )
    )
    for point_id, prefix in (
        ("source-clm-direct-step0000", "source_clm"),
        ("transferred-step0000", "full_attention_no_adaptation"),
        ("transferred-step1000", "transferred_mntp"),
        ("scratch-step1000", "scratch_mntp"),
        ("clm-original-step1000", "clm_continuation"),
    ):
        parity_rows.extend(
            compare_existing_scores(
                output_dir=output_dir,
                point_id=point_id,
                existing_prefix=prefix,
                cache_dir=cache_dir,
                files=files,
            )
        )
    dependency_summary, dependency_checks = run_dependency_audit(
        artifact_dir=artifact_dir,
        output_dir=output_dir,
        cache_dir=cache_dir,
        files=files,
        batch_size=dependency_batch_size,
    )
    parity_rows.extend(dependency_checks)
    dependency_summary.to_csv(
        output_dir / "nucleotide-dependency-triangles.csv",
        index=False,
    )

    loss_checks = losses[losses["logged_loss"].notna()][
        [
            "point_id",
            "arm",
            "step",
            "validation_mode",
            "logged_abs_error",
            "logged_parity_passed",
        ]
    ]
    for row in loss_checks.itertuples(index=False):
        parity_rows.append(
            {
                "comparison": f"{row.point_id}_{row.validation_mode}_vs_wandb",
                "dataset": "validation_plan",
                "score": "loss",
                "n_rows": 640,
                "max_abs_error": float(row.logged_abs_error),
                "mean_abs_error": float(row.logged_abs_error),
                "tolerance": LOSS_PARITY_TOLERANCE,
                "passed": bool(row.logged_parity_passed),
            }
        )
    checks = pd.DataFrame(parity_rows)
    checks.to_csv(output_dir / "parity-checks.csv", index=False)
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_auprc_trajectories(metrics, figures / "auprc-trajectories")
    plot_loss_trajectories(losses, logged_loss_csv, figures / "loss-trajectories")
    wandb_url = publish_wandb(output_dir, metrics, losses, checks)
    manifest = {
        "train_plan_sha256": plan_sha256(train_plan),
        "validation_plan_sha256": plan_sha256(validation_plan),
        "development_split": "odd autosomes and X only",
        "points": [
            point.__dict__ | {"path": None if point.path is None else str(point.path)}
            for point in points
        ],
        "replay": replay_runtime,
        "score_parity_tolerance": SCORE_PARITY_TOLERANCE,
        "loss_parity_tolerance": LOSS_PARITY_TOLERANCE,
        "all_contract_checks_passed": bool(checks["passed"].all()),
        "wandb_url": wandb_url,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    assert_budget_reserve()
    HfApi().upload_folder(
        folder_path=output_dir,
        path_in_repo="evaluation/checkpoint-audit",
        repo_id=hf_repo_id,
        repo_type="model",
        commit_message="Upload exp479 checkpoint trajectories and contract audit",
    )
    if not bool(checks["passed"].all()):
        failed = checks.loc[~checks["passed"], "comparison"].tolist()
        raise RuntimeError(f"checkpoint audit failed contract checks: {failed}")
