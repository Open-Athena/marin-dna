"""Sequential, gated, budget-bounded GPU execution for issue #515."""

from __future__ import annotations

import argparse
import base64
import gc
import json
import math
import os
import shutil
import signal
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from glm_experiments.data.lm_datamodule import has_eligible_target
from glm_experiments.exp515.config import (
    ACCELERATOR,
    ALL_IN_CAP_USD,
    ARMS,
    BRIDGE_STEPS,
    CANARY_STEPS,
    EFFECTIVE_BATCH_SIZE,
    GPU_COMPUTE_CAP_USD,
    GPU_PRICE_PER_HOUR_USD,
    GRADIENT_CLIP_VALUE,
    ISSUE_S3_PREFIX,
    MAX_CONTINUATION_STEPS,
    RUNTIME_MARGIN,
    SEED,
    SOURCE_CHECKPOINT,
    continuation_endpoint,
    continuation_midpoint,
)
from glm_experiments.exp515.data import (
    SequenceCollator,
    SequencePlanDataset,
    build_sequence_plan,
    validate_sequence_plan,
)
from glm_experiments.exp515.diagnostics import Exp515Diagnostics
from glm_experiments.exp515.evaluation import (
    evaluate_promoter_auprc,
    load_promoter_frame,
)
from glm_experiments.exp515.module import Exp515Module, checkpoint_next_sample_id
from glm_experiments.exp515.storage import upload_issue_artifact
from glm_experiments.models.components.lm import HFCLM
from glm_experiments.models.components.selection import (
    TokenSelector,
    select_token_mask,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _utc_timestamp(epoch_seconds: float | None = None) -> str:
    """Format one UTC timestamp for durable run metadata."""

    value = time.time() if epoch_seconds is None else epoch_seconds
    return datetime.fromtimestamp(value, tz=UTC).isoformat()


def _install_compute_guard(instance_start: float) -> None:
    """Raise before one GPU can exceed the registered compute cap."""

    maximum_seconds = GPU_COMPUTE_CAP_USD / GPU_PRICE_PER_HOUR_USD * 3600
    remaining = maximum_seconds - (time.time() - instance_start)
    if remaining <= 0:
        raise TimeoutError("issue #515 compute wall-clock allowance is exhausted")

    def stop_compute(_signum: int, _frame: Any) -> None:
        raise TimeoutError("issue #515 hard GPU wall-clock guard fired")

    signal.signal(signal.SIGALRM, stop_compute)
    signal.alarm(max(1, math.ceil(remaining)))


def _is_cuda_oom(error: RuntimeError) -> bool:
    """Recognize both direct and Lightning-wrapped CUDA OOM failures."""

    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        "out of memory" in str(error).lower() and "cuda" in str(error).lower()
    )


def _gcs_token() -> dict[str, Any] | str | None:
    encoded = os.getenv("GOOGLE_ADC_JSON_BASE64")
    if encoded:
        return json.loads(base64.b64decode(encoded).decode("utf-8"))
    credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    return credentials or None


def download_source_checkpoint(destination: Path) -> Path:
    """Download the exact step-2,000 HF export from GCS."""

    required = {
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    }
    if destination.exists() and required <= {
        path.name for path in destination.iterdir()
    }:
        return destination
    import gcsfs

    destination.mkdir(parents=True, exist_ok=True)
    filesystem = gcsfs.GCSFileSystem(token=_gcs_token())
    prefix = SOURCE_CHECKPOINT.removeprefix("gs://").rstrip("/")
    remote_files = [path for path in filesystem.find(prefix) if not path.endswith("/")]
    for remote in remote_files:
        target = destination / Path(remote).relative_to(prefix)
        target.parent.mkdir(parents=True, exist_ok=True)
        filesystem.get(remote, str(target))
    missing = required - {path.name for path in destination.iterdir()}
    if missing:
        raise FileNotFoundError(f"source checkpoint is incomplete: {sorted(missing)}")
    return destination


def _load_tokenizer(source_dir: Path) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(source_dir)
    if tokenizer.bos_token_id is None:
        raise ValueError("source checkpoint tokenizer lacks BOS")
    probe = tokenizer("A" * 255, add_special_tokens=True)
    if (
        len(probe["input_ids"]) != 256
        or probe["input_ids"][0] != tokenizer.bos_token_id
    ):
        raise ValueError(
            "source tokenizer is not one BOS plus one token per nucleotide"
        )
    return tokenizer


def _new_module(
    source_dir: Path,
    *,
    continuation_steps: int,
    plan_sha256: str,
    selector_mode: str,
    selector_ratio: float,
) -> Exp515Module:
    net = HFCLM(
        str(source_dir),
        torch_dtype="bfloat16",
        selector_enabled=True,
        selector_mode=selector_mode,  # type: ignore[arg-type]
        selector_ratio=selector_ratio,
        selector_seed=SEED + 1000,
    )
    return Exp515Module(
        net,
        continuation_steps=continuation_steps,
        plan_sha256=plan_sha256,
        selector_mode=selector_mode,  # type: ignore[arg-type]
        selector_ratio=selector_ratio,
    )


def _checkpoint_step(path: Path) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return int(checkpoint["global_step"])


def _optional_loggers(output_dir: Path, run_name: str) -> list[Any]:
    """Always return CSV; add W&B only when explicit initialization succeeds."""

    loggers: list[Any] = [CSVLogger(save_dir=output_dir, name="csv", version=run_name)]
    status = {"csv": "enabled", "wandb": "disabled"}
    if os.getenv("EXP515_USE_WANDB", "0") == "1":
        try:
            import wandb
            from lightning.pytorch.loggers import WandbLogger

            run = wandb.init(
                project="marin",
                name=f"exp515-{run_name}",
                config={"issue": 515, "run_name": run_name},
                settings=wandb.Settings(init_timeout=30),
            )
            loggers.append(WandbLogger(experiment=run))
            status["wandb"] = "enabled"
        # Logger availability must never supersede the authoritative CSV stream.
        except Exception as error:  # noqa: BLE001
            status["wandb"] = f"fallback_to_csv: {type(error).__name__}: {error}"
    _write_json(output_dir / "logging-status.json", status)
    return loggers


def run_training_phase(
    *,
    source_dir: Path,
    tokenizer: PreTrainedTokenizerBase,
    plan_dir: Path,
    output_dir: Path,
    run_name: str,
    selector_mode: str,
    selector_ratio: float,
    continuation_steps: int,
    target_global_step: int,
    microbatch_size: int,
    resume_from: Path | None,
) -> tuple[Exp515Module, Path, dict[str, Any]]:
    """Run one exact plan segment and save a full Lightning checkpoint."""

    if EFFECTIVE_BATCH_SIZE % microbatch_size:
        raise ValueError("microbatch must divide the effective batch")
    plan = validate_sequence_plan(plan_dir)
    start_sample = (
        0 if resume_from is None else checkpoint_next_sample_id(str(resume_from))
    )
    start_step = 0 if resume_from is None else _checkpoint_step(resume_from)
    expected_start = start_step * EFFECTIVE_BATCH_SIZE
    if start_sample != expected_start:
        raise ValueError(
            f"checkpoint data position {start_sample} != global-step position {expected_start}"
        )
    if target_global_step <= start_step:
        raise ValueError("phase target must exceed its checkpoint step")
    phase_rows = (target_global_step - start_step) * EFFECTIVE_BATCH_SIZE
    dataset = SequencePlanDataset(plan_dir, start=start_sample, rows=phase_rows)
    loader = DataLoader(
        dataset,
        batch_size=microbatch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        collate_fn=SequenceCollator(tokenizer),
    )
    module = _new_module(
        source_dir,
        continuation_steps=continuation_steps,
        plan_sha256=str(plan["sequences_sha256"]),
        selector_mode=selector_mode,
        selector_ratio=selector_ratio,
    )
    phase_dir = output_dir / run_name
    phase_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = Exp515Diagnostics(phase_dir, every_n_steps=10)
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        max_steps=target_global_step,
        accumulate_grad_batches=EFFECTIVE_BATCH_SIZE // microbatch_size,
        gradient_clip_val=GRADIENT_CLIP_VALUE,
        gradient_clip_algorithm="norm",
        logger=_optional_loggers(phase_dir, run_name),
        callbacks=[diagnostics],
        enable_checkpointing=False,
        enable_progress_bar=True,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        deterministic=True,
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    trainer.fit(
        module,
        train_dataloaders=loader,
        ckpt_path=str(resume_from) if resume_from else None,
    )
    elapsed = time.time() - started
    if int(trainer.global_step) != target_global_step:
        raise RuntimeError(
            f"phase ended at step {trainer.global_step}, expected {target_global_step}"
        )
    checkpoint = phase_dir / f"step-{target_global_step}.ckpt"
    trainer.save_checkpoint(checkpoint, weights_only=False)
    metadata = {
        "run_name": run_name,
        "selector_mode": selector_mode,
        "selector_ratio": selector_ratio,
        "start_global_step": start_step,
        "end_global_step": target_global_step,
        "start_sample_id": start_sample,
        "end_sample_id": target_global_step * EFFECTIVE_BATCH_SIZE,
        "microbatch_size": microbatch_size,
        "gradient_accumulation": EFFECTIVE_BATCH_SIZE // microbatch_size,
        "elapsed_seconds": elapsed,
        "seconds_per_optimizer_step": elapsed / (target_global_step - start_step),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "peak_memory_fraction": torch.cuda.max_memory_allocated()
        / torch.cuda.get_device_properties(0).total_memory,
        "checkpoint": str(checkpoint),
    }
    _write_json(phase_dir / "runtime.json", metadata)
    return module, checkpoint, metadata


def _selector_device_smoke(device: torch.device) -> dict[str, Any]:
    """Exercise every selector contract on the paid run's actual device."""

    losses = torch.tensor(
        [
            [1.0, 1.0, 3.0, 2.0, 4.0, 0.0],
            [9.0, 4.0, 1.0, 3.0, 2.0, 8.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ],
        device=device,
    )
    eligible = torch.tensor(
        [
            [True, True, True, True, True, False],
            [False, True, True, True, True, False],
            [False, False, False, False, False, False],
        ],
        device=device,
    )
    expected = {
        "student_low": torch.tensor(
            [
                [True, True, False, False, False, False],
                [False, False, True, False, True, False],
                [False, False, False, False, False, False],
            ],
            device=device,
        ),
        "student_middle": torch.tensor(
            [
                [False, True, False, True, False, False],
                [False, False, False, True, True, False],
                [False, False, False, False, False, False],
            ],
            device=device,
        ),
        "student_high": torch.tensor(
            [
                [False, False, True, False, True, False],
                [False, True, False, True, False, False],
                [False, False, False, False, False, False],
            ],
            device=device,
        ),
    }
    ranked_passed = all(
        torch.equal(
            select_token_mask(losses, eligible, mode=mode, ratio=0.5),
            expected_mask,
        )
        for mode, expected_mask in expected.items()
    )
    random_selector = TokenSelector(mode="random", ratio=0.5, seed=515)
    random_selector(losses, eligible)
    state = random_selector.state_dict()
    expected_random = random_selector(losses, eligible)
    resumed = TokenSelector(mode="random", ratio=0.5, seed=515)
    resumed.load_state_dict(state)
    random_resume_passed = torch.equal(resumed(losses, eligible), expected_random)
    payload = {
        "device": str(device),
        "ranked_masks_passed": ranked_passed,
        "random_resume_passed": random_resume_passed,
        "empty_row_passed": not bool(expected_random[2].any()),
    }
    payload["passed"] = all(
        bool(value) for key, value in payload.items() if key != "device"
    )
    if not payload["passed"]:
        raise RuntimeError(f"selector device smoke test failed: {payload}")
    return payload


def smoke_test(
    source_dir: Path,
    tokenizer: PreTrainedTokenizerBase,
    plan_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Check direct-HF logits and selector-disabled uniform loss parity."""

    dataset = SequencePlanDataset(plan_dir, start=0, rows=2)
    batch = SequenceCollator(tokenizer)([dataset[0], dataset[1]])
    selector_checks = _selector_device_smoke(torch.device("cuda"))
    synthetic_sequence = "A" + "c" + "A" * 253
    synthetic_batch = SequenceCollator(tokenizer)(
        [{"sample_id": 0, "sequence": synthetic_sequence, "species": "synthetic"}]
    )
    bos_repeat_alignment_passed = bool(
        not synthetic_batch["soft_masked"][0, 0]
        and not synthetic_batch["soft_masked"][0, 1]
        and synthetic_batch["soft_masked"][0, 2]
    )
    all_lowercase_filter_passed = not has_eligible_target(
        "a" * 255
    ) and has_eligible_target(synthetic_sequence)
    device_batch = {
        key: value.cuda() if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    direct = AutoModelForCausalLM.from_pretrained(
        source_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).cuda()
    direct.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = (
            direct(
                input_ids=device_batch["input_ids"],
                attention_mask=device_batch["attention_mask"],
                use_cache=False,
            )
            .logits.float()
            .cpu()
        )
    del direct
    gc.collect()
    torch.cuda.empty_cache()
    module = _new_module(
        source_dir,
        continuation_steps=MAX_CONTINUATION_STEPS,
        plan_sha256=dataset.sha256,
        selector_mode="uniform",
        selector_ratio=1.0,
    ).cuda()
    module.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        observed = (
            module.net.get_logits(
                device_batch["input_ids"],
                attention_mask=device_batch["attention_mask"],
            )
            .float()
            .cpu()
        )
        selected = module.forward(device_batch)
        module.net.selector_enabled = False
        disabled = module.net(
            input_ids=device_batch["input_ids"],
            labels=device_batch["labels"],
            soft_masked=device_batch["soft_masked"],
            soft_masked_weight=0.0,
            attention_mask=device_batch["attention_mask"],
        )
    max_logit_error = float((expected - observed).abs().max())
    loss_error = float((selected["loss"] - disabled["loss"]).abs())
    payload = {
        "max_absolute_logit_error": max_logit_error,
        "uniform_disabled_loss_absolute_error": loss_error,
        "eligible_count": int(selected["eligible_count"]),
        "selected_count": int(selected["selected_count"]),
        "selector_device": selector_checks,
        "bos_repeat_alignment_passed": bos_repeat_alignment_passed,
        "all_lowercase_filter_passed": all_lowercase_filter_passed,
        "passed": max_logit_error == 0.0
        and loss_error <= 1e-6
        and selector_checks["passed"]
        and bos_repeat_alignment_passed
        and all_lowercase_filter_passed,
    }
    _write_json(output_path, payload)
    if not payload["passed"]:
        raise RuntimeError(f"source parity smoke test failed: {payload}")
    del module
    gc.collect()
    torch.cuda.empty_cache()
    return payload


def choose_continuation_steps(
    *,
    canary_seconds_per_step: float,
    evaluation_seconds: float,
    instance_start: float,
) -> dict[str, Any]:
    """Choose one common arm length under the remaining compute envelope."""

    elapsed = time.time() - instance_start
    maximum_seconds = GPU_COMPUTE_CAP_USD / GPU_PRICE_PER_HOUR_USD * 3600
    remaining = maximum_seconds - elapsed
    remaining_evaluations = 10
    usable_training = (
        remaining / RUNTIME_MARGIN - remaining_evaluations * evaluation_seconds
    )
    steps = math.floor(usable_training / (len(ARMS) * canary_seconds_per_step))
    steps = min(MAX_CONTINUATION_STEPS, steps)
    if steps < 2:
        raise RuntimeError(
            "measured runtime leaves no budget for the registered matrix"
        )
    projected_seconds = RUNTIME_MARGIN * (
        len(ARMS) * steps * canary_seconds_per_step
        + remaining_evaluations * evaluation_seconds
    )
    payload = {
        "instance_elapsed_seconds": elapsed,
        "canary_seconds_per_optimizer_step": canary_seconds_per_step,
        "evaluation_seconds": evaluation_seconds,
        "runtime_margin": RUNTIME_MARGIN,
        "continuation_steps": steps,
        "remaining_evaluations": remaining_evaluations,
        "projected_remaining_seconds": projected_seconds,
        "projected_total_compute_usd": (elapsed + projected_seconds)
        / 3600
        * GPU_PRICE_PER_HOUR_USD,
        "gpu_compute_cap_usd": GPU_COMPUTE_CAP_USD,
        "all_in_cap_usd": ALL_IN_CAP_USD,
    }
    if payload["projected_total_compute_usd"] >= GPU_COMPUTE_CAP_USD:
        raise RuntimeError("initial runtime projection reaches the compute stop")
    return payload


def _retain_for_publication(relative: Path) -> bool:
    """Exclude reproducible caches and non-evidentiary canary checkpoints."""

    if relative.parts[0] in {"source-checkpoint", "evaluation-cache"}:
        return False
    if relative.parts[0] == "sequence-plan" and relative.name != "manifest.json":
        return False
    if relative.name == "s3-upload-manifest.json" or relative.name.endswith(".partial"):
        return False
    return not (
        relative.suffix == ".ckpt"
        and relative.parts[0] in {"canary-selection", "canary-20"}
    )


def publish_run_artifacts(root: Path, run_id: str) -> list[dict[str, object]]:
    """Publish curated run evidence and retained checkpoints, including failures."""

    if not root.exists():
        raise FileNotFoundError(root)
    import boto3

    client = boto3.client("s3")
    records: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if not _retain_for_publication(relative):
            continue
        parent = relative.parent.as_posix()
        remote_parent = f"runs/{run_id}"
        if parent != ".":
            remote_parent = f"{remote_parent}/{parent}"
        records.extend(
            upload_issue_artifact(
                path,
                destination_prefix=ISSUE_S3_PREFIX,
                relative_path=remote_parent,
                client=client,
            )
        )
    manifest_path = root / "s3-upload-manifest.json"
    _write_json(manifest_path, {"objects": records})
    records.extend(
        upload_issue_artifact(
            manifest_path,
            destination_prefix=ISSUE_S3_PREFIX,
            relative_path=f"runs/{run_id}",
            client=client,
        )
    )
    return records


def run_experiment(artifact_dir: Path, *, experiment_commit: str, run_id: str) -> None:
    """Execute the canary, bridge, gate, and conditional five-arm matrix."""

    if len(experiment_commit) != 40:
        raise ValueError("experiment commit must be a full SHA")
    seed_everything(SEED, workers=True)
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run ID must be one safe path component")
    instance_start = float(os.getenv("EXP515_INSTANCE_START_UNIX", time.time()))
    root = artifact_dir / run_id
    _install_compute_guard(instance_start)
    root.mkdir(parents=True, exist_ok=True)
    source_dir = download_source_checkpoint(root / "source-checkpoint")
    tokenizer = _load_tokenizer(source_dir)
    plan_dir = root / "sequence-plan"
    max_rows = (BRIDGE_STEPS + MAX_CONTINUATION_STEPS) * EFFECTIVE_BATCH_SIZE
    plan = build_sequence_plan(plan_dir, rows=max_rows, seed=SEED)
    audit_path = Path(os.environ["EXP515_CASE_AUDIT"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("revision") != plan["revision"] or audit.get("fallback_required"):
        raise RuntimeError(
            "prepaid case audit is missing, mismatched, or requires fallback"
        )
    shutil.copy2(audit_path, root / "case-distribution-audit.json")
    smoke_test(source_dir, tokenizer, plan_dir, root / "smoke-test.json")

    selected_microbatch = 0
    one_step_metadata: dict[str, Any] | None = None
    for candidate in (256, 128, 64, 32, 16, 8, 4, 2, 1):
        try:
            candidate_module, _, metadata = run_training_phase(
                source_dir=source_dir,
                tokenizer=tokenizer,
                plan_dir=plan_dir,
                output_dir=root / "canary-selection",
                run_name=f"microbatch-{candidate}",
                selector_mode="uniform",
                selector_ratio=1.0,
                continuation_steps=MAX_CONTINUATION_STEPS,
                target_global_step=1,
                microbatch_size=candidate,
                resume_from=None,
            )
        except RuntimeError as error:
            if not _is_cuda_oom(error):
                raise
            gc.collect()
            torch.cuda.empty_cache()
            continue
        del candidate_module
        gc.collect()
        torch.cuda.empty_cache()
        if float(metadata["peak_memory_fraction"]) < 0.85:
            selected_microbatch = candidate
            one_step_metadata = metadata
            break
    if not selected_microbatch or one_step_metadata is None:
        raise RuntimeError("no power-of-two microbatch passed the 85% HBM gate")

    _, canary_checkpoint, canary = run_training_phase(
        source_dir=source_dir,
        tokenizer=tokenizer,
        plan_dir=plan_dir,
        output_dir=root,
        run_name="canary-20",
        selector_mode="uniform",
        selector_ratio=1.0,
        continuation_steps=MAX_CONTINUATION_STEPS,
        target_global_step=CANARY_STEPS,
        microbatch_size=selected_microbatch,
        resume_from=None,
    )
    del canary_checkpoint
    bridge_module, bridge_checkpoint, bridge = run_training_phase(
        source_dir=source_dir,
        tokenizer=tokenizer,
        plan_dir=plan_dir,
        output_dir=root,
        run_name="bridge",
        selector_mode="uniform",
        selector_ratio=1.0,
        continuation_steps=MAX_CONTINUATION_STEPS,
        target_global_step=BRIDGE_STEPS,
        microbatch_size=selected_microbatch,
        resume_from=None,
    )
    evaluation_frame = load_promoter_frame(root / "evaluation-cache")
    bridge_eval = evaluate_promoter_auprc(
        bridge_module.net,
        tokenizer,
        frame=evaluation_frame,
        output_path=root / "evaluations" / "bridge.csv",
        batch_size=selected_microbatch,
        checkpoint_name="bridge",
    )
    projection = choose_continuation_steps(
        canary_seconds_per_step=float(canary["seconds_per_optimizer_step"]),
        evaluation_seconds=float(bridge_eval["elapsed_seconds"]),
        instance_start=instance_start,
    )
    _write_json(root / "budget-projection.json", projection)
    continuation_steps = int(projection["continuation_steps"])
    del bridge_module
    gc.collect()
    torch.cuda.empty_cache()

    summaries = [bridge_eval]
    runtimes = [bridge, canary]
    uniform_endpoint_auprc = float("nan")
    for arm_index, arm in enumerate(ARMS):
        if arm_index > 0 and not uniform_endpoint_auprc > float(bridge_eval["auprc"]):
            break
        midpoint = continuation_midpoint(continuation_steps)
        endpoint = continuation_endpoint(continuation_steps)
        midpoint_module, midpoint_checkpoint, midpoint_runtime = run_training_phase(
            source_dir=source_dir,
            tokenizer=tokenizer,
            plan_dir=plan_dir,
            output_dir=root,
            run_name=f"{arm.name}-midpoint",
            selector_mode=arm.selector_mode,
            selector_ratio=arm.selector_ratio,
            continuation_steps=continuation_steps,
            target_global_step=midpoint,
            microbatch_size=selected_microbatch,
            resume_from=bridge_checkpoint,
        )
        midpoint_eval = evaluate_promoter_auprc(
            midpoint_module.net,
            tokenizer,
            frame=evaluation_frame,
            output_path=root / "evaluations" / f"{arm.name}-midpoint.csv",
            batch_size=selected_microbatch,
            checkpoint_name=f"{arm.name}-midpoint",
        )
        del midpoint_module
        gc.collect()
        torch.cuda.empty_cache()
        endpoint_module, endpoint_checkpoint, endpoint_runtime = run_training_phase(
            source_dir=source_dir,
            tokenizer=tokenizer,
            plan_dir=plan_dir,
            output_dir=root,
            run_name=f"{arm.name}-endpoint",
            selector_mode=arm.selector_mode,
            selector_ratio=arm.selector_ratio,
            continuation_steps=continuation_steps,
            target_global_step=endpoint,
            microbatch_size=selected_microbatch,
            resume_from=midpoint_checkpoint,
        )
        endpoint_eval = evaluate_promoter_auprc(
            endpoint_module.net,
            tokenizer,
            frame=evaluation_frame,
            output_path=root / "evaluations" / f"{arm.name}-endpoint.csv",
            batch_size=selected_microbatch,
            checkpoint_name=f"{arm.name}-endpoint",
        )
        del endpoint_module
        gc.collect()
        torch.cuda.empty_cache()
        summaries.extend((midpoint_eval, endpoint_eval))
        runtimes.extend((midpoint_runtime, endpoint_runtime))
        _write_json(
            root / f"{arm.name}-result.json",
            {
                "arm": arm.name,
                "bridge_auprc": bridge_eval["auprc"],
                "midpoint_auprc": midpoint_eval["auprc"],
                "endpoint_auprc": endpoint_eval["auprc"],
                "endpoint_checkpoint": str(endpoint_checkpoint),
            },
        )
        if arm.name == "uniform-100":
            uniform_endpoint_auprc = float(endpoint_eval["auprc"])
            gate = {
                "bridge_auprc": bridge_eval["auprc"],
                "uniform_endpoint_auprc": uniform_endpoint_auprc,
                "passed": uniform_endpoint_auprc > float(bridge_eval["auprc"]),
            }
            _write_json(root / "uniform-gate.json", gate)
            observed_uniform_seconds = float(
                midpoint_runtime["elapsed_seconds"]
            ) + float(endpoint_runtime["elapsed_seconds"])
            estimated_remaining = RUNTIME_MARGIN * (
                4 * observed_uniform_seconds + 8 * float(bridge_eval["elapsed_seconds"])
            )
            projected_cost = (
                (time.time() - instance_start + estimated_remaining)
                / 3600
                * GPU_PRICE_PER_HOUR_USD
            )
            _write_json(
                root / "post-uniform-budget-projection.json",
                {
                    "projected_total_compute_usd": projected_cost,
                    "compute_cap_usd": GPU_COMPUTE_CAP_USD,
                    "passed": projected_cost < GPU_COMPUTE_CAP_USD,
                },
            )
            if not gate["passed"] or projected_cost >= GPU_COMPUTE_CAP_USD:
                break

    elapsed = time.time() - instance_start
    final = {
        "experiment_commit": experiment_commit,
        "run_id": run_id,
        "provider": "Lambda Cloud",
        "accelerator": ACCELERATOR,
        "hourly_rate_usd": GPU_PRICE_PER_HOUR_USD,
        "instance_start_unix": instance_start,
        "instance_start_utc": _utc_timestamp(instance_start),
        "continuation_steps": continuation_steps,
        "microbatch_size": selected_microbatch,
        "bridge_auprc": bridge_eval["auprc"],
        "uniform_endpoint_auprc": uniform_endpoint_auprc,
        "completed_evaluations": summaries,
        "runtimes": runtimes,
        "elapsed_seconds": elapsed,
        "completed_at_utc": _utc_timestamp(),
        "estimated_compute_cost_usd": elapsed / 3600 * GPU_PRICE_PER_HOUR_USD,
        "compute_cap_usd": GPU_COMPUTE_CAP_USD,
        "all_in_cap_usd": ALL_IN_CAP_USD,
    }
    _write_json(root / "final-manifest.json", final)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--experiment-commit", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in {".", ".."}:
        raise ValueError("run ID must be one safe path component")
    root = args.artifact_dir / args.run_id
    try:
        run_experiment(
            args.artifact_dir,
            experiment_commit=args.experiment_commit,
            run_id=args.run_id,
        )
    except BaseException as error:
        signal.alarm(0)
        instance_start = float(os.getenv("EXP515_INSTANCE_START_UNIX", time.time()))
        failure = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "failed_at_utc": _utc_timestamp(),
            "estimated_compute_cost_usd": (time.time() - instance_start)
            / 3600
            * GPU_PRICE_PER_HOUR_USD,
        }
        if root.exists():
            _write_json(root / "failure.json", failure)
            try:
                publish_run_artifacts(root, args.run_id)
            except Exception as publication_error:  # noqa: BLE001
                failure["publication_error"] = (
                    f"{type(publication_error).__name__}: {publication_error}"
                )
                _write_json(root / "failure.json", failure)
        raise
    else:
        try:
            publish_run_artifacts(root, args.run_id)
        finally:
            signal.alarm(0)


if __name__ == "__main__":
    main()
