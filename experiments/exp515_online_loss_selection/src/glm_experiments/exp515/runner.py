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
    CDS_ARM_WARMUP_STEPS,
    CDS_ARMS,
    CDS_GATE_CONTINUATION_STEPS,
    CDS_NUCLEOTIDE_LENGTH,
    CDS_SEQUENCE_LENGTH,
    EFFECTIVE_BATCH_SIZE,
    EXP58_STUDENT_CHECKPOINT,
    EXP58_TEACHER_CHECKPOINT,
    EXP58_TRAIN_DATASET,
    EXP58_TRAIN_REVISION,
    EXP58_TRAIN_TEXT_KEY,
    GPU_COMPUTE_CAP_USD,
    GPU_PRICE_PER_HOUR_USD,
    GRADIENT_CLIP_VALUE,
    ISSUE_S3_PREFIX,
    MAX_CONTINUATION_STEPS,
    NUCLEOTIDE_LENGTH,
    REFSEQ_INITIAL_CONTINUATION_STEPS,
    REFSEQ_MIDPOINT_STEPS,
    REFSEQ_TRAIN_DATASET,
    REFSEQ_TRAIN_REVISION,
    REFSEQ_TRAIN_TEXT_KEY,
    RUNTIME_MARGIN,
    SEED,
    SEQUENCE_LENGTH,
    SOURCE_CHECKPOINT,
    ObjectiveKind,
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
from glm_experiments.exp515.module import (
    Exp515Module,
    ScheduleKind,
    checkpoint_next_sample_id,
)
from glm_experiments.exp515.significance import statistically_not_worse_gate
from glm_experiments.exp515.storage import (
    ISSUE_BUCKET_REGION,
    upload_issue_artifact,
)
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


def download_source_checkpoint(
    destination: Path,
    *,
    checkpoint_uri: str = SOURCE_CHECKPOINT,
) -> Path:
    """Download one exact Hugging Face export from GCS."""

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
    prefix = checkpoint_uri.removeprefix("gs://").rstrip("/")
    remote_files = [path for path in filesystem.find(prefix) if not path.endswith("/")]
    for remote in remote_files:
        target = destination / Path(remote).relative_to(prefix)
        target.parent.mkdir(parents=True, exist_ok=True)
        filesystem.get(remote, str(target))
    missing = required - {path.name for path in destination.iterdir()}
    if missing:
        raise FileNotFoundError(f"source checkpoint is incomplete: {sorted(missing)}")
    return destination


def _load_tokenizer(
    source_dir: Path,
    *,
    nucleotide_length: int = NUCLEOTIDE_LENGTH,
    sequence_length: int = SEQUENCE_LENGTH,
    require_bos: bool = True,
) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(source_dir)
    if require_bos and tokenizer.bos_token_id is None:
        raise ValueError("source checkpoint tokenizer lacks BOS")
    if not require_bos and tokenizer.bos_token_id is not None:
        raise ValueError("source checkpoint unexpectedly defines BOS")
    probe = tokenizer("A" * nucleotide_length, add_special_tokens=True)
    if len(probe["input_ids"]) != sequence_length:
        raise ValueError("source tokenizer is not one token per nucleotide")
    if require_bos and probe["input_ids"][0] != tokenizer.bos_token_id:
        raise ValueError("source tokenizer does not prepend its BOS token")
    return tokenizer


def _nucleotide_token_ids(
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, int]:
    """Return the exact singleton IDs for A/C/G/T under one tokenizer."""

    result: dict[str, int] = {}
    for base in "ACGT":
        encoded = tokenizer(base, add_special_tokens=False)["input_ids"]
        if len(encoded) != 1:
            raise ValueError(f"tokenizer does not encode {base} as one token")
        result[base] = int(encoded[0])
    if len(set(result.values())) != 4:
        raise ValueError("nucleotide tokenizer IDs are not distinct")
    return result


def _new_module(
    source_dir: Path,
    *,
    continuation_steps: int,
    plan_sha256: str,
    selector_mode: str,
    selector_ratio: float,
    objective_kind: ObjectiveKind = "hard_ce",
    teacher_dir: Path | None = None,
    schedule_kind: ScheduleKind = "warmup_cosine",
    sample_id_offset: int = 0,
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
        objective_kind=objective_kind,
        teacher_checkpoint=str(teacher_dir) if teacher_dir is not None else None,
        schedule_kind=schedule_kind,
        sample_id_offset=sample_id_offset,
    )


def _checkpoint_step(path: Path) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return int(checkpoint["global_step"])


def _completed_bridge_state(
    root: Path,
) -> tuple[Path, int, dict[str, Any], dict[str, Any]]:
    """Validate and return a completed bridge for an explicit repair resume."""

    smoke = json.loads((root / "smoke-test.json").read_text(encoding="utf-8"))
    if smoke.get("passed") is not True:
        raise ValueError("bridge resume requires a passing smoke test")
    canary = json.loads(
        (root / "canary-20" / "runtime.json").read_text(encoding="utf-8")
    )
    bridge = json.loads((root / "bridge" / "runtime.json").read_text(encoding="utf-8"))
    selected_microbatch = int(canary["microbatch_size"])
    if (
        int(canary["end_global_step"]) != CANARY_STEPS
        or int(bridge["end_global_step"]) != BRIDGE_STEPS
        or int(bridge["microbatch_size"]) != selected_microbatch
    ):
        raise ValueError(
            "bridge resume metadata does not match the registered protocol"
        )
    bridge_checkpoint = root / "bridge" / f"step-{BRIDGE_STEPS}.ckpt"
    if _checkpoint_step(bridge_checkpoint) != BRIDGE_STEPS:
        raise ValueError(
            "bridge resume checkpoint is not at the registered bridge step"
        )
    return bridge_checkpoint, selected_microbatch, canary, bridge


def _module_from_checkpoint(
    source_dir: Path,
    checkpoint: Path,
    *,
    plan_sha256: str,
) -> Exp515Module:
    """Restore model weights for evaluation; arm training restores full state later."""

    module = _new_module(
        source_dir,
        continuation_steps=MAX_CONTINUATION_STEPS,
        plan_sha256=plan_sha256,
        selector_mode="uniform",
        selector_ratio=1.0,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    module.load_state_dict(payload["state_dict"], strict=True)
    return module


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


def _phase_position(
    resume_from: Path | None,
    fork_weights_from: Path | None,
) -> tuple[int, int, int]:
    """Return absolute data start, local optimizer step, and data offset."""

    if resume_from is not None and fork_weights_from is not None:
        raise ValueError("choose full resume or fresh-optimizer fork, not both")
    state_source = resume_from or fork_weights_from
    start_sample = (
        0 if state_source is None else checkpoint_next_sample_id(str(state_source))
    )
    start_step = 0 if resume_from is None else _checkpoint_step(resume_from)
    sample_id_offset = start_sample - start_step * EFFECTIVE_BATCH_SIZE
    if sample_id_offset < 0:
        raise ValueError(
            "checkpoint data position precedes its optimizer-step position"
        )
    return start_sample, start_step, sample_id_offset


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
    fork_weights_from: Path | None = None,
    schedule_kind: ScheduleKind = "warmup_cosine",
    objective_kind: ObjectiveKind = "hard_ce",
    teacher_dir: Path | None = None,
    nucleotide_length: int = NUCLEOTIDE_LENGTH,
    sequence_length: int = SEQUENCE_LENGTH,
) -> tuple[Exp515Module, Path, dict[str, Any]]:
    """Run one exact plan segment and save a full Lightning checkpoint."""

    if EFFECTIVE_BATCH_SIZE % microbatch_size:
        raise ValueError("microbatch must divide the effective batch")
    plan = validate_sequence_plan(plan_dir)
    start_sample, start_step, sample_id_offset = _phase_position(
        resume_from,
        fork_weights_from,
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
        collate_fn=SequenceCollator(
            tokenizer,
            nucleotide_length=nucleotide_length,
        ),
    )
    module = _new_module(
        source_dir,
        continuation_steps=continuation_steps,
        plan_sha256=str(plan["sequences_sha256"]),
        selector_mode=selector_mode,
        selector_ratio=selector_ratio,
        objective_kind=objective_kind,
        teacher_dir=teacher_dir,
        schedule_kind=schedule_kind,
        sample_id_offset=sample_id_offset,
    )
    if fork_weights_from is not None:
        payload = torch.load(fork_weights_from, map_location="cpu", weights_only=False)
        module.load_state_dict(payload["state_dict"], strict=True)
    phase_dir = output_dir / run_name
    phase_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = Exp515Diagnostics(
        phase_dir,
        every_n_steps=10,
        sequence_length=sequence_length,
        nucleotide_token_ids=_nucleotide_token_ids(tokenizer),
    )
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
        weights_only=False,
    )
    elapsed = time.time() - started
    if int(trainer.global_step) != target_global_step:
        raise RuntimeError(
            f"phase ended at step {trainer.global_step}, expected {target_global_step}"
        )
    checkpoint = phase_dir / f"step-{target_global_step}.ckpt"
    trainer.save_checkpoint(checkpoint, weights_only=False)
    module.release_teacher()
    gc.collect()
    torch.cuda.empty_cache()
    metadata = {
        "run_name": run_name,
        "selector_mode": selector_mode,
        "selector_ratio": selector_ratio,
        "objective_kind": objective_kind,
        "start_global_step": start_step,
        "end_global_step": target_global_step,
        "start_sample_id": start_sample,
        "end_sample_id": start_sample + phase_rows,
        "sample_id_offset": sample_id_offset,
        "fresh_optimizer_fork": fork_weights_from is not None,
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
    *,
    nucleotide_length: int = NUCLEOTIDE_LENGTH,
    require_bos: bool = True,
) -> dict[str, Any]:
    """Check direct-HF logits and selector-disabled uniform loss parity."""

    dataset = SequencePlanDataset(plan_dir, start=0, rows=2)
    collator = SequenceCollator(tokenizer, nucleotide_length=nucleotide_length)
    batch = collator([dataset[0], dataset[1]])
    selector_checks = _selector_device_smoke(torch.device("cuda"))
    synthetic_sequence = "A" + "c" + "A" * (nucleotide_length - 2)
    synthetic_batch = collator(
        [{"sample_id": 0, "sequence": synthetic_sequence, "species": "synthetic"}]
    )
    repeat_offset = int(require_bos)
    repeat_alignment_passed = bool(
        not synthetic_batch["soft_masked"][0, repeat_offset]
        and synthetic_batch["soft_masked"][0, repeat_offset + 1]
    )
    all_lowercase_filter_passed = not has_eligible_target(
        "a" * nucleotide_length
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
        "repeat_alignment_passed": repeat_alignment_passed,
        "all_lowercase_filter_passed": all_lowercase_filter_passed,
        "passed": max_logit_error == 0.0
        and loss_error <= 1e-6
        and selector_checks["passed"]
        and repeat_alignment_passed
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

    if relative.parts[0] in {
        "source-checkpoint",
        "teacher-checkpoint",
        "evaluation-cache",
    }:
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

    client = boto3.client("s3", region_name=ISSUE_BUCKET_REGION)
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


def _archive_precompletion_failure(root: Path) -> Path | None:
    """Move a stale failure aside once the same run has a final manifest."""

    failure = root / "failure.json"
    if not failure.exists() or not (root / "final-manifest.json").exists():
        return None
    destination = root / "repair-history" / "pre-completion-failure.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != failure.read_bytes():
            raise FileExistsError(
                "repair-history failure record already exists with different content"
            )
        failure.unlink()
    else:
        failure.replace(destination)
    return destination


def run_experiment(
    artifact_dir: Path,
    *,
    experiment_commit: str,
    run_id: str,
    resume_from_bridge: bool = False,
) -> None:
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

    if resume_from_bridge:
        bridge_checkpoint, selected_microbatch, canary, bridge = (
            _completed_bridge_state(root)
        )
        bridge_module = _module_from_checkpoint(
            source_dir,
            bridge_checkpoint,
            plan_sha256=str(plan["sequences_sha256"]),
        )
    else:
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


def run_refseq_screen(
    artifact_dir: Path,
    *,
    experiment_commit: str,
    run_id: str,
    resume_from_bridge: bool = False,
) -> None:
    """Run the fresh RefSeq five-arm screen with a statistical midpoint gate."""

    if len(experiment_commit) != 40:
        raise ValueError("experiment commit must be a full SHA")
    seed_everything(SEED, workers=True)
    instance_start = float(os.getenv("EXP515_INSTANCE_START_UNIX", time.time()))
    screen_started = time.time()
    root = artifact_dir / run_id
    _install_compute_guard(instance_start)
    root.mkdir(parents=True, exist_ok=True)

    source_dir = download_source_checkpoint(root / "source-checkpoint")
    tokenizer = _load_tokenizer(source_dir)
    plan_dir = root / "sequence-plan"
    maximum_global_step = BRIDGE_STEPS + REFSEQ_INITIAL_CONTINUATION_STEPS
    plan = build_sequence_plan(
        plan_dir,
        rows=maximum_global_step * EFFECTIVE_BATCH_SIZE,
        seed=SEED,
        dataset=REFSEQ_TRAIN_DATASET,
        revision=REFSEQ_TRAIN_REVISION,
        text_key=REFSEQ_TRAIN_TEXT_KEY,
        species_key=None,
    )
    _write_json(
        root / "refseq-plan-audit.json",
        {
            "dataset": plan["dataset"],
            "revision": plan["revision"],
            "rows": plan["rows"],
            "nucleotide_length": plan["nucleotide_length"],
            "filtered_all_lowercase": plan["filtered_all_lowercase"],
            "filtered_wrong_length": plan["filtered_wrong_length"],
            "lowercase_bases": plan["lowercase_bases"],
            "total_bases": plan["total_bases"],
            "lowercase_base_fraction": plan["lowercase_base_fraction"],
            "sequence_plan_sha256": plan["sequences_sha256"],
            "repeat_definition": "lowercase source-assembly soft masking",
        },
    )

    if resume_from_bridge:
        bridge_checkpoint, selected_microbatch, canary, bridge = (
            _completed_bridge_state(root)
        )
        bridge_module = _module_from_checkpoint(
            source_dir,
            bridge_checkpoint,
            plan_sha256=str(plan["sequences_sha256"]),
        )
    else:
        smoke_test(source_dir, tokenizer, plan_dir, root / "smoke-test.json")
        selected_microbatch = 0
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
                    continuation_steps=REFSEQ_INITIAL_CONTINUATION_STEPS,
                    target_global_step=1,
                    microbatch_size=candidate,
                    resume_from=None,
                    schedule_kind="warmup_constant",
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
                break
        if not selected_microbatch:
            raise RuntimeError("no power-of-two microbatch passed the 85% HBM gate")

        canary_module, _, canary = run_training_phase(
            source_dir=source_dir,
            tokenizer=tokenizer,
            plan_dir=plan_dir,
            output_dir=root,
            run_name="canary-20",
            selector_mode="uniform",
            selector_ratio=1.0,
            continuation_steps=REFSEQ_INITIAL_CONTINUATION_STEPS,
            target_global_step=CANARY_STEPS,
            microbatch_size=selected_microbatch,
            resume_from=None,
            schedule_kind="warmup_constant",
        )
        del canary_module
        gc.collect()
        torch.cuda.empty_cache()
        bridge_module, bridge_checkpoint, bridge = run_training_phase(
            source_dir=source_dir,
            tokenizer=tokenizer,
            plan_dir=plan_dir,
            output_dir=root,
            run_name="bridge",
            selector_mode="uniform",
            selector_ratio=1.0,
            continuation_steps=REFSEQ_INITIAL_CONTINUATION_STEPS,
            target_global_step=BRIDGE_STEPS,
            microbatch_size=selected_microbatch,
            resume_from=None,
            schedule_kind="warmup_constant",
        )

    evaluation_frame = load_promoter_frame(root / "evaluation-cache")
    bridge_csv = root / "evaluations" / "bridge.csv"
    bridge_eval = evaluate_promoter_auprc(
        bridge_module.net,
        tokenizer,
        frame=evaluation_frame,
        output_path=bridge_csv,
        batch_size=selected_microbatch,
        checkpoint_name="bridge",
    )
    del bridge_module
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - instance_start
    worst_case_training_seconds = (
        len(ARMS)
        * REFSEQ_INITIAL_CONTINUATION_STEPS
        * float(canary["seconds_per_optimizer_step"])
    )
    worst_case_evaluations = len(ARMS) * 2
    projected_remaining_seconds = RUNTIME_MARGIN * (
        worst_case_training_seconds
        + worst_case_evaluations * float(bridge_eval["elapsed_seconds"])
    )
    projected_total_cost = (
        (elapsed + projected_remaining_seconds) / 3600 * GPU_PRICE_PER_HOUR_USD
    )
    projection = {
        "instance_elapsed_seconds": elapsed,
        "canary_seconds_per_optimizer_step": canary["seconds_per_optimizer_step"],
        "worst_case_arm_steps": (len(ARMS) * REFSEQ_INITIAL_CONTINUATION_STEPS),
        "worst_case_remaining_evaluations": worst_case_evaluations,
        "runtime_margin": RUNTIME_MARGIN,
        "projected_remaining_seconds": projected_remaining_seconds,
        "projected_total_compute_usd": projected_total_cost,
        "gpu_compute_cap_usd": GPU_COMPUTE_CAP_USD,
        "all_in_cap_usd": ALL_IN_CAP_USD,
        "passed": projected_total_cost < GPU_COMPUTE_CAP_USD,
    }
    _write_json(root / "budget-projection.json", projection)
    if not projection["passed"]:
        raise RuntimeError("RefSeq screen projection reaches the compute stop")

    midpoint_global_step = BRIDGE_STEPS + REFSEQ_MIDPOINT_STEPS
    endpoint_global_step = BRIDGE_STEPS + REFSEQ_INITIAL_CONTINUATION_STEPS
    summaries = [bridge_eval]
    runtimes = [canary, bridge]
    midpoint_checkpoints: dict[str, Path] = {}
    midpoint_evaluations: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        run_name = f"{arm.name}-midpoint"
        module, checkpoint, runtime = run_training_phase(
            source_dir=source_dir,
            tokenizer=tokenizer,
            plan_dir=plan_dir,
            output_dir=root,
            run_name=run_name,
            selector_mode=arm.selector_mode,
            selector_ratio=arm.selector_ratio,
            continuation_steps=REFSEQ_INITIAL_CONTINUATION_STEPS,
            target_global_step=midpoint_global_step,
            microbatch_size=selected_microbatch,
            resume_from=bridge_checkpoint,
            schedule_kind="warmup_constant",
        )
        evaluation = evaluate_promoter_auprc(
            module.net,
            tokenizer,
            frame=evaluation_frame,
            output_path=root / "evaluations" / f"{run_name}.csv",
            batch_size=selected_microbatch,
            checkpoint_name=run_name,
        )
        midpoint_checkpoints[arm.name] = checkpoint
        midpoint_evaluations[arm.name] = evaluation
        summaries.append(evaluation)
        runtimes.append(runtime)
        del module
        gc.collect()
        torch.cuda.empty_cache()

    gate = statistically_not_worse_gate(
        bridge_csv,
        {
            arm.name: root / "evaluations" / f"{arm.name}-midpoint.csv"
            for arm in ARMS
            if arm.name != "uniform-100"
        },
        permutations=20_000,
        alpha=0.05,
        seed=SEED + 5000,
    )
    _write_json(root / "midpoint-gate.json", gate)

    endpoint_evaluations: dict[str, dict[str, Any]] = {}
    endpoint_checkpoints: dict[str, Path] = {}
    for arm in ARMS:
        continue_to_endpoint = arm.name == "uniform-100" or bool(
            gate["tests"][arm.name]["continue_to_endpoint"]
        )
        if not continue_to_endpoint:
            _write_json(
                root / f"{arm.name}-result.json",
                {
                    "arm": arm.name,
                    "bridge_auprc": bridge_eval["auprc"],
                    "midpoint_auprc": midpoint_evaluations[arm.name]["auprc"],
                    "continued_to_endpoint": False,
                    "gate": gate["tests"][arm.name],
                },
            )
            continue
        run_name = f"{arm.name}-endpoint"
        module, checkpoint, runtime = run_training_phase(
            source_dir=source_dir,
            tokenizer=tokenizer,
            plan_dir=plan_dir,
            output_dir=root,
            run_name=run_name,
            selector_mode=arm.selector_mode,
            selector_ratio=arm.selector_ratio,
            continuation_steps=REFSEQ_INITIAL_CONTINUATION_STEPS,
            target_global_step=endpoint_global_step,
            microbatch_size=selected_microbatch,
            resume_from=midpoint_checkpoints[arm.name],
            schedule_kind="warmup_constant",
        )
        evaluation = evaluate_promoter_auprc(
            module.net,
            tokenizer,
            frame=evaluation_frame,
            output_path=root / "evaluations" / f"{run_name}.csv",
            batch_size=selected_microbatch,
            checkpoint_name=run_name,
        )
        endpoint_checkpoints[arm.name] = checkpoint
        endpoint_evaluations[arm.name] = evaluation
        summaries.append(evaluation)
        runtimes.append(runtime)
        _write_json(
            root / f"{arm.name}-result.json",
            {
                "arm": arm.name,
                "bridge_auprc": bridge_eval["auprc"],
                "midpoint_auprc": midpoint_evaluations[arm.name]["auprc"],
                "continued_to_endpoint": True,
                "endpoint_auprc": evaluation["auprc"],
                "endpoint_checkpoint": str(checkpoint),
                "gate": gate["tests"].get(arm.name),
            },
        )
        del module
        gc.collect()
        torch.cuda.empty_cache()

    now = time.time()
    final = {
        "status": "complete",
        "protocol": "refseq-250-500-warmup-constant",
        "experiment_commit": experiment_commit,
        "run_id": run_id,
        "provider": "Lambda Cloud",
        "accelerator": ACCELERATOR,
        "hourly_rate_usd": GPU_PRICE_PER_HOUR_USD,
        "instance_start_unix": instance_start,
        "instance_start_utc": _utc_timestamp(instance_start),
        "screen_started_utc": _utc_timestamp(screen_started),
        "completed_at_utc": _utc_timestamp(now),
        "continuation_steps": REFSEQ_INITIAL_CONTINUATION_STEPS,
        "midpoint_steps": REFSEQ_MIDPOINT_STEPS,
        "microbatch_size": selected_microbatch,
        "schedule": {
            "warmup_steps": BRIDGE_STEPS,
            "warmup_start_learning_rate": 1e-5,
            "peak_and_constant_learning_rate": 1e-3,
        },
        "dataset": {
            "name": REFSEQ_TRAIN_DATASET,
            "revision": REFSEQ_TRAIN_REVISION,
            "text_key": REFSEQ_TRAIN_TEXT_KEY,
            "sequence_plan_sha256": plan["sequences_sha256"],
            "lowercase_base_fraction": plan["lowercase_base_fraction"],
        },
        "bridge_auprc": bridge_eval["auprc"],
        "midpoint_gate": gate,
        "completed_evaluations": summaries,
        "runtimes": runtimes,
        "screen_elapsed_seconds": now - screen_started,
        "estimated_screen_compute_usd": (
            (now - screen_started) / 3600 * GPU_PRICE_PER_HOUR_USD
        ),
        "provider_allocation_elapsed_seconds": now - instance_start,
        "estimated_provider_allocation_usd": (
            (now - instance_start) / 3600 * GPU_PRICE_PER_HOUR_USD
        ),
        "compute_cap_usd": GPU_COMPUTE_CAP_USD,
        "all_in_cap_usd": ALL_IN_CAP_USD,
    }
    _write_json(root / "final-manifest.json", final)


def _checkpoint_contract(student_dir: Path, teacher_dir: Path) -> dict[str, Any]:
    """Verify that exp58 student and teacher exports share one exact contract."""

    student_config = json.loads((student_dir / "config.json").read_text())
    teacher_config = json.loads((teacher_dir / "config.json").read_text())
    fields = (
        "model_type",
        "vocab_size",
        "max_position_embeddings",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
    )
    differences = {
        field: {
            "student": student_config.get(field),
            "teacher": teacher_config.get(field),
        }
        for field in fields
        if student_config.get(field) != teacher_config.get(field)
    }
    tokenizer_equal = (student_dir / "tokenizer.json").read_bytes() == (
        teacher_dir / "tokenizer.json"
    ).read_bytes()
    payload = {
        "checked_fields": list(fields),
        "differences": differences,
        "tokenizer_json_equal": tokenizer_equal,
        "passed": not differences and tokenizer_equal,
    }
    if not payload["passed"]:
        raise ValueError(f"student/teacher checkpoint contract mismatch: {payload}")
    return payload


def run_cds_gate(
    artifact_dir: Path,
    *,
    experiment_commit: str,
    run_id: str,
    resume_from_bridge: bool = False,
) -> None:
    """Run the exp58 step-1k CDS bridge and seven 100-step arm gate."""

    if len(experiment_commit) != 40:
        raise ValueError("experiment commit must be a full SHA")
    seed_everything(SEED, workers=True)
    instance_start = float(os.getenv("EXP515_INSTANCE_START_UNIX", time.time()))
    screen_started = time.time()
    root = artifact_dir / run_id
    _install_compute_guard(instance_start)
    root.mkdir(parents=True, exist_ok=True)

    source_dir = download_source_checkpoint(
        root / "source-checkpoint",
        checkpoint_uri=EXP58_STUDENT_CHECKPOINT,
    )
    teacher_dir = download_source_checkpoint(
        root / "teacher-checkpoint",
        checkpoint_uri=EXP58_TEACHER_CHECKPOINT,
    )
    _write_json(
        root / "checkpoint-contract.json",
        _checkpoint_contract(source_dir, teacher_dir),
    )
    tokenizer = _load_tokenizer(
        source_dir,
        nucleotide_length=CDS_NUCLEOTIDE_LENGTH,
        sequence_length=CDS_SEQUENCE_LENGTH,
        require_bos=False,
    )
    plan_dir = root / "sequence-plan"
    gate_global_step = BRIDGE_STEPS + CDS_GATE_CONTINUATION_STEPS
    plan = build_sequence_plan(
        plan_dir,
        rows=gate_global_step * EFFECTIVE_BATCH_SIZE,
        seed=SEED,
        dataset=EXP58_TRAIN_DATASET,
        revision=EXP58_TRAIN_REVISION,
        text_key=EXP58_TRAIN_TEXT_KEY,
        species_key=None,
        nucleotide_length=CDS_NUCLEOTIDE_LENGTH,
        exclude_first_base_from_eligibility=True,
    )
    _write_json(
        root / "cds-plan-audit.json",
        {
            "dataset": plan["dataset"],
            "revision": plan["revision"],
            "rows": plan["rows"],
            "nucleotide_length": plan["nucleotide_length"],
            "filtered_all_lowercase": plan["filtered_all_lowercase"],
            "filtered_wrong_length": plan["filtered_wrong_length"],
            "lowercase_bases": plan["lowercase_bases"],
            "total_bases": plan["total_bases"],
            "lowercase_base_fraction": plan["lowercase_base_fraction"],
            "sequence_plan_sha256": plan["sequences_sha256"],
            "repeat_policy": "lowercase targets excluded in every arm",
        },
    )

    if resume_from_bridge:
        bridge_checkpoint, selected_microbatch, teacher_canary, bridge = (
            _completed_bridge_state(root)
        )
        bridge_module = _module_from_checkpoint(
            source_dir,
            bridge_checkpoint,
            plan_sha256=str(plan["sequences_sha256"]),
        )
    else:
        smoke_test(
            source_dir,
            tokenizer,
            plan_dir,
            root / "smoke-test.json",
            nucleotide_length=CDS_NUCLEOTIDE_LENGTH,
            require_bos=False,
        )
        selected_microbatch = 0
        for candidate in (64, 32, 16, 8, 4, 2, 1):
            try:
                candidate_module, _, metadata = run_training_phase(
                    source_dir=source_dir,
                    tokenizer=tokenizer,
                    plan_dir=plan_dir,
                    output_dir=root / "canary-selection",
                    run_name=f"teacher-kl-microbatch-{candidate}",
                    selector_mode="uniform",
                    selector_ratio=1.0,
                    continuation_steps=CDS_GATE_CONTINUATION_STEPS,
                    target_global_step=1,
                    microbatch_size=candidate,
                    resume_from=None,
                    schedule_kind="warmup_constant",
                    objective_kind="teacher_kl",
                    teacher_dir=teacher_dir,
                    nucleotide_length=CDS_NUCLEOTIDE_LENGTH,
                    sequence_length=CDS_SEQUENCE_LENGTH,
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
                break
        if not selected_microbatch:
            raise RuntimeError("no power-of-two teacher microbatch passed the HBM gate")

        teacher_canary_module, _, teacher_canary = run_training_phase(
            source_dir=source_dir,
            tokenizer=tokenizer,
            plan_dir=plan_dir,
            output_dir=root,
            run_name="canary-20",
            selector_mode="uniform",
            selector_ratio=1.0,
            continuation_steps=CDS_GATE_CONTINUATION_STEPS,
            target_global_step=CANARY_STEPS,
            microbatch_size=selected_microbatch,
            resume_from=None,
            schedule_kind="warmup_constant",
            objective_kind="teacher_kl",
            teacher_dir=teacher_dir,
            nucleotide_length=CDS_NUCLEOTIDE_LENGTH,
            sequence_length=CDS_SEQUENCE_LENGTH,
        )
        del teacher_canary_module
        gc.collect()
        torch.cuda.empty_cache()
        bridge_module, bridge_checkpoint, bridge = run_training_phase(
            source_dir=source_dir,
            tokenizer=tokenizer,
            plan_dir=plan_dir,
            output_dir=root,
            run_name="bridge",
            selector_mode="uniform",
            selector_ratio=1.0,
            continuation_steps=CDS_GATE_CONTINUATION_STEPS,
            target_global_step=BRIDGE_STEPS,
            microbatch_size=selected_microbatch,
            resume_from=None,
            schedule_kind="warmup_constant",
            nucleotide_length=CDS_NUCLEOTIDE_LENGTH,
            sequence_length=CDS_SEQUENCE_LENGTH,
        )

    evaluation_frame = load_promoter_frame(
        root / "evaluation-cache",
        subsets=("missense_variant", "splicing"),
        nucleotide_length=CDS_NUCLEOTIDE_LENGTH,
    )
    bridge_csv = root / "evaluations" / "bridge.csv"
    bridge_eval = evaluate_promoter_auprc(
        bridge_module.net,
        tokenizer,
        frame=evaluation_frame,
        output_path=bridge_csv,
        batch_size=selected_microbatch,
        checkpoint_name="bridge",
        nucleotide_length=CDS_NUCLEOTIDE_LENGTH,
        sequence_length=CDS_SEQUENCE_LENGTH,
    )
    del bridge_module
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - instance_start
    hard_seconds_per_step = float(bridge["seconds_per_optimizer_step"])
    teacher_seconds_per_step = float(teacher_canary["seconds_per_optimizer_step"])
    projected_remaining_seconds = RUNTIME_MARGIN * (
        5 * CDS_GATE_CONTINUATION_STEPS * hard_seconds_per_step
        + 2 * CDS_GATE_CONTINUATION_STEPS * teacher_seconds_per_step
        + len(CDS_ARMS) * float(bridge_eval["elapsed_seconds"])
    )
    projected_total_cost = (
        (elapsed + projected_remaining_seconds) / 3600 * GPU_PRICE_PER_HOUR_USD
    )
    projection = {
        "instance_elapsed_seconds": elapsed,
        "hard_seconds_per_optimizer_step": hard_seconds_per_step,
        "teacher_seconds_per_optimizer_step": teacher_seconds_per_step,
        "remaining_arm_steps": len(CDS_ARMS) * CDS_GATE_CONTINUATION_STEPS,
        "projected_remaining_seconds": projected_remaining_seconds,
        "projected_total_compute_usd": projected_total_cost,
        "gpu_compute_cap_usd": GPU_COMPUTE_CAP_USD,
        "all_in_cap_usd": ALL_IN_CAP_USD,
        "passed": projected_total_cost < GPU_COMPUTE_CAP_USD,
    }
    _write_json(root / "budget-projection.json", projection)
    if not projection["passed"]:
        raise RuntimeError("CDS gate projection reaches the compute stop")

    summaries = [bridge_eval]
    runtimes = [teacher_canary, bridge]
    arm_evaluations: dict[str, dict[str, Any]] = {}
    for arm in CDS_ARMS:
        module, checkpoint, runtime = run_training_phase(
            source_dir=source_dir,
            tokenizer=tokenizer,
            plan_dir=plan_dir,
            output_dir=root,
            run_name=f"{arm.name}-gate-100",
            selector_mode=arm.selector_mode,
            selector_ratio=arm.selector_ratio,
            continuation_steps=CDS_GATE_CONTINUATION_STEPS,
            target_global_step=CDS_GATE_CONTINUATION_STEPS,
            microbatch_size=selected_microbatch,
            resume_from=None,
            fork_weights_from=bridge_checkpoint,
            schedule_kind="arm_warmup_constant",
            objective_kind=arm.objective_kind,
            teacher_dir=(teacher_dir if arm.objective_kind != "hard_ce" else None),
            nucleotide_length=CDS_NUCLEOTIDE_LENGTH,
            sequence_length=CDS_SEQUENCE_LENGTH,
        )
        evaluation = evaluate_promoter_auprc(
            module.net,
            tokenizer,
            frame=evaluation_frame,
            output_path=root / "evaluations" / f"{arm.name}-gate-100.csv",
            batch_size=selected_microbatch,
            checkpoint_name=f"{arm.name}-gate-100",
            nucleotide_length=CDS_NUCLEOTIDE_LENGTH,
            sequence_length=CDS_SEQUENCE_LENGTH,
        )
        arm_evaluations[arm.name] = evaluation
        summaries.append(evaluation)
        runtimes.append(runtime)
        _write_json(
            root / f"{arm.name}-result.json",
            {
                "arm": arm.name,
                "objective_kind": arm.objective_kind,
                "bridge_auprc": bridge_eval["auprc"],
                "gate_100_auprc": evaluation["auprc"],
                "subsets": evaluation["subsets"],
                "checkpoint": str(checkpoint),
            },
        )
        del module
        gc.collect()
        torch.cuda.empty_cache()

    gate = statistically_not_worse_gate(
        bridge_csv,
        {
            arm.name: root / "evaluations" / f"{arm.name}-gate-100.csv"
            for arm in CDS_ARMS
            if arm.name != "uniform-100"
        },
        permutations=20_000,
        alpha=0.05,
        seed=SEED + 5800,
    )
    _write_json(root / "gate-100-significance.json", gate)
    now = time.time()
    final = {
        "status": "complete",
        "protocol": "exp58-animals-step1000-100-bridge-100-seven-arm-gate",
        "experiment_commit": experiment_commit,
        "run_id": run_id,
        "provider": "Lambda Cloud",
        "accelerator": ACCELERATOR,
        "hourly_rate_usd": GPU_PRICE_PER_HOUR_USD,
        "instance_start_unix": instance_start,
        "instance_start_utc": _utc_timestamp(instance_start),
        "screen_started_utc": _utc_timestamp(screen_started),
        "completed_at_utc": _utc_timestamp(now),
        "source_checkpoint": EXP58_STUDENT_CHECKPOINT,
        "teacher_checkpoint": EXP58_TEACHER_CHECKPOINT,
        "bridge_steps": BRIDGE_STEPS,
        "arm_steps": CDS_GATE_CONTINUATION_STEPS,
        "microbatch_size": selected_microbatch,
        "schedule": {
            "bridge_warmup_steps": BRIDGE_STEPS,
            "arm_warmup_steps": CDS_ARM_WARMUP_STEPS,
            "warmup_start_learning_rate": 1e-5,
            "peak_and_constant_learning_rate": 1e-3,
            "arm_optimizer_state": "fresh AdamW for every arm",
        },
        "dataset": {
            "name": EXP58_TRAIN_DATASET,
            "revision": EXP58_TRAIN_REVISION,
            "text_key": EXP58_TRAIN_TEXT_KEY,
            "nucleotide_length": CDS_NUCLEOTIDE_LENGTH,
            "bos_token": None,
            "sequence_plan_sha256": plan["sequences_sha256"],
            "lowercase_base_fraction": plan["lowercase_base_fraction"],
        },
        "evaluation_subsets": [
            "missense_variant",
            "splicing",
        ],
        "bridge_evaluation": bridge_eval,
        "arm_evaluations": arm_evaluations,
        "gate_100_significance": gate,
        "completed_evaluations": summaries,
        "runtimes": runtimes,
        "screen_elapsed_seconds": now - screen_started,
        "estimated_screen_compute_usd": (
            (now - screen_started) / 3600 * GPU_PRICE_PER_HOUR_USD
        ),
        "provider_allocation_elapsed_seconds": now - instance_start,
        "estimated_provider_allocation_usd": (
            (now - instance_start) / 3600 * GPU_PRICE_PER_HOUR_USD
        ),
        "compute_cap_usd": GPU_COMPUTE_CAP_USD,
        "all_in_cap_usd": ALL_IN_CAP_USD,
    }
    _write_json(root / "final-manifest.json", final)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--experiment-commit", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume-from-bridge", action="store_true")
    parser.add_argument("--publish-only", action="store_true")
    parser.add_argument("--refseq-screen", action="store_true")
    parser.add_argument("--cds-gate", action="store_true")
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in {".", ".."}:
        raise ValueError("run ID must be one safe path component")
    root = args.artifact_dir / args.run_id
    if args.publish_only:
        _archive_precompletion_failure(root)
        publish_run_artifacts(root, args.run_id)
        return
    try:
        if args.refseq_screen and args.cds_gate:
            raise ValueError("choose only one of --refseq-screen and --cds-gate")
        if args.cds_gate:
            run_cds_gate(
                args.artifact_dir,
                experiment_commit=args.experiment_commit,
                run_id=args.run_id,
                resume_from_bridge=args.resume_from_bridge,
            )
        elif args.refseq_screen:
            run_refseq_screen(
                args.artifact_dir,
                experiment_commit=args.experiment_commit,
                run_id=args.run_id,
                resume_from_bridge=args.resume_from_bridge,
            )
        else:
            run_experiment(
                args.artifact_dir,
                experiment_commit=args.experiment_commit,
                run_id=args.run_id,
                resume_from_bridge=args.resume_from_bridge,
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
            _archive_precompletion_failure(root)
            publish_run_artifacts(root, args.run_id)
        finally:
            signal.alarm(0)


if __name__ == "__main__":
    main()
