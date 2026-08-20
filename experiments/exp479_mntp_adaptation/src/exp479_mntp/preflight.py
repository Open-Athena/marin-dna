"""Behavioral, optimizer, memory, throughput, and budget preflight on GH200."""

from __future__ import annotations

import json
import os
import time
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from exp479_mntp.config import (
    BUDGET_USD,
    LAMBDA_GH200_PRICE_PER_HOUR_USD,
    SEQUENCE_LENGTH,
    TRAIN_STEPS,
    optimizer_hyperparameters,
)
from exp479_mntp.loss import per_sequence_weighted_loss
from exp479_mntp.masking import corrupt_for_mntp
from exp479_mntp.modeling import (
    add_mask_token,
    canonical_token_ids,
    load_model_bundle,
    model_logits,
)
from exp479_mntp.optimizer import build_optimizer, build_wsd_scheduler


def enable_training_determinism() -> None:
    """Match the CUDA backend flags set by ``Trainer(deterministic=True)``."""

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


BATCH_CANDIDATES = (1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1)


def _encoded_pair(
    tokenizer: Any, mask_token_id: int
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    first = "ACGTACGTACGTACGT"
    second = first[:-1] + "A"
    encoded = tokenizer(
        [first, second],
        add_special_tokens=True,
        padding=True,
        return_tensors="pt",
    )
    target_position = 7
    flank_position = 16
    encoded["input_ids"][:, target_position] = mask_token_id
    return encoded["input_ids"], encoded["attention_mask"], target_position - 1, flank_position


def behavioral_attention_check(model: Any, tokenizer: Any, mask_token_id: int) -> dict[str, float]:
    """Prove causal/full attention from logits and position-specific gradients."""

    model.eval()
    device = next(model.parameters()).device
    input_ids, attention_mask, readout_position, flank_position = _encoded_pair(
        tokenizer, mask_token_id
    )
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        causal = model_logits(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            attention_mode="causal",
        )
        full = model_logits(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            attention_mode="full",
        )
    causal_delta = torch.max(torch.abs(causal[0, readout_position] - causal[1, readout_position]))
    full_delta = torch.max(torch.abs(full[0, readout_position] - full[1, readout_position]))
    if causal_delta.item() != 0.0:
        raise AssertionError(f"causal right-flank logit delta is nonzero: {causal_delta.item()}")
    if full_delta.item() <= 0.0:
        raise AssertionError("full-attention right-flank logit delta is zero")

    original_requires_grad = [parameter.requires_grad for parameter in model.parameters()]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    gradient_norms: dict[str, float] = {}
    try:
        for mode in ("causal", "full"):
            embeddings = model.get_input_embeddings()(input_ids[:1]).detach().requires_grad_(True)
            logits = model_logits(
                model,
                inputs_embeds=embeddings,
                attention_mask=attention_mask[:1],
                attention_mode=mode,
            )
            logits[0, readout_position, 3].backward()
            gradient_norms[mode] = float(
                torch.linalg.vector_norm(embeddings.grad[0, flank_position])
            )
    finally:
        for parameter, requires_grad in zip(
            model.parameters(), original_requires_grad, strict=True
        ):
            parameter.requires_grad_(requires_grad)

    if gradient_norms["causal"] != 0.0:
        raise AssertionError(f"causal right-flank gradient is nonzero: {gradient_norms['causal']}")
    if gradient_norms["full"] <= 0.0:
        raise AssertionError("full-attention right-flank gradient is zero")
    return {
        "causal_right_flank_logit_delta": float(causal_delta),
        "full_right_flank_logit_delta": float(full_delta),
        "causal_right_flank_gradient_norm": gradient_norms["causal"],
        "full_right_flank_gradient_norm": gradient_norms["full"],
    }


def causal_logit_parity(model: Any, tokenizer: Any) -> float:
    """Compare default Qwen3 causal logits with an explicit causal forward."""

    model.eval()
    device = next(model.parameters()).device
    encoded = tokenizer("ACGT" * 32, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    with torch.no_grad():
        default = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        explicit = model_logits(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            attention_mode="causal",
        )
    delta = torch.max(torch.abs(default - explicit)).item()
    if delta != 0.0:
        raise AssertionError(f"explicit causal logits differ from checkpoint defaults by {delta}")
    return float(delta)


def _synthetic_batch(
    batch_size: int,
    *,
    mask_token_id: int,
    canonical_ids: tuple[int, ...],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    input_ids = torch.empty((batch_size, SEQUENCE_LENGTH), dtype=torch.long)
    input_ids[:, 0] = 2
    bases = torch.tensor(canonical_ids, dtype=torch.long)
    input_ids[:, 1:] = bases[torch.arange(SEQUENCE_LENGTH - 1) % len(bases)]
    lowercase = torch.zeros_like(input_ids, dtype=torch.bool)
    sample_ids = torch.arange(batch_size, dtype=torch.long)
    corrupted = corrupt_for_mntp(
        input_ids,
        lowercase,
        sample_ids,
        mask_token_id=mask_token_id,
        canonical_token_ids=canonical_ids,
        seed=479,
    )
    return {
        "input_ids": corrupted.input_ids.to(device),
        "attention_mask": torch.ones_like(input_ids).to(device),
        "labels": corrupted.labels.to(device),
        "loss_weights": corrupted.loss_weights.to(device),
    }


def _optimizer_step(
    model: Any,
    optimizer: Any,
    batch: dict[str, torch.Tensor],
) -> float:
    optimizer.zero_grad(set_to_none=True)
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if batch["input_ids"].device.type == "cuda"
        else nullcontext()
    )
    with autocast:
        logits = model_logits(
            model,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            attention_mode="full",
        )
        metrics = per_sequence_weighted_loss(logits, batch["labels"], batch["loss_weights"])
    metrics.loss.backward()
    if not torch.isfinite(metrics.loss):
        raise AssertionError(f"non-finite preflight loss: {metrics.loss}")
    if any(
        parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    ):
        raise AssertionError("non-finite preflight gradient")
    optimizer.step()
    return float(metrics.loss.detach())


def select_batch_and_measure(
    model: Any, mask_token_id: int, canonical_ids: tuple[int, ...]
) -> dict[str, Any]:
    """Select the largest candidate whose first update leaves 10% memory headroom."""

    if not torch.cuda.is_available():
        raise RuntimeError("GH200 preflight requires CUDA")
    device = torch.device("cuda")
    model.to(device)
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    total_memory = torch.cuda.get_device_properties(device).total_memory
    attempts: list[dict[str, Any]] = []

    for batch_size in BATCH_CANDIDATES:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        hyperparameters = optimizer_hyperparameters(batch_size)
        optimizer = build_optimizer(model, hyperparameters)
        scheduler = build_wsd_scheduler(
            optimizer,
            warmup_steps=100,
            cooldown_start_step=800,
            total_steps=1_000,
        )
        batch = _synthetic_batch(
            batch_size,
            mask_token_id=mask_token_id,
            canonical_ids=canonical_ids,
            device=device,
        )
        try:
            linear_weight = next(
                parameter
                for module in model.modules()
                if isinstance(module, torch.nn.Linear)
                for parameter in module.parameters(recurse=False)
                if parameter.ndim == 2
            )
            norm_before = float(torch.linalg.vector_norm(linear_weight))
            loss = _optimizer_step(model, optimizer, batch)
            scheduler.step()
            torch.cuda.synchronize()
            peak = int(torch.cuda.max_memory_allocated())
            norm_after = float(torch.linalg.vector_norm(linear_weight))
            norm_relative_error = abs(norm_after - norm_before) / norm_before
            if norm_relative_error > 2e-6:
                raise AssertionError(f"AdamH norm drift {norm_relative_error}")
            headroom = 1.0 - peak / total_memory
            attempt = {
                "batch_size": batch_size,
                "status": "ok",
                "loss": loss,
                "peak_allocated_bytes": peak,
                "total_memory_bytes": total_memory,
                "headroom_fraction": headroom,
                "adamh_norm_relative_error": norm_relative_error,
            }
            attempts.append(attempt)
            if headroom < 0.10:
                continue

            durations: list[float] = []
            for _ in range(3):
                started = time.perf_counter()
                _optimizer_step(model, optimizer, batch)
                scheduler.step()
                torch.cuda.synchronize()
                durations.append(time.perf_counter() - started)
            seconds_per_step = sum(durations) / len(durations)
            projected_training_hours = seconds_per_step * TRAIN_STEPS * 3 / 3600
            instance_start = float(os.getenv("EXP479_INSTANCE_START_UNIX", time.time()))
            accrued_hours = max(0.0, (time.time() - instance_start) / 3600)
            evaluation_reserve_usd = 10.0
            projected_total_cost = (
                accrued_hours + 1.10 * projected_training_hours
            ) * LAMBDA_GH200_PRICE_PER_HOUR_USD + evaluation_reserve_usd
            attempt.update(
                seconds_per_step=seconds_per_step,
                model_tokens_per_second=batch_size * SEQUENCE_LENGTH / seconds_per_step,
                projected_training_hours=projected_training_hours,
                accrued_hours=accrued_hours,
                evaluation_reserve_usd=evaluation_reserve_usd,
                projected_total_cost_usd=projected_total_cost,
            )
            if projected_total_cost >= BUDGET_USD:
                raise RuntimeError(
                    f"projected exp479 cost ${projected_total_cost:.2f} exceeds ${BUDGET_USD:.2f} cap"
                )
            return {"selected": attempt, "attempts": attempts}
        except torch.OutOfMemoryError:
            attempts.append({"batch_size": batch_size, "status": "oom"})
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue

    raise RuntimeError(f"no batch candidate left 10% GH200 memory headroom: {attempts}")


def run_preflight(output_path: Path) -> dict[str, Any]:
    """Run all actual-checkpoint checks that must precede paid training arms."""

    started = time.time()
    enable_training_determinism()
    bundle = load_model_bundle(initialization="transferred", add_mask=False)
    if not torch.cuda.is_available():
        raise RuntimeError("actual-checkpoint preflight requires the Lambda GH200 CUDA device")
    bundle.model.to("cuda")
    parity_delta = causal_logit_parity(bundle.model, bundle.tokenizer)
    mask_token_id, tied = add_mask_token(bundle.model, bundle.tokenizer)
    ids = canonical_token_ids(bundle.tokenizer)
    attention = behavioral_attention_check(bundle.model, bundle.tokenizer, mask_token_id)
    memory = select_batch_and_measure(bundle.model, mask_token_id, ids)
    result = {
        "status": "passed",
        "started_at_utc": datetime.fromtimestamp(started, UTC).isoformat(),
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "causal_logit_parity_max_abs_delta": parity_delta,
        "input_output_tied": tied,
        "mask_token_id": mask_token_id,
        "attention": attention,
        "memory_and_throughput": memory,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
