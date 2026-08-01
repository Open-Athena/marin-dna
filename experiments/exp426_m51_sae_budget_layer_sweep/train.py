"""Train the preregistered four-layer, two-budget SAE sweep for issue 426."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from marin_dna.model.sae import M51_HIDDEN_SIZE, M51_NUM_BLOCKS, load_frozen_m51
from sae_lens.config import LoggingConfig
from sae_lens.constants import SPARSITY_FILENAME
from sae_lens.evals import EvalConfig, run_evals
from sae_lens.load_model import HookedProxyLM
from sae_lens.multi_sae_training_runner import (
    MultiSAETrainingRunner,
    MultiSAETrainingRunnerConfig,
)
from sae_lens.saes.batchtopk_sae import (
    BatchTopKTrainingSAE,
    BatchTopKTrainingSAEConfig,
)
from sae_lens.saes.sae import SAE, SAEMetadata
from sae_lens.training.activation_scaler import ActivationScaler
from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.sae_trainer import SAETrainer
from safetensors.torch import load_file

from data import (
    CONTEXT_TOKENS,
    N_STREAMS,
    ORIENTATIONS,
    SOURCES,
    WINDOW_BP,
    build_balanced_dataset,
    load_pinned_tokenizer,
    provenance_manifest,
)

ISSUE = 426
SEED = 288
MODEL_ID = "marin-dna/marin-dna-exp135-m5.1"
MODEL_REVISION = "c0676b2012b8b9c526deb26ff517f6b92b6d375d"
MODEL_STEP = 59_158
SAELENS_REVISION = "8be14080485952f729ed58d674bcddf9778e0aa4"
MARIN_DNA_REVISION = "c4c1c86bbfc0bd58ff76dda3bac1d2acea856a33"
HUMAN_FASTA_URI = (
    "s3://oa-bolinas/data/genomes/homo_sapiens/GRCh38/ensembl-release-115/"
    "Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz"
)

BLOCK_INDICES = (3, 9, 15, 18)
REPORTED_BLOCKS = tuple(index + 1 for index in BLOCK_INDICES)
ARM_NAMES = tuple(f"block{block:02d}" for block in REPORTED_BLOCKS)
HOOK_NAMES = tuple(f"model.layers.{index}" for index in BLOCK_INDICES)
ARM_TO_BLOCK = dict(zip(ARM_NAMES, BLOCK_INDICES, strict=True))
ARM_TO_HOOK = dict(zip(ARM_NAMES, HOOK_NAMES, strict=True))

D_SAE = 15_360
K = 64
TRAIN_BATCH_TOKENS = N_STREAMS * WINDOW_BP
BUFFER_CONTEXT_BATCHES = 10
# MultiSAETrainer caches all hooks for this many batches simultaneously. At
# four hooks, 1,000 batches would occupy about 78 GB before model/SAE state.
# One hundred exactly balanced batches still estimate each layer's mean norm
# from 255,000 nucleotide activations while keeping peak memory safe on H100.
NORM_ESTIMATE_BATCHES = 100
COMPILE_SMOKE_BATCHES = 1
SHORT_BUDGET = 5_000_550
LONG_BUDGET = 25_000_200
BUDGETS = (SHORT_BUDGET, LONG_BUDGET)
HELDOUT_GAP_PER_STREAM = 1_000
HELDOUT_EVAL_BATCHES = 8

assert M51_HIDDEN_SIZE == 1_920
assert max(BLOCK_INDICES) < M51_NUM_BLOCKS
assert TRAIN_BATCH_TOKENS == 2_550
assert all(budget % TRAIN_BATCH_TOKENS == 0 for budget in BUDGETS)
assert SHORT_BUDGET // TRAIN_BATCH_TOKENS == 1_961
assert LONG_BUDGET // TRAIN_BATCH_TOKENS == 9_804


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def assert_commit(value: str) -> None:
    assert len(value) == 40
    assert all(character in "0123456789abcdef" for character in value)


def arm_label(block_index: int, budget: int) -> str:
    assert block_index in BLOCK_INDICES and budget in BUDGETS
    return f"block{block_index + 1:02d}-{budget // 1_000_000}m"


def training_windows_per_stream(budget: int) -> int:
    assert budget in BUDGETS
    windows = budget // (N_STREAMS * WINDOW_BP)
    assert windows * N_STREAMS * WINDOW_BP == budget
    return windows


def checkpoint_thresholds(total: int, count: int) -> list[int]:
    """Mirror SAELens threshold creation for a testable exact-checkpoint contract."""

    import math

    assert total > 0 and count > 0
    return list(range(0, total, math.ceil(total / (count + 1))))[1:]


def first_step_after(threshold: int, batch: int) -> int:
    """Return the first batch boundary strictly above a SAELens threshold."""

    assert threshold >= 0 and batch > 0
    return (threshold // batch + 1) * batch


def make_sae_config(block_index: int) -> BatchTopKTrainingSAEConfig:
    assert block_index in BLOCK_INDICES
    return BatchTopKTrainingSAEConfig(
        d_in=M51_HIDDEN_SIZE,
        d_sae=D_SAE,
        dtype="float32",
        device="cuda",
        normalize_activations="expected_average_only_in",
        metadata=SAEMetadata(
            issue=ISSUE,
            model_name=MODEL_ID,
            model_revision=MODEL_REVISION,
            model_step=MODEL_STEP,
            hook_name=f"model.layers.{block_index}",
            block_index=block_index,
            report_block=block_index + 1,
            dataset_path="exp426-pinned-five-way-fwd-rc-stream",
            seed=SEED,
            trajectory_training_tokens=LONG_BUDGET,
        ),
        k=float(K),
        aux_loss_coefficient=1.0,
        rescale_acts_by_decoder_norm=True,
        topk_threshold_lr=0.01,
        decoder_init_norm=0.1,
    )


def make_runner_config(
    *,
    checkpoint_path: Path,
    run_name: str,
    bos_token_id: int,
    log_to_wandb: bool,
    compile_llm: bool,
) -> MultiSAETrainingRunnerConfig:
    saes = {
        name: make_sae_config(block_index) for name, block_index in ARM_TO_BLOCK.items()
    }
    logger = LoggingConfig(
        log_to_wandb=log_to_wandb,
        log_activations_store_to_wandb=False,
        log_optimizer_state_to_wandb=False,
        log_weights_to_wandb=False,
        wandb_project="marin",
        run_name=run_name,
        wandb_log_frequency=10,
        eval_every_n_wandb_logs=100_000,
    )
    cfg = MultiSAETrainingRunnerConfig(
        saes=saes,
        hook_names=ARM_TO_HOOK,
        model_name=MODEL_ID,
        model_class_name="HookedProxyLM",
        dataset_path="exp426-pinned-five-way-fwd-rc-stream",
        streaming=True,
        context_size=CONTEXT_TOKENS,
        n_batches_in_buffer=BUFFER_CONTEXT_BATCHES,
        training_tokens=LONG_BUDGET,
        store_batch_size_prompts=N_STREAMS,
        train_batch_size_tokens=TRAIN_BATCH_TOKENS,
        disable_concat_sequences=True,
        sequence_separator_token="bos",
        activations_mixing_fraction=0.0,
        device="cuda",
        llm_device="cuda",
        act_store_device="cuda",
        prefetch_llm_batches=2,
        seed=SEED,
        dtype="float32",
        prepend_bos=True,
        autocast=False,
        autocast_lm=True,
        compile_llm=compile_llm,
        llm_compilation_mode="reduce-overhead" if compile_llm else None,
        lr=3e-4,
        lr_scheduler_name="constant",
        dead_feature_window=1_000,
        feature_sampling_window=2_000,
        n_eval_batches=1,
        logger=logger,
        n_checkpoints=4,
        checkpoint_path=str(checkpoint_path),
        save_final_checkpoint=True,
        output_path=None,
        model_from_pretrained_kwargs={"revision": MODEL_REVISION},
        exclude_special_tokens=[bos_token_id],
        n_batches_for_norm_estimate=NORM_ESTIMATE_BATCHES,
    )
    thresholds = checkpoint_thresholds(cfg.training_tokens, cfg.n_checkpoints)
    assert first_step_after(thresholds[0], TRAIN_BATCH_TOKENS) == SHORT_BUDGET
    return cfg


@torch.inference_mode()
def validate_hook_contract(
    model: HookedProxyLM,
    tokenizer: Any,
    dataset: Any,
) -> dict[str, Any]:
    rows = list(dataset.take(N_STREAMS))
    expected = [
        (source.name, orientation) for source in SOURCES for orientation in ORIENTATIONS
    ]
    assert [(row["source"], row["orientation"]) for row in rows] == expected
    for offset in range(0, N_STREAMS, len(ORIENTATIONS)):
        forward, reverse = rows[offset : offset + len(ORIENTATIONS)]
        assert forward["record_id"] == reverse["record_id"]
    tokens = torch.tensor(
        [row["input_ids"] for row in rows], dtype=torch.long, device="cuda"
    )
    assert tokens.shape == (N_STREAMS, CONTEXT_TOKENS)
    assert torch.all(tokens[:, 0] == tokenizer.bos_token_id)
    output, cache = model.run_with_cache(
        tokens,
        names_filter=list(HOOK_NAMES),
        stop_at_layer=max(BLOCK_INDICES) + 1,
    )
    assert output is None
    assert set(cache) == set(HOOK_NAMES)
    shapes: dict[str, list[int]] = {}
    for hook_name in HOOK_NAMES:
        activations = cache[hook_name]
        assert activations.shape == (
            N_STREAMS,
            CONTEXT_TOKENS,
            M51_HIDDEN_SIZE,
        )
        assert torch.isfinite(activations).all()
        shapes[hook_name] = list(activations.shape)
    return {
        "stream_order": [
            {"source": row["source"], "orientation": row["orientation"]} for row in rows
        ],
        "paired_record_ids": True,
        "hook_shapes": shapes,
        "activations_finite": True,
    }


def find_checkpoint(checkpoint_root: Path, budget: int) -> Path:
    name = str(budget) if budget != LONG_BUDGET else f"final_{budget}"
    matches = [path for path in checkpoint_root.rglob(name) if path.is_dir()]
    assert len(matches) == 1, (name, matches)
    checkpoint = matches[0]
    assert set(ARM_NAMES) <= {path.name for path in checkpoint.iterdir()}
    assert (checkpoint / "runner_cfg.json").is_file()
    return checkpoint


def cast_multi_hook_activations(
    activations: dict[str, torch.Tensor], dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    assert activations
    cast = {name: value.to(dtype=dtype) for name, value in activations.items()}
    assert all(value.dtype == dtype for value in cast.values())
    assert all(torch.isfinite(value).all() for value in cast.values())
    return cast


def install_multi_hook_dtype_adapter(store: ActivationsStore) -> None:
    """Honor the configured store dtype in SAELens' pinned multi-hook path."""

    original = store.get_multi_hook_activations

    def casted(*args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        return cast_multi_hook_activations(original(*args, **kwargs), store.dtype)

    store.get_multi_hook_activations = casted  # type: ignore[method-assign]


def run_compile_smoke(
    runner: MultiSAETrainingRunner,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Exercise the shared-forward path before normalization/training."""

    assert COMPILE_SMOKE_BATCHES == 1
    started = time.monotonic()
    before = runner.activations_store.n_dataset_processed
    # Bypass the disposable mixing-buffer generator: asking that generator for
    # one item would read two 2,550-token batches to clear its 2,560-token
    # threshold, then silently discard the second batch when the generator is
    # dropped. This pinned low-level store seam consumes exactly one balanced
    # ten-window group while exercising the same shared-forward model path.
    batch = runner.activations_store._get_filtered_multi_hook_llm_batch()
    counter_delta = runner.activations_store.n_dataset_processed - before
    # ActivationsStore increments this counter only when its generator resumes
    # after a yield. The final yielded sequence is therefore consumed but not
    # yet reflected in the counter until the next request.
    assert counter_delta == N_STREAMS - 1
    assert set(batch) == set(HOOK_NAMES)
    for hook_name, activations in batch.items():
        assert hook_name in HOOK_NAMES
        assert activations.shape == (TRAIN_BATCH_TOKENS, M51_HIDDEN_SIZE)
        assert activations.dtype == torch.float32
        assert torch.isfinite(activations).all()
    torch.cuda.synchronize()
    metadata = {
        "batches": COMPILE_SMOKE_BATCHES,
        "logical_sequences_consumed": N_STREAMS,
        "activation_store_counter_delta": counter_delta,
        "activation_store_counter_lags_last_yield": True,
        "nucleotide_activations_per_layer": TRAIN_BATCH_TOKENS,
        "hook_shapes": {name: list(value.shape) for name, value in batch.items()},
        "elapsed_seconds": time.monotonic() - started,
        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    return metadata, batch


def run_optimizer_smoke(
    runner: MultiSAETrainingRunner, batch: dict[str, torch.Tensor]
) -> dict[str, Any]:
    """Run one real optimizer step on an isolated SAE clone."""

    started = time.monotonic()
    name = ARM_NAMES[0]
    hook_name = ARM_TO_HOOK[name]
    sae = copy.deepcopy(runner.saes[name])
    trainer = SAETrainer(
        cfg=runner.cfg.to_sae_trainer_config(),
        sae=sae,
        data_provider=iter(()),
        evaluator=None,
        save_checkpoint_fn=None,
    )
    output = trainer.step(batch[hook_name])
    torch.cuda.synchronize()
    loss = float(output.loss.detach())
    assert torch.isfinite(output.loss).all()
    assert trainer.n_training_samples == TRAIN_BATCH_TOKENS
    del output, trainer, sae
    torch.cuda.empty_cache()
    return {
        "arm": name,
        "input_dtype": str(batch[hook_name].dtype),
        "samples": TRAIN_BATCH_TOKENS,
        "loss": loss,
        "elapsed_seconds": time.monotonic() - started,
    }


def _checkpoint_sparsity(path: Path) -> tuple[float, int]:
    values = load_file(path)["sparsity"].double()
    assert values.shape == (D_SAE,) and torch.isfinite(values).all()
    densities = torch.pow(10.0, values)
    return float(densities.sum()), int((densities <= 1e-9).sum())


def export_budget(
    *,
    checkpoint: Path,
    budget: int,
    output_root: Path,
    multi_runner_config: dict[str, Any],
    expected_scaling_factors: dict[int, float] | None = None,
) -> dict[str, Any]:
    assert budget in BUDGETS
    outputs: dict[str, Any] = {}
    for name, block_index in ARM_TO_BLOCK.items():
        checkpoint_sae = checkpoint / name
        output = output_root / arm_label(block_index, budget)
        assert not output.exists()
        training_sae = BatchTopKTrainingSAE.load_from_disk(
            checkpoint_sae, device="cuda", dtype="float32"
        )
        scaler = ActivationScaler()
        scaler.load(checkpoint_sae / "activation_scaler.json")
        checkpoint_scaling_factor = scaler.scaling_factor
        if checkpoint_scaling_factor is not None:
            assert budget == SHORT_BUDGET and checkpoint_scaling_factor > 0
            assert expected_scaling_factors is None
            training_sae.fold_activation_norm_scaling_factor(checkpoint_scaling_factor)
            recorded_scaling_factor = checkpoint_scaling_factor
        else:
            assert budget == LONG_BUDGET
            assert expected_scaling_factors is not None
            recorded_scaling_factor = expected_scaling_factors[block_index]
            assert recorded_scaling_factor > 0
        assert training_sae.cfg.normalize_activations == "none"
        l0, dead = _checkpoint_sparsity(checkpoint_sae / SPARSITY_FILENAME)
        training_sae.cfg.metadata.training_tokens = budget
        training_sae.cfg.metadata.l0 = l0
        training_sae.cfg.metadata.num_dead_features = dead
        training_sae.cfg.metadata.activation_norm_scaling_factor = (
            recorded_scaling_factor
        )
        weights_path, cfg_path = training_sae.save_inference_model(output)
        shutil.copy2(checkpoint_sae / SPARSITY_FILENAME, output / SPARSITY_FILENAME)
        inference_cfg = json.loads(cfg_path.read_text())
        runner_cfg = {
            "model_name": MODEL_ID,
            "model_class_name": "HookedProxyLM",
            "model_from_pretrained_kwargs": {"revision": MODEL_REVISION},
            "dataset_path": "exp426-pinned-five-way-fwd-rc-stream",
            "seed": SEED,
            "training_tokens": budget,
            "train_batch_size_tokens": TRAIN_BATCH_TOKENS,
            "n_batches_for_norm_estimate": NORM_ESTIMATE_BATCHES,
            "sae": inference_cfg,
            "multi_sae_runner": multi_runner_config,
        }
        write_json(output / "runner_cfg.json", runner_cfg)
        outputs[arm_label(block_index, budget)] = {
            "reported_block": block_index + 1,
            "implementation_block_index": block_index,
            "hook_name": ARM_TO_HOOK[name],
            "training_tokens": budget,
            "training_windows_per_stream": training_windows_per_stream(budget),
            "activation_norm_scaling_factor": recorded_scaling_factor,
            "checkpoint_scaling_factor_already_folded": checkpoint_scaling_factor
            is None,
            "source_checkpoint_name": checkpoint.name,
            "temporary_checkpoint_removed_after_verification": True,
            "files": {
                path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in (
                    weights_path,
                    cfg_path,
                    output / SPARSITY_FILENAME,
                    output / "runner_cfg.json",
                )
            },
        }
        del training_sae
    torch.cuda.empty_cache()
    return outputs


def _heldout_store(
    *,
    model: HookedProxyLM,
    tokenizer: Any,
) -> ActivationsStore:
    skip = (
        COMPILE_SMOKE_BATCHES
        + NORM_ESTIMATE_BATCHES
        + training_windows_per_stream(LONG_BUDGET)
        + HELDOUT_GAP_PER_STREAM
    )
    return ActivationsStore.from_config_multi_hook(
        model=model,
        dataset=build_balanced_dataset(tokenizer, skip_per_stream=skip),
        hook_names=list(HOOK_NAMES),
        hook_d_ins={hook: M51_HIDDEN_SIZE for hook in HOOK_NAMES},
        hook_head_indices={hook: None for hook in HOOK_NAMES},
        streaming=True,
        context_size=CONTEXT_TOKENS,
        n_batches_in_buffer=BUFFER_CONTEXT_BATCHES,
        total_training_tokens=HELDOUT_EVAL_BATCHES * TRAIN_BATCH_TOKENS,
        store_batch_size_prompts=N_STREAMS,
        train_batch_size_tokens=TRAIN_BATCH_TOKENS,
        prepend_bos=True,
        normalize_activations="none",
        device=torch.device("cuda"),
        dtype="float32",
        model_kwargs=None,
        autocast_lm=True,
        dataset_trust_remote_code=False,
        seqpos_slice=(None,),
        exclude_special_tokens=torch.tensor(
            [tokenizer.bos_token_id], dtype=torch.long, device="cuda"
        ),
        disable_concat_sequences=True,
        sequence_separator_token="bos",
        activations_mixing_fraction=0.0,
        use_chat_formatting=False,
    )


@torch.inference_mode()
def evaluate_exports(
    *,
    models_root: Path,
    model: HookedProxyLM,
    tokenizer: Any,
) -> dict[str, Any]:
    saes = {
        arm_label(block_index, budget): SAE.load_from_disk(
            models_root / arm_label(block_index, budget),
            device="cuda",
            dtype="float32",
        ).eval()
        for block_index in BLOCK_INDICES
        for budget in BUDGETS
    }
    stats = {
        name: {
            "samples": 0,
            "sse": 0.0,
            "raw_sum": torch.zeros(M51_HIDDEN_SIZE, dtype=torch.float64),
            "raw_square_sum": 0.0,
            "cosine_sum": 0.0,
            "active": torch.zeros(D_SAE, dtype=torch.int64),
            "l0_sum": 0.0,
        }
        for name in saes
    }
    provider = _heldout_store(
        model=model, tokenizer=tokenizer
    ).get_multi_hook_data_loader()
    for _ in range(HELDOUT_EVAL_BATCHES):
        activations = next(provider)
        assert set(activations) == set(HOOK_NAMES)
        for block_index in BLOCK_INDICES:
            raw = activations[f"model.layers.{block_index}"].float()
            assert raw.shape == (TRAIN_BATCH_TOKENS, M51_HIDDEN_SIZE)
            assert torch.isfinite(raw).all()
            for budget in BUDGETS:
                name = arm_label(block_index, budget)
                features = saes[name].encode(raw)
                reconstruction = saes[name].decode(features)
                assert features.shape == (TRAIN_BATCH_TOKENS, D_SAE)
                assert torch.isfinite(features).all() and torch.all(features >= 0)
                residual = raw - reconstruction
                cosine = torch.nn.functional.cosine_similarity(
                    raw, reconstruction, dim=-1
                )
                assert torch.isfinite(cosine).all()
                entry = stats[name]
                entry["samples"] += raw.shape[0]
                entry["sse"] += float(torch.square(residual).sum().item())
                entry["raw_sum"] += raw.sum(dim=0).double().cpu()
                entry["raw_square_sum"] += float(torch.square(raw).sum().item())
                entry["cosine_sum"] += float(cosine.sum().item())
                support = features > 0
                entry["active"] += support.sum(dim=0).cpu()
                entry["l0_sum"] += float(support.sum().item())
    output: dict[str, Any] = {}
    for name, entry in stats.items():
        samples = int(entry["samples"])
        assert samples == HELDOUT_EVAL_BATCHES * TRAIN_BATCH_TOKENS
        total_values = samples * M51_HIDDEN_SIZE
        centered = (
            entry["raw_square_sum"]
            - float(torch.square(entry["raw_sum"]).sum().item()) / samples
        )
        assert centered > 0
        active = entry["active"]
        output[name] = {
            "samples": samples,
            "mse": entry["sse"] / total_values,
            "explained_variance": 1.0 - entry["sse"] / centered,
            "cosine_similarity": entry["cosine_sum"] / samples,
            "l0": entry["l0_sum"] / samples,
            "heldout_inactive_features": int((active == 0).sum().item()),
            "heldout_inactive_fraction": float((active == 0).double().mean()),
            "top_1pct_activation_share": float(
                active.topk(max(1, round(0.01 * D_SAE))).values.sum().item()
                / active.sum().item()
            ),
        }
        assert all(
            torch.isfinite(torch.tensor(value))
            for value in output[name].values()
            if isinstance(value, float)
        )
    for block_index in BLOCK_INDICES:
        for budget in BUDGETS:
            name = arm_label(block_index, budget)
            evaluation_store = _heldout_store(model=model, tokenizer=tokenizer)
            metrics, _ = run_evals(
                sae=saes[name],
                activation_store=evaluation_store,
                model=model,
                activation_scaler=ActivationScaler(),
                eval_config=EvalConfig(
                    batch_size_prompts=N_STREAMS,
                    n_eval_reconstruction_batches=2,
                    compute_kl=True,
                    compute_ce_loss=True,
                ),
                exclude_special_tokens=[tokenizer.bos_token_id],
                verbose=True,
            )
            performance = metrics["model_performance_preservation"]
            behavior = metrics["model_behavior_preservation"]
            output[name]["ce_loss_without_sae"] = performance["ce_loss_without_sae"]
            output[name]["ce_loss_with_sae"] = performance["ce_loss_with_sae"]
            output[name]["ce_loss_degradation"] = (
                performance["ce_loss_with_sae"] - performance["ce_loss_without_sae"]
            )
            output[name]["ce_loss_score"] = performance["ce_loss_score"]
            output[name]["kl_div_with_sae"] = behavior["kl_div_with_sae"]
            output[name]["kl_div_score"] = behavior["kl_div_score"]
    return output


def dry_run_manifest(run_id: str, *, compile_llm: bool) -> dict[str, Any]:
    thresholds = checkpoint_thresholds(LONG_BUDGET, 4)
    return {
        "run_id": run_id,
        "issue": ISSUE,
        "fixed_config": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_step": MODEL_STEP,
            "reported_blocks": list(REPORTED_BLOCKS),
            "implementation_block_indices": list(BLOCK_INDICES),
            "hook_names": list(HOOK_NAMES),
            "budgets": list(BUDGETS),
            "optimizer_steps": [budget // TRAIN_BATCH_TOKENS for budget in BUDGETS],
            "d_in": M51_HIDDEN_SIZE,
            "d_sae": D_SAE,
            "k": K,
            "seed": SEED,
            "saelens_revision": SAELENS_REVISION,
            "marin_dna_revision": MARIN_DNA_REVISION,
            "train_batch_tokens": TRAIN_BATCH_TOKENS,
            "store_batch_size_prompts": N_STREAMS,
            "n_batches_in_buffer": BUFFER_CONTEXT_BATCHES,
            "n_batches_for_norm_estimate": NORM_ESTIMATE_BATCHES,
            "compile_smoke_batches": COMPILE_SMOKE_BATCHES,
            "normalization_observations_per_layer": NORM_ESTIMATE_BATCHES
            * TRAIN_BATCH_TOKENS,
            "normalize_activations": "expected_average_only_in",
            "activations_mixing_fraction": 0.0,
            "lm_dtype": "bfloat16",
            "sae_dtype": "float32",
            "autocast_lm": True,
            "compile_llm": compile_llm,
            "model_use_cache": False,
            "multi_hook_activation_dtype_adapter": "configured_store_dtype",
            "llm_compilation_mode": "reduce-overhead" if compile_llm else None,
            "prefetch_llm_batches": 2,
            "checkpoint_thresholds": thresholds,
            "first_checkpoint_batch_boundaries": [
                first_step_after(threshold, TRAIN_BATCH_TOKENS)
                for threshold in thresholds
            ],
            "save_final_checkpoint": True,
        },
        "data": provenance_manifest(),
        "heldout": {
            "skip_per_stream": COMPILE_SMOKE_BATCHES
            + NORM_ESTIMATE_BATCHES
            + training_windows_per_stream(LONG_BUDGET)
            + HELDOUT_GAP_PER_STREAM,
            "gap_after_long_training_prefix_per_stream": HELDOUT_GAP_PER_STREAM,
            "eval_batches": HELDOUT_EVAL_BATCHES,
        },
        "interpretation_boundary": {
            "training": "pinned equal five-source stream, exactly balanced FWD/RC",
            "post_training": "fixed #422 chr21 GRCh38 consequence panel under the #424 paired protocol",
            "human_fasta_uri": HUMAN_FASTA_URI,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_id = args.run_id
    compile_llm = not args.no_compile
    dry = dry_run_manifest(run_id, compile_llm=compile_llm)
    if args.dry_run:
        return dry
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert_commit(experiment_commit)
    if args.wandb:
        assert os.environ.get("WANDB_API_KEY")
    output_dir = args.output_root / run_id
    assert not output_dir.exists()
    output_dir.mkdir(parents=True)
    checkpoints = output_dir / "checkpoints"
    models_root = output_dir / "models"

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    started = time.monotonic()
    snapshot = Path(snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION))
    assert MODEL_REVISION in snapshot.parts
    frozen = load_frozen_m51(snapshot, device="cuda", dtype=torch.bfloat16)
    # Full-context activation extraction never consumes a generation cache.
    # Disabling it also removes dynamic cache initialization guards that make
    # torch.compile repeatedly specialize Qwen's forward graph.
    frozen.model.config.use_cache = False
    assert frozen.model.config.use_cache is False
    tokenizer = load_pinned_tokenizer()
    assert tokenizer.get_vocab() == frozen.tokenizer.get_vocab()
    model = HookedProxyLM(frozen.model, tokenizer, hook_names=list(HOOK_NAMES))
    hook_validation = validate_hook_contract(
        model, tokenizer, build_balanced_dataset(tokenizer)
    )
    cfg = make_runner_config(
        checkpoint_path=checkpoints,
        run_name=run_id,
        bos_token_id=tokenizer.bos_token_id,
        log_to_wandb=args.wandb,
        compile_llm=compile_llm,
    )
    runner = MultiSAETrainingRunner(
        cfg,
        override_dataset=build_balanced_dataset(tokenizer),
        override_model=model,
    )
    install_multi_hook_dtype_adapter(runner.activations_store)
    torch.cuda.reset_peak_memory_stats()
    compile_smoke, smoke_batch = run_compile_smoke(runner)
    optimizer_smoke = run_optimizer_smoke(runner, smoke_batch)
    del smoke_batch
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    runner_started = time.monotonic()
    runner.run()
    runner_wall_seconds = time.monotonic() - runner_started
    training_peak_memory = {
        "allocated_bytes": torch.cuda.max_memory_allocated(),
        "reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    short_checkpoint = find_checkpoint(checkpoints, SHORT_BUDGET)
    long_checkpoint = find_checkpoint(checkpoints, LONG_BUDGET)
    multi_runner_config = cfg.to_dict()
    short_exports = export_budget(
        checkpoint=short_checkpoint,
        budget=SHORT_BUDGET,
        output_root=models_root,
        multi_runner_config=multi_runner_config,
    )
    scaling_factors = {
        block_index: short_exports[arm_label(block_index, SHORT_BUDGET)][
            "activation_norm_scaling_factor"
        ]
        for block_index in BLOCK_INDICES
    }
    long_exports = export_budget(
        checkpoint=long_checkpoint,
        budget=LONG_BUDGET,
        output_root=models_root,
        multi_runner_config=multi_runner_config,
        expected_scaling_factors=scaling_factors,
    )
    exports = {**short_exports, **long_exports}
    assert len(exports) == len(BLOCK_INDICES) * len(BUDGETS)
    torch.cuda.reset_peak_memory_stats()
    health = evaluate_exports(
        models_root=models_root, model=runner.model, tokenizer=tokenizer
    )
    export_evaluation_peak_memory = {
        "allocated_bytes": torch.cuda.max_memory_allocated(),
        "reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    shutil.rmtree(checkpoints)
    assert not checkpoints.exists()
    elapsed = time.monotonic() - started
    manifest = {
        **dry,
        "created_at": datetime.now(UTC).isoformat(),
        "experiment_commit": experiment_commit,
        "runtime": {
            "wall_seconds": elapsed,
            "normalization_and_training_runner_wall_seconds": runner_wall_seconds,
            "trajectory_nucleotide_activations_per_second_per_layer": LONG_BUDGET
            / runner_wall_seconds,
        },
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "saelens": importlib.metadata.version("sae-lens"),
        "hook_validation": hook_validation,
        "optimizer_smoke": optimizer_smoke,
        "compile_smoke": compile_smoke,
        "training_peak_memory": training_peak_memory,
        "export_evaluation_peak_memory": export_evaluation_peak_memory,
        "temporary_optimizer_checkpoints_removed": True,
        "exports": exports,
        "health": health,
    }
    write_json(output_dir / "results.json", manifest)
    artifacts = {
        str(path.relative_to(output_dir)): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(models_root.rglob("*"))
        if path.is_file()
    }
    artifacts["results.json"] = {
        "bytes": (output_dir / "results.json").stat().st_size,
        "sha256": sha256_file(output_dir / "results.json"),
    }
    output_manifest = {**manifest, "artifacts": artifacts}
    write_json(output_dir / "manifest.json", output_manifest)
    return output_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="dna-exp426-layer-budget-seed288")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
