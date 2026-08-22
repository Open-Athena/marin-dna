"""Frozen-source diagnostic for BICO's reflected future-position RoPE."""

from __future__ import annotations

import gc
import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

from exp479_mntp.attention_anneal_diagnostic import _endpoint_delta
from exp479_mntp.config import BUDGET_USD, EXPERIMENT_TAGS, MODEL_REVISION, WANDB_PROJECT
from exp479_mntp.data import plan_sha256
from exp479_mntp.modeling import ModelBundle, load_model_bundle
from exp479_mntp.paired_nucleotide_gate import (
    EXPECTED_VALIDATION_PLAN_SHA256,
    evaluate_readout,
    information_gate,
    paired_comparison,
    summarize_readouts,
)
from exp479_mntp.publishing import assert_budget_reserve, write_cost_estimate

BICO_ATTENTION_RUN_NAME = "dna-exp479-source-bico-attention-diagnostic"
BICO_ATTENTION_ARTIFACT = BICO_ATTENTION_RUN_NAME
BICO_REFERENCE = "https://aclanthology.org/2024.emnlp-main.754/"
MAXIMUM_INSTANCE_HOURS = 1.0
ATTENTION_PARITY_CE_TOLERANCE = 2e-3


def _qwen_attention_modules(model: nn.Module) -> list[nn.Module]:
    """Return Qwen3 attention modules from a bare or PEFT-wrapped model."""

    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        base_model = getattr(model, "base_model", None)
        wrapped_model = getattr(base_model, "model", None)
        layers = getattr(getattr(wrapped_model, "model", None), "layers", None)
    if layers is None:
        raise TypeError("BICO attention requires a bare or PEFT-wrapped Qwen3 model")
    return [layer.self_attn for layer in layers]


def excluded_selected_key_mask(
    attention_mask: torch.Tensor,
    output_positions: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build full attention while excluding each shifted target token as a key."""

    if attention_mask.ndim != 2:
        raise ValueError("token attention mask must have shape [batch, sequence]")
    if output_positions.ndim != 1 or output_positions.shape[0] != attention_mask.shape[0]:
        raise ValueError("output positions must contain one position per sequence")
    if not dtype.is_floating_point:
        raise ValueError("additive attention mask requires a floating dtype")
    batch_size, sequence_length = attention_mask.shape
    target_input_positions = (
        output_positions.to(
            device=attention_mask.device,
            dtype=torch.long,
        )
        + 1
    )
    if torch.any(target_input_positions < 0) or torch.any(
        target_input_positions >= sequence_length
    ):
        raise ValueError("shifted target input position is outside the token sequence")

    key_allowed = attention_mask.to(dtype=torch.bool).clone()
    rows = torch.arange(batch_size, device=attention_mask.device)
    key_allowed[rows, target_input_positions] = False
    allowed = key_allowed[:, None, None, :].expand(
        batch_size,
        1,
        sequence_length,
        sequence_length,
    )
    additive = torch.zeros(allowed.shape, dtype=dtype, device=attention_mask.device)
    return additive.masked_fill(~allowed, torch.finfo(dtype).min)


def bico_attention_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Any | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply non-positive relative RoPE distances on both sides of each query."""

    del kwargs
    if past_key_values is not None:
        raise ValueError("BICO diagnostic does not support cached key/value states")
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    query = module.q_norm(module.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key = module.k_norm(module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    standard_query, standard_key = apply_rotary_pos_emb(query, key, cos, sin)
    reflected_query, reflected_key = apply_rotary_pos_emb(query, key, cos, -sin)
    standard_key = repeat_kv(standard_key, module.num_key_value_groups)
    reflected_key = repeat_kv(reflected_key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)

    standard_logits = torch.matmul(standard_query, standard_key.transpose(2, 3))
    reflected_logits = torch.matmul(reflected_query, reflected_key.transpose(2, 3))
    query_positions = torch.arange(
        standard_logits.shape[-2],
        device=standard_logits.device,
    )
    key_positions = torch.arange(
        standard_logits.shape[-1],
        device=standard_logits.device,
    )
    future = key_positions[None, :] > query_positions[:, None]
    attention_logits = (
        torch.where(
            future[None, None, :, :],
            reflected_logits,
            standard_logits,
        )
        * module.scaling
    )
    if attention_mask is not None:
        attention_logits = attention_logits + attention_mask
    attention_weights = nn.functional.softmax(
        attention_logits,
        dim=-1,
        dtype=torch.float32,
    ).to(query.dtype)
    attention_weights = nn.functional.dropout(
        attention_weights,
        p=0.0 if not module.training else module.attention_dropout,
        training=module.training,
    )
    attention_output = torch.matmul(attention_weights, value)
    attention_output = attention_output.transpose(1, 2).contiguous()
    attention_output = module.o_proj(attention_output.reshape(*input_shape, -1).contiguous())
    return attention_output, attention_weights


@contextmanager
def reflected_future_rope(model: nn.Module) -> Iterator[None]:
    """Temporarily replace every Qwen3 self-attention layer with BICO attention."""

    modules = _qwen_attention_modules(model)
    originals = [module.forward for module in modules]
    try:
        for module in modules:
            module.forward = MethodType(bico_attention_forward, module)
        yield
    finally:
        for module, original in zip(modules, originals, strict=True):
            module.forward = original


def install_reflected_future_rope(model: nn.Module) -> None:
    """Install BICO attention for an entire training and backward lifetime."""

    for module in _qwen_attention_modules(model):
        module.forward = MethodType(bico_attention_forward, module)


def _plot_bico(
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot absolute metrics and paired deltas for the mechanism controls."""

    display = {
        "source_causal_standard": "Causal + UNK",
        "source_full_standard_unk": "Full standard + UNK",
        "source_full_standard_pad_attended": "Full standard + PAD",
        "source_full_standard_pad_excluded": "Full standard + PAD excluded",
        "source_full_bico_unk": "Full BICO + UNK",
        "source_full_bico_pad_attended": "Full BICO + PAD",
        "source_full_bico_pad_excluded": "Full BICO + PAD excluded",
    }
    colors = {
        "source_causal_standard": "#4C78A8",
        "source_full_standard_unk": "#E45756",
        "source_full_standard_pad_attended": "#F58518",
        "source_full_standard_pad_excluded": "#FFBF79",
        "source_full_bico_unk": "#59A14F",
        "source_full_bico_pad_attended": "#76B7B2",
        "source_full_bico_pad_excluded": "#0B7A75",
    }
    order = list(display)
    indexed = summary.set_index("readout").loc[order]
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    x = np.arange(len(order))
    labels = [display[name] for name in order]
    bar_colors = [colors[name] for name in order]
    axes[0, 0].bar(x, indexed["nucleotide_ce"], color=bar_colors)
    axes[0, 1].bar(x, indexed["nucleotide_accuracy"], color=bar_colors)
    axes[0, 0].set_ylabel("Four-way nucleotide CE")
    axes[0, 1].set_ylabel("Four-way nucleotide accuracy")
    axes[0, 1].set_ylim(0, 1)
    for axis in axes[0]:
        axis.set_xticks(x, labels, rotation=23, ha="right")
        axis.grid(axis="y", alpha=0.25)

    causal_candidates = comparisons[comparisons["contrast"].str.endswith("_vs_causal")]
    candidate_order = [name for name in order[1:] if name in set(causal_candidates["candidate"])]
    ordered = causal_candidates.set_index("candidate").loc[candidate_order].reset_index()
    delta_x = np.arange(len(ordered))
    delta_labels = [display[name] for name in ordered["candidate"]]
    ce = ordered["nucleotide_ce_delta"].to_numpy(dtype=float)
    ce_low = ordered["nucleotide_ce_delta_ci95_low"].to_numpy(dtype=float)
    ce_high = ordered["nucleotide_ce_delta_ci95_high"].to_numpy(dtype=float)
    accuracy = ordered["nucleotide_accuracy_delta"].to_numpy(dtype=float)
    accuracy_low = ordered["nucleotide_accuracy_delta_ci95_low"].to_numpy(dtype=float)
    accuracy_high = ordered["nucleotide_accuracy_delta_ci95_high"].to_numpy(dtype=float)
    axes[1, 0].errorbar(
        delta_x,
        ce,
        yerr=np.maximum(0, np.vstack((ce - ce_low, ce_high - ce))),
        fmt="o",
        color="#333333",
        capsize=4,
    )
    axes[1, 1].errorbar(
        delta_x,
        accuracy,
        yerr=np.maximum(0, np.vstack((accuracy - accuracy_low, accuracy_high - accuracy))),
        fmt="o",
        color="#333333",
        capsize=4,
    )
    axes[1, 0].axhline(0, color="black", linewidth=1)
    axes[1, 1].axhline(0, color="black", linewidth=1)
    axes[1, 0].set_ylabel("Paired CE delta vs causal")
    axes[1, 1].set_ylabel("Paired accuracy delta vs causal")
    for axis in axes[1]:
        axis.set_xticks(delta_x, delta_labels, rotation=23, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Frozen Qwen3: reflected future RoPE and masked-key exclusion")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_bico_attention_diagnostic(
    *,
    artifact_dir: Path,
    output_dir: Path,
    validation_plan: Path,
    batch_size: int,
    n_bootstrap: int,
) -> None:
    """Compare standard and BICO attention on identical frozen-source targets."""

    if not torch.cuda.is_available():
        raise RuntimeError("BICO attention diagnostic requires one CUDA GPU")
    if batch_size <= 0 or n_bootstrap <= 0:
        raise ValueError("batch size and bootstrap count must be positive")
    validation_hash = plan_sha256(validation_plan)
    if validation_hash != EXPECTED_VALIDATION_PLAN_SHA256:
        raise RuntimeError("BICO diagnostic validation plan differs from the paired gate")
    assert_budget_reserve()
    prior_cost = float(os.getenv("EXP479_PRIOR_COST_USD", "0"))
    price = float(os.getenv("EXP479_INSTANCE_PRICE_PER_HOUR_USD", "1.29"))
    if prior_cost + MAXIMUM_INSTANCE_HOURS * price >= BUDGET_USD:
        raise RuntimeError("BICO diagnostic projection reaches the issue budget cap")

    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=WANDB_PROJECT,
        group="dna-exp479-paired-nucleotide-information",
        name=BICO_ATTENTION_RUN_NAME,
        tags=[*EXPERIMENT_TAGS, "no-training", "bico", "paired-targets"],
        config={
            "base_revision": MODEL_REVISION,
            "validation_plan_sha256": validation_hash,
            "validation_targets": 640,
            "parameter_updates": 0,
            "reference": BICO_REFERENCE,
            "attention_backend": "manual eager Qwen3 attention",
            "batch_size": batch_size,
            "n_bootstrap": n_bootstrap,
        },
    )
    if run is None:
        raise RuntimeError("W&B did not create the BICO attention diagnostic")

    try:
        loaded = load_model_bundle(
            initialization="transferred",
            add_mask=False,
            attention_implementation="eager",
            dtype=torch.bfloat16,
        )
        unk_token_id = loaded.tokenizer.unk_token_id
        pad_token_id = loaded.tokenizer.pad_token_id
        if unk_token_id is None or pad_token_id is None:
            raise RuntimeError("source tokenizer must define [UNK] and [PAD]")
        source = ModelBundle(
            model=loaded.model,
            tokenizer=loaded.tokenizer,
            canonical_token_ids=loaded.canonical_token_ids,
            mask_token_id=int(unk_token_id),
            input_output_tied=loaded.input_output_tied,
        )
        source.model.to(device="cuda", dtype=torch.bfloat16).eval()

        def exclude_selected_key(
            token_mask: torch.Tensor,
            sample_ids: torch.Tensor,
            output_positions: torch.Tensor,
        ) -> torch.Tensor:
            del sample_ids
            return excluded_selected_key_mask(
                token_mask,
                output_positions,
                dtype=torch.bfloat16,
            )

        common = {
            "validation_plan": validation_plan,
            "batch_size": batch_size,
        }
        standard_causal = evaluate_readout(
            source,
            **common,
            readout="source_causal_standard",
            attention_mode="causal",
        )
        standard_full_unk = evaluate_readout(
            source,
            **common,
            readout="source_full_standard_unk",
            attention_mode="full",
        )
        standard_full_pad_attended = evaluate_readout(
            source,
            **common,
            readout="source_full_standard_pad_attended",
            attention_mode="full",
            replacement_mask_token_id=int(pad_token_id),
        )
        standard_full_pad_excluded = evaluate_readout(
            source,
            **common,
            readout="source_full_standard_pad_excluded",
            attention_mode="full",
            replacement_mask_token_id=int(pad_token_id),
            attention_mask_transform=exclude_selected_key,
        )
        with reflected_future_rope(source.model):
            bico_causal = evaluate_readout(
                source,
                **common,
                readout="source_causal_bico",
                attention_mode="causal",
            )
            bico_full_unk = evaluate_readout(
                source,
                **common,
                readout="source_full_bico_unk",
                attention_mode="full",
            )
            bico_full_pad_attended = evaluate_readout(
                source,
                **common,
                readout="source_full_bico_pad_attended",
                attention_mode="full",
                replacement_mask_token_id=int(pad_token_id),
            )
            bico_full_pad_excluded = evaluate_readout(
                source,
                **common,
                readout="source_full_bico_pad_excluded",
                attention_mode="full",
                replacement_mask_token_id=int(pad_token_id),
                attention_mask_transform=exclude_selected_key,
            )

        parity = _endpoint_delta(bico_causal, standard_causal)
        parity["ce_tolerance"] = ATTENTION_PARITY_CE_TOLERANCE
        parity["passed"] = bool(
            parity["maximum_absolute_nucleotide_ce_delta"] <= ATTENTION_PARITY_CE_TOLERANCE
            and parity["nucleotide_prediction_mismatches"] == 0
        )
        if not parity["passed"]:
            raise RuntimeError(f"BICO causal attention does not match standard eager: {parity}")

        scores = pd.concat(
            [
                standard_causal,
                bico_causal,
                standard_full_unk,
                standard_full_pad_attended,
                standard_full_pad_excluded,
                bico_full_unk,
                bico_full_pad_attended,
                bico_full_pad_excluded,
            ],
            ignore_index=True,
        )
        identity = scores.groupby(["sample_id", "target_nucleotide_index"]).size()
        if len(identity) != 640 or not (identity == 8).all():
            raise RuntimeError("BICO readouts did not evaluate an identical 640-target panel")
        summary = summarize_readouts(scores)

        comparisons: list[dict[str, object]] = []
        candidates = (
            "source_full_standard_unk",
            "source_full_standard_pad_attended",
            "source_full_standard_pad_excluded",
            "source_full_bico_unk",
            "source_full_bico_pad_attended",
            "source_full_bico_pad_excluded",
        )
        for candidate in candidates:
            comparison = paired_comparison(
                scores,
                candidate=candidate,
                baseline="source_causal_standard",
                n_bootstrap=n_bootstrap,
            )
            comparison["contrast"] = f"{candidate}_vs_causal"
            comparisons.append(comparison)
        for contrast, candidate, baseline in (
            (
                "reflected_rope_with_pad_exclusion",
                "source_full_bico_pad_excluded",
                "source_full_standard_pad_excluded",
            ),
            (
                "standard_pad_key_exclusion",
                "source_full_standard_pad_excluded",
                "source_full_standard_pad_attended",
            ),
            (
                "bico_pad_key_exclusion",
                "source_full_bico_pad_excluded",
                "source_full_bico_pad_attended",
            ),
        ):
            comparison = paired_comparison(
                scores,
                candidate=candidate,
                baseline=baseline,
                n_bootstrap=n_bootstrap,
            )
            comparison["contrast"] = contrast
            comparisons.append(comparison)
        comparison_frame = pd.DataFrame(comparisons)
        primary = next(
            item
            for item in comparisons
            if item["contrast"] == "source_full_bico_pad_excluded_vs_causal"
        )
        mechanism = next(
            item for item in comparisons if item["contrast"] == "reflected_rope_with_pad_exclusion"
        )
        gate = information_gate(primary)
        mechanism_gate = information_gate(mechanism)

        scores_path = output_dir / "bico-attention-scores.csv"
        summary_path = output_dir / "bico-attention-summary.csv"
        comparisons_path = output_dir / "bico-attention-comparisons.csv"
        gate_path = output_dir / "bico-attention-gate.json"
        parity_path = output_dir / "bico-causal-parity.json"
        runtime_path = output_dir / "runtime.json"
        figure_path = output_dir / "figures" / "bico-attention"
        scores.to_csv(scores_path, index=False)
        summary.to_csv(summary_path, index=False)
        comparison_frame.to_csv(comparisons_path, index=False)
        gate_payload = {
            "single_pass_vs_causal": gate,
            "reflected_rope_vs_standard": mechanism_gate,
        }
        gate_path.write_text(json.dumps(gate_payload, indent=2) + "\n", encoding="utf-8")
        parity_path.write_text(json.dumps(parity, indent=2) + "\n", encoding="utf-8")
        _plot_bico(summary, comparison_frame, figure_path)
        runtime_path.write_text(
            json.dumps({"elapsed_seconds": time.time() - started}, indent=2) + "\n",
            encoding="utf-8",
        )
        cost_path = write_cost_estimate(artifact_dir=artifact_dir)
        manifest = {
            "status": "completed",
            "wandb_url": run.get_url(),
            "base_revision": MODEL_REVISION,
            "validation_plan_sha256": validation_hash,
            "paired_target_count": 640,
            "parameter_updates": 0,
            "reference": BICO_REFERENCE,
            "attention_backend": "manual eager Qwen3 attention",
            "future_relative_rope": "reflected so every relative distance is non-positive",
            "primary_mask_token": "[PAD] excluded as an attention key",
            "causal_parity": parity,
            "gates": gate_payload,
            "checkpoint_deletion": "not performed",
            "hugging_face_upload": "not performed",
            "vep_evaluation": "not performed",
            "nucleotide_dependency": "not performed",
            "knowledge_base_update": "not performed",
        }
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        run.log(
            {
                "bico_attention/summary": wandb.Table(dataframe=summary),
                "bico_attention/comparisons": wandb.Table(dataframe=comparison_frame),
                "bico_attention/figure": wandb.Image(str(figure_path.with_suffix(".png"))),
            }
        )
        run.summary["bico_attention/single_pass_gate_passed"] = bool(gate["passed"])
        run.summary["bico_attention/mechanism_supported"] = bool(mechanism_gate["passed"])
        run.summary["bico_attention/causal_parity_passed"] = bool(parity["passed"])
        artifact = wandb.Artifact(BICO_ATTENTION_ARTIFACT, type="evaluation")
        for path in (
            scores_path,
            summary_path,
            comparisons_path,
            gate_path,
            parity_path,
            runtime_path,
            cost_path,
            manifest_path,
            figure_path.with_suffix(".svg"),
        ):
            artifact.add_file(str(path))
        logged = run.log_artifact(artifact, aliases=["zero-training", "bico-rope"])
        logged.wait()
        run.finish(exit_code=0)
        del source, loaded
        gc.collect()
        torch.cuda.empty_cache()
    except BaseException:
        run.finish(exit_code=1)
        raise
