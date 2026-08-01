"""Benchmark one exact BF16 LLR-only configuration for issue #430."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from sklearn.metrics import average_precision_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from marin_dna.model.scoring import (
    compute_variant_llr,
    compute_variant_llr_branch_packed,
    compute_variant_llr_full_pair,
    compute_variant_llr_sequential_branches,
    compute_variant_score_bundle,
)
from marin_dna.pipelines.evals.inference_benchmark import (
    VARIANT_KEY_COLUMNS,
    ExecutionLayout,
    LlrBenchmarkResult,
    PreparedHarnessLlr,
    aggregate_harness_llr,
    benchmark_prepared_llr,
    prepare_harness_llr,
)


MODEL_REVISION = "c0676b2012b8b9c526deb26ff517f6b92b6d375d"
DATASET_ID = "marin-dna/evals_mendelian_traits_harness_255"
DATASET_REVISION = "7b92f047f9a36f90e9ac47886afa2a99264ee35c"
BYTES_PER_GIB = 1024**3
QUANTIZATION_CHOICES = [
    "none",
    "fp8-dynamic",
    "fp8-rowwise",
    "int8-dynamic",
    "int8-weight-only",
    "int4-weight-only",
    "te-bf16",
    "te-fp8-delayed",
]


class _TransformerEngineModelWrapper(nn.Module):
    """Enter Transformer Engine FP8 autocast for every HF model forward."""

    def __init__(self, model: nn.Module, *, enabled: bool, recipe: Any) -> None:
        super().__init__()
        self.model = model
        self.enabled = enabled
        self.recipe = recipe

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        import transformer_engine.pytorch as te

        if hasattr(te, "autocast"):
            context = te.autocast(enabled=self.enabled, recipe=self.recipe)
        elif self.enabled:
            context = te.fp8_autocast(enabled=True, fp8_recipe=self.recipe)
        else:
            context = nullcontext()
        with context:
            return self.model(*args, **kwargs)


def _replace_qwen3_mlps_with_transformer_engine(model: nn.Module, te: Any) -> int:
    """Fuse each Qwen3 post-attention RMSNorm and gated MLP."""
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    assert layers is not None, "expected Qwen3ForCausalLM.model.layers"
    assert len(layers) == model.config.num_hidden_layers

    for layer in layers:
        norm = layer.post_attention_layernorm
        mlp = layer.mlp
        assert isinstance(mlp.gate_proj, nn.Linear)
        assert isinstance(mlp.up_proj, nn.Linear)
        assert isinstance(mlp.down_proj, nn.Linear)
        assert mlp.gate_proj.bias is None
        assert mlp.up_proj.bias is None
        assert mlp.down_proj.bias is None
        assert norm.weight.shape == (mlp.gate_proj.in_features,)
        fused = te.LayerNormMLP(
            hidden_size=mlp.gate_proj.in_features,
            ffn_hidden_size=mlp.gate_proj.out_features,
            eps=norm.variance_epsilon,
            bias=False,
            normalization="RMSNorm",
            activation="swiglu",
            params_dtype=mlp.gate_proj.weight.dtype,
            device=mlp.gate_proj.weight.device,
        )
        expected_fc1 = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], dim=0)
        assert fused.fc1_weight.shape == expected_fc1.shape
        assert fused.fc2_weight.shape == mlp.down_proj.weight.shape
        fused.layer_norm_weight.copy_(norm.weight)
        fused.fc1_weight.copy_(expected_fc1)
        fused.fc2_weight.copy_(mlp.down_proj.weight)
        layer.post_attention_layernorm = nn.Identity()
        layer.mlp = fused

    return len(layers)


def _replace_linears_with_transformer_engine(
    model: nn.Module,
    *,
    fp8_enabled: bool,
    fp8_model_init: bool,
    fused_mlp: bool,
    amax_history_len: int,
    amax_compute_algo: str,
) -> tuple[nn.Module, int, int]:
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe

    assert not fp8_model_init or fp8_enabled
    fp8_recipe = recipe.DelayedScaling(
        margin=0,
        fp8_format=recipe.Format.E4M3,
        amax_history_len=amax_history_len,
        amax_compute_algo=amax_compute_algo,
        reduce_amax=False,
    )
    init_context = (
        te.fp8_model_init(enabled=True, recipe=fp8_recipe)
        if fp8_model_init
        else nullcontext()
    )
    with torch.no_grad(), init_context:
        fused_mlp_count = (
            _replace_qwen3_mlps_with_transformer_engine(model, te) if fused_mlp else 0
        )
        selected = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear) and not name.endswith("lm_head")
        ]
        assert selected, "no non-LM-head linear modules selected for Transformer Engine"
        for name, module in selected:
            replacement = te.Linear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                params_dtype=module.weight.dtype,
                device=module.weight.device,
            )
            replacement.weight.copy_(module.weight)
            if module.bias is not None:
                assert replacement.bias is not None
                replacement.bias.copy_(module.bias)
            model.set_submodule(name, replacement)

    return (
        _TransformerEngineModelWrapper(
            model,
            enabled=fp8_enabled,
            recipe=fp8_recipe,
        ),
        len(selected),
        fused_mlp_count,
    )


def _quantize_model(
    model: nn.Module,
    quantization: str,
    *,
    te_amax_history_len: int,
    te_amax_compute_algo: str,
    te_fp8_model_init: bool,
    te_fused_mlp: bool,
) -> tuple[nn.Module, float, int, int]:
    if quantization == "none":
        return model, 0.0, 0, 0

    if quantization in ("te-bf16", "te-fp8-delayed"):
        start = time.perf_counter()
        (
            model,
            replaced_linear_count,
            fused_mlp_count,
        ) = _replace_linears_with_transformer_engine(
            model,
            fp8_enabled=quantization == "te-fp8-delayed",
            fp8_model_init=te_fp8_model_init,
            fused_mlp=te_fused_mlp,
            amax_history_len=te_amax_history_len,
            amax_compute_algo=te_amax_compute_algo,
        )
        return (
            model,
            time.perf_counter() - start,
            replaced_linear_count,
            fused_mlp_count,
        )

    from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig,
        Int4WeightOnlyConfig,
        Int8DynamicActivationInt8WeightConfig,
        Int8WeightOnlyConfig,
        quantize_,
    )
    from torchao.quantization.granularity import PerRow

    configs = {
        "fp8-dynamic": Float8DynamicActivationFloat8WeightConfig(),
        "fp8-rowwise": Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()),
        "int8-dynamic": Int8DynamicActivationInt8WeightConfig(),
        "int8-weight-only": Int8WeightOnlyConfig(),
        "int4-weight-only": Int4WeightOnlyConfig(group_size=128),
    }
    assert quantization in configs, f"unsupported quantization: {quantization}"
    selected_names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and not name.endswith("lm_head")
    ]
    assert selected_names, "no non-LM-head linear modules selected for quantization"
    start = time.perf_counter()
    quantize_(
        model,
        configs[quantization],
        filter_fn=lambda module, fqn: (
            isinstance(module, nn.Linear) and not fqn.endswith("lm_head")
        ),
    )
    quantization_seconds = time.perf_counter() - start
    return model, quantization_seconds, len(selected_names), 0


def _parity_check(
    model: torch.nn.Module,
    prepared: PreparedHarnessLlr,
    *,
    n_variants: int,
    execution_layout: ExecutionLayout,
) -> dict[str, float | int]:
    metadata = prepared.metadata
    variant_number = metadata.groupby(
        VARIANT_KEY_COLUMNS, sort=False, dropna=False
    ).ngroup()
    indices = np.flatnonzero(variant_number.to_numpy() < n_variants)
    assert len(indices) == 2 * min(
        n_variants, metadata[VARIANT_KEY_COLUMNS].drop_duplicates().shape[0]
    )
    input_ids = prepared.input_ids[indices].cuda()
    alt_token_id = prepared.alt_token_id[indices].cuda()
    nuc_token_ids = prepared.nuc_token_ids.cuda()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        bundled = compute_variant_score_bundle(
            model,
            input_ids,
            alt_token_id,
            var_pos=prepared.var_pos,
            nuc_token_ids=nuc_token_ids,
        )
        if execution_layout == "prefix-cache":
            llr_only = compute_variant_llr(
                model,
                input_ids,
                alt_token_id,
                var_pos=prepared.var_pos,
                nuc_token_ids=nuc_token_ids,
            )
        elif execution_layout == "sequential-branches":
            llr_only = compute_variant_llr_sequential_branches(
                model,
                input_ids,
                alt_token_id,
                var_pos=prepared.var_pos,
                nuc_token_ids=nuc_token_ids,
            )
        elif execution_layout == "branch-packed":
            llr_only = compute_variant_llr_branch_packed(
                model,
                input_ids,
                alt_token_id,
                var_pos=prepared.var_pos,
                nuc_token_ids=nuc_token_ids,
            )
        else:
            assert execution_layout == "full-pair"
            llr_only = compute_variant_llr_full_pair(
                model,
                input_ids,
                alt_token_id,
                var_pos=prepared.var_pos,
                nuc_token_ids=nuc_token_ids,
            )
    torch.cuda.synchronize()
    delta = (llr_only - bundled[:, 0]).abs().float()
    assert torch.isfinite(delta).all()
    return {
        "n_rows": int(len(indices)),
        "max_abs_llr_delta": float(delta.max().item()),
        "mean_abs_llr_delta": float(delta.mean().item()),
    }


def _combine_separate_strands(
    plus: LlrBenchmarkResult,
    minus: LlrBenchmarkResult,
    n_rows: int,
) -> LlrBenchmarkResult:
    assert len(plus.repeat_seconds) == len(minus.repeat_seconds)
    row_indices = np.concatenate([plus.row_indices, minus.row_indices])
    llr = np.concatenate([plus.llr, minus.llr])
    order = np.argsort(row_indices)
    row_indices = row_indices[order]
    llr = llr[order]
    np.testing.assert_array_equal(row_indices, np.arange(n_rows))
    return LlrBenchmarkResult(
        row_indices=row_indices,
        llr=llr,
        warmup_seconds=plus.warmup_seconds + minus.warmup_seconds,
        repeat_seconds=tuple(
            plus_seconds + minus_seconds
            for plus_seconds, minus_seconds in zip(
                plus.repeat_seconds, minus.repeat_seconds, strict=True
            )
        ),
        peak_vram_allocated_bytes=max(
            plus.peak_vram_allocated_bytes, minus.peak_vram_allocated_bytes
        ),
        peak_vram_reserved_bytes=max(
            plus.peak_vram_reserved_bytes, minus.peak_vram_reserved_bytes
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(Path.home() / "ckpt"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--subset", default="missense_variant")
    parser.add_argument("--batching", choices=["separate", "fused"], default="separate")
    parser.add_argument(
        "--execution-layout",
        choices=[
            "prefix-cache",
            "sequential-branches",
            "branch-packed",
            "full-pair",
        ],
        default="prefix-cache",
    )
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument(
        "--quantization",
        choices=QUANTIZATION_CHOICES,
        default="none",
    )
    parser.add_argument(
        "--compile-mode",
        choices=[
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ],
        default=None,
    )
    parser.add_argument("--dynamo-recompile-limit", type=int, default=None)
    parser.add_argument("--te-amax-history-len", type=int, default=16)
    parser.add_argument(
        "--te-amax-compute-algo", choices=["max", "most_recent"], default="max"
    )
    parser.add_argument("--te-fp8-model-init", action="store_true")
    parser.add_argument("--te-fused-mlp", action="store_true")
    parser.add_argument("--parity-variants", type=int, default=32)
    parser.add_argument("--price-per-hour", type=float, default=2.29)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA GPU required"
    assert args.torch_compile or args.compile_mode is None
    assert args.te_amax_history_len > 0
    assert not args.te_fp8_model_init or args.quantization == "te-fp8-delayed"
    assert not args.te_fused_mlp or args.quantization in (
        "te-bf16",
        "te-fp8-delayed",
    )
    assert not (args.te_fused_mlp and args.te_fp8_model_init), (
        "fused MLP + FP8 model init is a separate, untested combination"
    )
    if args.dynamo_recompile_limit is not None:
        assert args.torch_compile, "Dynamo recompile limit requires torch_compile=True"
        assert args.dynamo_recompile_limit > 0
        torch._dynamo.config.recompile_limit = args.dynamo_recompile_limit
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preprocess_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint,
        revision=MODEL_REVISION
        if args.checkpoint == "marin-dna/marin-dna-exp135-m5.1"
        else None,
    )
    harness = load_dataset(
        DATASET_ID,
        revision=DATASET_REVISION,
        split="train",
    ).to_pandas()
    selected_subset = None if args.subset == "all" else args.subset
    prepared = prepare_harness_llr(harness, tokenizer, subset=selected_subset)
    preprocessing_seconds = time.perf_counter() - preprocess_start

    model_load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        revision=MODEL_REVISION
        if args.checkpoint == "marin-dna/marin-dna-exp135-m5.1"
        else None,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).cuda()
    model.eval()
    model_load_seconds = time.perf_counter() - model_load_start
    conversion_reference = None
    conversion_input_ids = None
    conversion_alt_token_id = None
    if args.te_fused_mlp:
        conversion_rows = min(2 * args.parity_variants, len(prepared.metadata))
        conversion_input_ids = prepared.input_ids[:conversion_rows].cuda()
        conversion_alt_token_id = prepared.alt_token_id[:conversion_rows].cuda()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            conversion_reference = compute_variant_llr(
                model,
                conversion_input_ids,
                conversion_alt_token_id,
                var_pos=prepared.var_pos,
                nuc_token_ids=prepared.nuc_token_ids.cuda(),
            ).float()
        torch.cuda.synchronize()

    (
        model,
        quantization_seconds,
        quantized_linear_count,
        fused_mlp_count,
    ) = _quantize_model(
        model,
        args.quantization,
        te_amax_history_len=args.te_amax_history_len,
        te_amax_compute_algo=args.te_amax_compute_algo,
        te_fp8_model_init=args.te_fp8_model_init,
        te_fused_mlp=args.te_fused_mlp,
    )
    conversion_parity = None
    if args.te_fused_mlp:
        assert conversion_reference is not None
        assert conversion_input_ids is not None
        assert conversion_alt_token_id is not None
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            conversion_candidate = compute_variant_llr(
                model,
                conversion_input_ids,
                conversion_alt_token_id,
                var_pos=prepared.var_pos,
                nuc_token_ids=prepared.nuc_token_ids.cuda(),
            ).float()
        torch.cuda.synchronize()
        conversion_delta = (conversion_candidate - conversion_reference).abs()
        assert torch.isfinite(conversion_delta).all()
        conversion_parity = {
            "n_rows": int(len(conversion_delta)),
            "max_abs_llr_delta": float(conversion_delta.max().item()),
            "mean_abs_llr_delta": float(conversion_delta.mean().item()),
        }
        assert conversion_parity["mean_abs_llr_delta"] <= 2.0, conversion_parity
        assert conversion_parity["max_abs_llr_delta"] <= 20.0, conversion_parity

    parity = _parity_check(
        model,
        prepared,
        n_variants=args.parity_variants,
        execution_layout=args.execution_layout,
    )
    if args.execution_layout in (
        "prefix-cache",
        "sequential-branches",
    ) and not args.quantization.startswith("te-"):
        assert parity["max_abs_llr_delta"] <= 1e-6, parity
    elif args.execution_layout in ("prefix-cache", "sequential-branches"):
        assert parity["mean_abs_llr_delta"] <= 2.0, parity
        assert parity["max_abs_llr_delta"] <= 20.0, parity
    else:
        assert parity["mean_abs_llr_delta"] <= 0.5, parity
        assert parity["max_abs_llr_delta"] <= 5.0, parity

    common = {
        "batch_size": args.batch_size,
        "device": "cuda",
        "repetitions": args.repetitions,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "torch_compile": args.torch_compile,
        "compile_mode": args.compile_mode,
        "use_bf16_autocast": True,
        "execution_layout": args.execution_layout,
    }
    if args.batching == "fused":
        benchmark = benchmark_prepared_llr(model, prepared, **common)
    else:
        plus_indices = np.flatnonzero(prepared.metadata["strand"].to_numpy() == "+")
        minus_indices = np.flatnonzero(prepared.metadata["strand"].to_numpy() == "-")
        plus = benchmark_prepared_llr(
            model, prepared, row_indices=plus_indices, **common
        )
        minus = benchmark_prepared_llr(
            model, prepared, row_indices=minus_indices, **common
        )
        benchmark = _combine_separate_strands(
            plus, minus, n_rows=len(prepared.metadata)
        )

    variant_scores = aggregate_harness_llr(
        prepared, benchmark.row_indices, benchmark.llr
    )
    auprc = float(
        average_precision_score(
            variant_scores["target"].astype(int),
            variant_scores["minus_llr_avg"],
        )
    )
    n_variants = len(variant_scores)
    variants_per_second = n_variants / benchmark.median_seconds
    seconds_per_million = 1_000_000 / variants_per_second
    dollars_per_million = seconds_per_million / 3600 * args.price_per_hour
    variant_scores.to_parquet(out_dir / "scores.parquet", index=False)

    properties = torch.cuda.get_device_properties(0)
    summary = {
        "model": "marin-dna/marin-dna-exp135-m5.1",
        "model_revision": MODEL_REVISION,
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "split": "train",
        "subset": args.subset,
        "n_variants": n_variants,
        "n_strand_rows": len(prepared.metadata),
        "batching": args.batching,
        "execution_layout": args.execution_layout,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "torch_compile": args.torch_compile,
        "compile_mode": args.compile_mode,
        "dynamo_recompile_limit": args.dynamo_recompile_limit,
        "quantization": args.quantization,
        "te_amax_history_len": (
            args.te_amax_history_len if args.quantization.startswith("te-") else None
        ),
        "te_amax_compute_algo": (
            args.te_amax_compute_algo if args.quantization.startswith("te-") else None
        ),
        "te_fp8_model_init": args.te_fp8_model_init,
        "te_fused_mlp": args.te_fused_mlp,
        "fused_mlp_count": fused_mlp_count,
        "quantization_seconds": quantization_seconds,
        "quantized_linear_count": quantized_linear_count,
        "conversion_parity": conversion_parity,
        "preprocessing_seconds": preprocessing_seconds,
        "model_load_seconds": model_load_seconds,
        "warmup_seconds": benchmark.warmup_seconds,
        "repeat_seconds": benchmark.repeat_seconds,
        "median_seconds": benchmark.median_seconds,
        "variants_per_second": variants_per_second,
        "variants_per_hour": variants_per_second * 3600,
        "seconds_per_million": seconds_per_million,
        "price_per_hour": args.price_per_hour,
        "dollars_per_million": dollars_per_million,
        "peak_vram_allocated_gib": benchmark.peak_vram_allocated_bytes / BYTES_PER_GIB,
        "peak_vram_reserved_gib": benchmark.peak_vram_reserved_bytes / BYTES_PER_GIB,
        "auprc": auprc,
        "parity": parity,
        "gpu": properties.name,
        "gpu_total_memory_gib": properties.total_memory / BYTES_PER_GIB,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torchao": (
            importlib.metadata.version("torchao")
            if args.quantization not in ("none", "te-bf16", "te-fp8-delayed")
            else None
        ),
        "transformer_engine": (
            importlib.metadata.version("transformer-engine")
            if args.quantization.startswith("te-")
            else None
        ),
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "platform": platform.platform(),
    }
    if args.subset == "missense_variant":
        summary["missense_auprc"] = auprc
    (out_dir / "benchmark.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
