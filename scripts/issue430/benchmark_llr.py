"""Benchmark one exact BF16 LLR-only configuration for issue #430."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path

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
    compute_variant_score_bundle,
)
from marin_dna.pipelines.evals.inference_benchmark import (
    VARIANT_KEY_COLUMNS,
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
    "int8-dynamic",
    "int8-weight-only",
    "int4-weight-only",
]


def _quantize_model(model: nn.Module, quantization: str) -> tuple[float, int]:
    if quantization == "none":
        return 0.0, 0

    from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig,
        Int4WeightOnlyConfig,
        Int8DynamicActivationInt8WeightConfig,
        Int8WeightOnlyConfig,
        quantize_,
    )

    configs = {
        "fp8-dynamic": Float8DynamicActivationFloat8WeightConfig(),
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
    return quantization_seconds, len(selected_names)


def _parity_check(
    model: torch.nn.Module,
    prepared: PreparedHarnessLlr,
    *,
    n_variants: int,
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
        llr_only = compute_variant_llr(
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
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument(
        "--quantization",
        choices=QUANTIZATION_CHOICES,
        default="none",
    )
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default=None,
    )
    parser.add_argument("--parity-variants", type=int, default=32)
    parser.add_argument("--price-per-hour", type=float, default=2.29)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA GPU required"
    assert args.torch_compile or args.compile_mode is None
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
    quantization_seconds, quantized_linear_count = _quantize_model(
        model, args.quantization
    )

    parity = _parity_check(
        model,
        prepared,
        n_variants=args.parity_variants,
    )
    assert parity["max_abs_llr_delta"] <= 1e-6, parity

    common = {
        "batch_size": args.batch_size,
        "device": "cuda",
        "repetitions": args.repetitions,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "torch_compile": args.torch_compile,
        "compile_mode": args.compile_mode,
        "use_bf16_autocast": True,
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
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "torch_compile": args.torch_compile,
        "compile_mode": args.compile_mode,
        "quantization": args.quantization,
        "quantization_seconds": quantization_seconds,
        "quantized_linear_count": quantized_linear_count,
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
            if args.quantization != "none"
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
