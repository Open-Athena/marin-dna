"""Benchmark the terminal pooled-embeddings contract for issue #430."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from sklearn.metrics import average_precision_score
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark_llr import _quantize_model

from marin_dna.model.scoring import compute_variant_score_bundle
from marin_dna.pipelines.evals.inference_benchmark import (
    VARIANT_KEY_COLUMNS,
    PreparedHarnessLlr,
    aggregate_harness_llr,
    prepare_harness_llr,
)


MODEL_REVISION = "c0676b2012b8b9c526deb26ff517f6b92b6d375d"
DATASET_ID = "marin-dna/evals_mendelian_traits_harness_255"
DATASET_REVISION = "7b92f047f9a36f90e9ac47886afa2a99264ee35c"
BYTES_PER_GIB = 1024**3


class _EmbeddingBundleModule(nn.Module):
    def __init__(self, model: nn.Module, prepared: PreparedHarnessLlr) -> None:
        super().__init__()
        self.model = model
        self.var_pos = prepared.var_pos
        self.register_buffer("nuc_token_ids", prepared.nuc_token_ids)

    def forward(self, input_ids: Tensor, alt_token_id: Tensor) -> Tensor:
        return compute_variant_score_bundle(
            self.model,
            input_ids,
            alt_token_id,
            var_pos=self.var_pos,
            nuc_token_ids=self.nuc_token_ids,
            return_embeddings=True,
            pool_lo=1,
            pool_hi=256,
        )


def _autocast_context() -> Any:
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _run_rows(
    wrapped: nn.Module,
    prepared: PreparedHarnessLlr,
    row_indices: np.ndarray,
    *,
    batch_size: int,
    repetitions: int,
    num_workers: int,
    prefetch_factor: int,
) -> tuple[np.ndarray, float, tuple[float, ...]]:
    pad_n = (batch_size - len(row_indices) % batch_size) % batch_size
    padded_indices = (
        np.concatenate([row_indices, np.repeat(row_indices[-1], pad_n)])
        if pad_n > 0
        else row_indices
    )
    index_tensor = torch.from_numpy(padded_indices)
    loader = DataLoader(
        TensorDataset(
            prepared.input_ids[index_tensor],
            prepared.alt_token_id[index_tensor],
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    assert len(loader) > 0

    warm_input_ids, warm_alt = next(iter(loader))
    warm_input_ids = warm_input_ids.cuda(non_blocking=True)
    warm_alt = warm_alt.cuda(non_blocking=True)
    torch.cuda.synchronize()
    warm_start = time.perf_counter()
    with torch.inference_mode(), _autocast_context():
        warm_output = wrapped(warm_input_ids, warm_alt)
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - warm_start
    assert warm_output.ndim == 2 and warm_output.shape[0] == batch_size
    assert warm_output.shape[1] > 2

    repeat_seconds: list[float] = []
    last_output: np.ndarray | None = None
    for _ in range(repetitions):
        chunks: list[np.ndarray] = []
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode(), _autocast_context():
            for batch_input_ids, batch_alt in loader:
                batch_input_ids = batch_input_ids.cuda(non_blocking=True)
                batch_alt = batch_alt.cuda(non_blocking=True)
                output = wrapped(batch_input_ids, batch_alt)
                chunks.append(output.float().cpu().numpy())
        combined = np.concatenate(chunks, axis=0)[: len(row_indices)]
        torch.cuda.synchronize()
        repeat_seconds.append(time.perf_counter() - start)
        assert combined.shape == (len(row_indices), warm_output.shape[1])
        assert np.isfinite(combined).all()
        last_output = combined
    assert last_output is not None
    return last_output, warmup_seconds, tuple(repeat_seconds)


def _parity_check(
    model: nn.Module,
    prepared: PreparedHarnessLlr,
    n_rows: int = 32,
) -> dict[str, float | int]:
    n_rows = min(n_rows, len(prepared.metadata))
    input_ids = prepared.input_ids[:n_rows].cuda()
    alt_token_id = prepared.alt_token_id[:n_rows].cuda()
    nuc_token_ids = prepared.nuc_token_ids.cuda()
    with torch.inference_mode(), _autocast_context():
        without_embeddings = compute_variant_score_bundle(
            model,
            input_ids,
            alt_token_id,
            var_pos=prepared.var_pos,
            nuc_token_ids=nuc_token_ids,
        )
        with_embeddings = compute_variant_score_bundle(
            model,
            input_ids,
            alt_token_id,
            var_pos=prepared.var_pos,
            nuc_token_ids=nuc_token_ids,
            return_embeddings=True,
            pool_lo=1,
            pool_hi=256,
        )
    torch.cuda.synchronize()
    delta = (without_embeddings - with_embeddings[:, :2]).abs().float()
    assert torch.isfinite(delta).all()
    return {
        "n_rows": n_rows,
        "max_abs_score_delta": float(delta.max().item()),
        "mean_abs_score_delta": float(delta.mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(Path.home() / "ckpt"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--subset", default="missense_variant")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default=None,
    )
    parser.add_argument("--dynamo-recompile-limit", type=int, default=None)
    parser.add_argument(
        "--quantization", choices=["none", "te-fp8-delayed"], default="none"
    )
    parser.add_argument("--te-amax-history-len", type=int, default=1)
    parser.add_argument(
        "--te-amax-compute-algo", choices=["max", "most_recent"], default="most_recent"
    )
    parser.add_argument("--te-fused-mlp", action="store_true")
    parser.add_argument("--te-fused-qkv", action="store_true")
    parser.add_argument("--price-per-hour", type=float, default=2.29)
    parser.add_argument(
        "--save-scores",
        action="store_true",
        help="Write production-compatible f16 emb_ref/emb_alt columns to scores.parquet",
    )
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA GPU required"
    assert args.batch_size > 0
    assert args.num_workers >= 0
    assert args.prefetch_factor >= 1
    assert args.repetitions >= 1
    assert args.torch_compile or args.compile_mode is None
    assert not (args.te_fused_mlp or args.te_fused_qkv) or (
        args.quantization == "te-fp8-delayed"
    )
    if args.dynamo_recompile_limit is not None:
        assert args.torch_compile
        assert args.dynamo_recompile_limit > 0
        torch._dynamo.config.recompile_limit = args.dynamo_recompile_limit
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    harness = load_dataset(
        DATASET_ID,
        revision=DATASET_REVISION,
        split="train",
    ).to_pandas()
    prepared = prepare_harness_llr(harness, tokenizer, subset=args.subset)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        revision=(
            MODEL_REVISION
            if args.checkpoint == "marin-dna/marin-dna-exp135-m5.1"
            else None
        ),
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).cuda()
    model.eval()
    (
        model,
        quantization_seconds,
        quantized_linear_count,
        fused_mlp_count,
        fused_qkv_count,
    ) = _quantize_model(
        model,
        args.quantization,
        te_amax_history_len=args.te_amax_history_len,
        te_amax_compute_algo=args.te_amax_compute_algo,
        te_fp8_model_init=False,
        te_fused_mlp=args.te_fused_mlp,
        te_fused_qkv=args.te_fused_qkv,
    )
    parity = _parity_check(model, prepared)
    assert parity["max_abs_score_delta"] <= 1e-6, parity

    wrapped: nn.Module = _EmbeddingBundleModule(model, prepared).cuda().eval()
    if args.torch_compile:
        compile_kwargs: dict[str, object] = {}
        if args.compile_mode is not None:
            compile_kwargs["mode"] = args.compile_mode
        wrapped = torch.compile(wrapped, **compile_kwargs)

    plus_indices = np.flatnonzero(prepared.metadata["strand"].to_numpy() == "+")
    minus_indices = np.flatnonzero(prepared.metadata["strand"].to_numpy() == "-")
    plus_keys = prepared.metadata.iloc[plus_indices][VARIANT_KEY_COLUMNS].reset_index(
        drop=True
    )
    minus_keys = prepared.metadata.iloc[minus_indices][VARIANT_KEY_COLUMNS].reset_index(
        drop=True
    )
    assert plus_keys.equals(minus_keys), "FWD/RC variant order differs"

    torch.cuda.reset_peak_memory_stats()
    plus_output, plus_warmup, plus_seconds = _run_rows(
        wrapped,
        prepared,
        plus_indices,
        batch_size=args.batch_size,
        repetitions=args.repetitions,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    minus_output, minus_warmup, minus_seconds = _run_rows(
        wrapped,
        prepared,
        minus_indices,
        batch_size=args.batch_size,
        repetitions=args.repetitions,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    repeat_seconds = tuple(
        plus + minus for plus, minus in zip(plus_seconds, minus_seconds, strict=True)
    )
    median_seconds = float(np.median(repeat_seconds))

    row_indices = np.concatenate([plus_indices, minus_indices])
    row_llr = np.concatenate([plus_output[:, 0], minus_output[:, 0]])
    scores = aggregate_harness_llr(prepared, row_indices, row_llr)
    auprc = float(
        average_precision_score(scores["target"].astype(int), scores["minus_llr_avg"])
    )
    averaged_embeddings = 0.5 * (plus_output[:, 2:] + minus_output[:, 2:])
    assert averaged_embeddings.shape[0] == len(scores)
    assert averaged_embeddings.shape[1] % 2 == 0
    embedding_checksum = float(averaged_embeddings.astype(np.float64).sum())
    if args.save_scores:
        embedding_width = averaged_embeddings.shape[1] // 2
        stored_embeddings = averaged_embeddings.astype(np.float16)
        assert np.isfinite(stored_embeddings).all(), (
            "non-finite pooled embedding after f16 cast"
        )
        scores["emb_ref"] = list(stored_embeddings[:, :embedding_width])
        scores["emb_alt"] = list(stored_embeddings[:, embedding_width:])
        scores.to_parquet(out_dir / "scores.parquet", index=False)

    n_variants = len(scores)
    variants_per_second = n_variants / median_seconds
    seconds_per_million = 1_000_000 / variants_per_second
    properties = torch.cuda.get_device_properties(0)
    summary = {
        "model": "marin-dna/marin-dna-exp135-m5.1",
        "model_revision": MODEL_REVISION,
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "subset": args.subset,
        "n_variants": n_variants,
        "n_strand_rows": len(prepared.metadata),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "torch_compile": args.torch_compile,
        "compile_mode": args.compile_mode,
        "dynamo_recompile_limit": args.dynamo_recompile_limit,
        "quantization": args.quantization,
        "te_amax_history_len": args.te_amax_history_len,
        "te_amax_compute_algo": args.te_amax_compute_algo,
        "te_fused_mlp": args.te_fused_mlp,
        "fused_mlp_count": fused_mlp_count,
        "te_fused_qkv": args.te_fused_qkv,
        "fused_qkv_count": fused_qkv_count,
        "quantization_seconds": quantization_seconds,
        "quantized_linear_count": quantized_linear_count,
        "output_contract": "llr+jsd+ref_embedding+alt_embedding; FWD+RC",
        "per_strand_output_width": int(plus_output.shape[1]),
        "fwd_rc_averaged_embedding_width": int(averaged_embeddings.shape[1]),
        "embedding_checksum": embedding_checksum,
        "scores_saved": args.save_scores,
        "warmup_seconds": plus_warmup + minus_warmup,
        "repeat_seconds": repeat_seconds,
        "median_seconds": median_seconds,
        "variants_per_second": variants_per_second,
        "variants_per_hour": variants_per_second * 3600,
        "seconds_per_million": seconds_per_million,
        "price_per_hour": args.price_per_hour,
        "dollars_per_million": seconds_per_million / 3600 * args.price_per_hour,
        "peak_vram_allocated_gib": torch.cuda.max_memory_allocated() / BYTES_PER_GIB,
        "peak_vram_reserved_gib": torch.cuda.max_memory_reserved() / BYTES_PER_GIB,
        "auprc": auprc,
        "parity": parity,
        "gpu": properties.name,
        "gpu_total_memory_gib": properties.total_memory / BYTES_PER_GIB,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformer_engine": (
            importlib.metadata.version("transformer-engine")
            if args.quantization.startswith("te-")
            else None
        ),
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "platform": platform.platform(),
    }
    (out_dir / "benchmark.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
