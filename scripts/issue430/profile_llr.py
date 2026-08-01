"""Profile the exact eager BF16 LLR-only GH200 baseline for issue #430."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from marin_dna.model.scoring import compute_variant_llr
from marin_dna.pipelines.evals.inference_benchmark import prepare_harness_llr


MODEL_REVISION = "c0676b2012b8b9c526deb26ff517f6b92b6d375d"
DATASET_ID = "marin-dna/evals_mendelian_traits_harness_255"
DATASET_REVISION = "7b92f047f9a36f90e9ac47886afa2a99264ee35c"
BYTES_PER_GIB = 1024**3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(Path.home() / "ckpt"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument("--profile-batches", type=int, default=8)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA GPU required"
    assert args.batch_size > 0
    assert args.num_workers >= 0
    assert args.prefetch_factor >= 1
    assert args.warmup_batches >= 1
    assert args.profile_batches >= 1
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    harness = load_dataset(
        DATASET_ID,
        revision=DATASET_REVISION,
        split="train",
    ).to_pandas()
    prepared = prepare_harness_llr(
        harness,
        tokenizer,
        subset="missense_variant",
    )
    plus_indices = np.flatnonzero(prepared.metadata["strand"].to_numpy() == "+")
    input_ids = prepared.input_ids[plus_indices]
    alt_token_id = prepared.alt_token_id[plus_indices]
    loader = DataLoader(
        TensorDataset(input_ids, alt_token_id),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        drop_last=True,
    )
    assert len(loader) >= args.warmup_batches + args.profile_batches, (
        f"need {args.warmup_batches + args.profile_batches} batches, got {len(loader)}"
    )

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
    nuc_token_ids = prepared.nuc_token_ids.cuda()

    iterator = iter(loader)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(args.warmup_batches):
            batch_input_ids, batch_alt = next(iterator)
            batch_input_ids = batch_input_ids.cuda(non_blocking=True)
            batch_alt = batch_alt.cuda(non_blocking=True)
            output = compute_variant_llr(
                model,
                batch_input_ids,
                batch_alt,
                var_pos=prepared.var_pos,
                nuc_token_ids=nuc_token_ids,
            )
            assert output.shape == (args.batch_size,)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    dataloader_wait_seconds: list[float] = []
    h2d_milliseconds: list[float] = []
    forward_milliseconds: list[float] = []
    profile_start = time.perf_counter()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as profiler:
        for _ in range(args.profile_batches):
            wait_start = time.perf_counter()
            batch_input_ids, batch_alt = next(iterator)
            dataloader_wait_seconds.append(time.perf_counter() - wait_start)

            h2d_start = torch.cuda.Event(enable_timing=True)
            h2d_end = torch.cuda.Event(enable_timing=True)
            forward_start = torch.cuda.Event(enable_timing=True)
            forward_end = torch.cuda.Event(enable_timing=True)
            h2d_start.record()
            batch_input_ids = batch_input_ids.cuda(non_blocking=True)
            batch_alt = batch_alt.cuda(non_blocking=True)
            h2d_end.record()
            forward_start.record()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                output = compute_variant_llr(
                    model,
                    batch_input_ids,
                    batch_alt,
                    var_pos=prepared.var_pos,
                    nuc_token_ids=nuc_token_ids,
                )
            forward_end.record()
            output_cpu = output.float().cpu()
            torch.cuda.synchronize()
            assert output_cpu.shape == (args.batch_size,)
            assert torch.isfinite(output_cpu).all()
            h2d_milliseconds.append(h2d_start.elapsed_time(h2d_end))
            forward_milliseconds.append(forward_start.elapsed_time(forward_end))
    profile_seconds = time.perf_counter() - profile_start

    profiler.export_chrome_trace(str(out_dir / "trace.json"))
    events = profiler.key_averages(group_by_input_shape=True)
    top_events = sorted(
        events,
        key=lambda event: event.self_device_time_total,
        reverse=True,
    )[:30]
    event_rows = [
        {
            "key": event.key,
            "count": event.count,
            "input_shapes": event.input_shapes,
            "self_cpu_time_us": event.self_cpu_time_total,
            "self_device_time_us": event.self_device_time_total,
            "cpu_memory_bytes": event.cpu_memory_usage,
            "device_memory_bytes": event.device_memory_usage,
        }
        for event in top_events
    ]
    gpu_self_time_seconds = (
        sum(event.self_device_time_total for event in events) / 1_000_000
    )
    summary = {
        "model": "marin-dna/marin-dna-exp135-m5.1",
        "model_revision": MODEL_REVISION,
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "subset": "missense_variant",
        "strand": "+",
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "warmup_batches": args.warmup_batches,
        "profile_batches": args.profile_batches,
        "profile_wall_seconds": profile_seconds,
        "dataloader_wait_seconds": dataloader_wait_seconds,
        "h2d_milliseconds": h2d_milliseconds,
        "forward_milliseconds": forward_milliseconds,
        "gpu_self_time_seconds": gpu_self_time_seconds,
        "gpu_self_time_fraction_of_wall": gpu_self_time_seconds / profile_seconds,
        "peak_vram_allocated_gib": torch.cuda.max_memory_allocated() / BYTES_PER_GIB,
        "peak_vram_reserved_gib": torch.cuda.max_memory_reserved() / BYTES_PER_GIB,
        "gpu": torch.cuda.get_device_name(0),
        "top_events_by_self_device_time": event_rows,
    }
    (out_dir / "profile.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "key_averages.txt").write_text(
        profiler.key_averages(group_by_input_shape=True).table(
            sort_by="self_device_time_total",
            row_limit=50,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
