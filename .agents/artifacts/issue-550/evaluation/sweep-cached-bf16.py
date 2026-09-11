"""Measure BF16 batch throughput using the maintained cached RAG scoring path."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path


def trial(checkpoint: str, batch: int, output: Path) -> None:
    import numpy as np
    import torch
    from datasets import Dataset
    from marin_dna_evals.hf_compat import load_hf_causal_lm_and_tokenizer
    from marin_dna_evals.rag import score_rag_dataset

    torch.set_num_threads(2)
    tokenizer, model = load_hf_causal_lm_and_tokenizer(checkpoint)
    model = model.to(device="cuda", dtype=torch.bfloat16).eval()
    rng = np.random.default_rng(42)
    rows = []
    for count in (1, 2, 20, 40, 40, 20, 2, 1):
        segments = ["".join(rng.choice(list("ACGT"), 255)) for _ in range(count)]
        ref = segments[-1][127]
        rows.append(
            {
                "sequence": "[SEQ]".join(segments),
                "species_order": [f"synthetic_{i}" for i in range(count - 1)]
                + ["hg38"],
                "ref": ref,
                "alt": next(base for base in "ACGT" if base != ref),
            }
        )
    kwargs = {
        "per_device_eval_batch_size": batch,
        "bf16_full_eval": True,
        "torch_compile": True,
        "dataloader_num_workers": 2,
        "dataloader_pin_memory": True,
        "dataloader_prefetch_factor": 2,
        "remove_unused_columns": False,
        "report_to": "none",
        "disable_tqdm": True,
        "eval_accumulation_steps": 8,
    }

    def score(data: Dataset) -> float:
        torch.cuda.synchronize()
        start = time.monotonic()
        arrays = score_rag_dataset(model, tokenizer, data, inference_kwargs=kwargs)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        assert all(
            a.shape == (len(data), 1282) and np.isfinite(a).all()
            for a in arrays.values()
        )
        return elapsed

    warmup = Dataset.from_list(rows * max(1, batch // len(rows)))
    score(warmup)
    data = Dataset.from_list(rows * 16)
    torch.cuda.reset_peak_memory_stats()
    durations = [score(data), score(data)]
    result = {
        "batch": batch,
        "bf16": True,
        "compiled": True,
        "prefix_cache": True,
        "left_padded_tokens": 10240,
        "embeddings": True,
        "variants_per_trial": len(data),
        "seconds": durations,
        "variants_per_second": len(data) / statistics.median(durations),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "device_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "finite_outputs": True,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int)
    args = parser.parse_args()
    if args.batch is not None:
        trial(args.checkpoint, args.batch, args.output)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "synthetic_only": True,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "precision_authority": "User explicitly requested BF16 on EC2 A10G on 2026-09-11",
        "timing_unit": "variants/sec including REF/ALT, FWD/RC, tokenization, loading, and embeddings; compilation excluded after warmup",
        "measurements": [],
        "failed_batches": [],
        "completed": False,
    }
    for batch in (2, 4, 8, 16, 32, 64):
        path = args.output.with_name(f"batch-{batch}.json")
        log = args.output.with_name(f"batch-{batch}.log")
        print(f"BATCH_TRIAL_START {batch}", flush=True)
        with log.open("w") as stream:
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        __file__,
                        "--checkpoint",
                        args.checkpoint,
                        "--output",
                        str(path),
                        "--batch",
                        str(batch),
                    ],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=420,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                report["failed_batches"].append(
                    {"batch": batch, "reason": "420-second timeout"}
                )
                break
        if result.returncode:
            error = log.read_text()
            is_oom = "OutOfMemoryError" in error or "CUDA out of memory" in error
            report["failed_batches"].append(
                {
                    "batch": batch,
                    "exit_code": result.returncode,
                    "out_of_memory": is_oom,
                }
            )
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            if not is_oom:
                raise RuntimeError(f"Batch {batch} failed unexpectedly; see {log}")
            print(f"BATCH_TRIAL_FAILED {batch}; see {log}", flush=True)
            break
        measured = json.loads(path.read_text())
        report["measurements"].append(measured)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"BATCH_TRIAL_DONE {batch}: {measured['variants_per_second']:.3f} variants/sec",
            flush=True,
        )
        if measured["peak_allocated_bytes"] > 0.90 * measured["device_total_bytes"]:
            break
    eligible = [
        m
        for m in report["measurements"]
        if m["peak_allocated_bytes"] <= 0.90 * m["device_total_bytes"]
    ]
    if not eligible:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        raise RuntimeError("No stable BF16 batch size with memory headroom")
    best = max(eligible, key=lambda m: m["variants_per_second"])
    report["selected_batch"] = best["batch"]
    report["projected_51623_variant_hours"] = 51623 / best["variants_per_second"] / 3600
    report["completed"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("BATCH_SELECTED " + json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
