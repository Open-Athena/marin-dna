"""Bounded synthetic runtime check; contains no biological evaluation rows."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from datasets import Dataset

from marin_dna_evals.hf_compat import load_hf_causal_lm_and_tokenizer
from marin_dna_evals.rag import score_rag_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    rng = np.random.default_rng(42)
    rows = []
    for count in [1, 2, 20, 40, 40, 20, 2, 1]:
        segments = ["".join(rng.choice(list("ACGT"), 255)) for _ in range(count)]
        ref = segments[-1][127]
        rows.append({
            "sequence": "[SEQ]".join(segments),
            "species_order": [f"synthetic_{i}" for i in range(count - 1)] + ["hg38"],
            "ref": ref,
            "alt": next(base for base in "ACGT" if base != ref),
        })
    fixture = Dataset.from_list(rows)
    throughput_data = Dataset.from_list(rows * 12)
    # These gates are fixed before execution. Score tolerances carry forward
    # the maintained GPU runtime gate; embeddings add explicit fp32 parity.
    tolerances = {
        "fp32_repeat": {"llr": [0.0001, 0.15], "jsd": [0.001, 0.0001], "embedding": [0.01, 0.03]},
        "compiled_vs_fp32": {"llr": [0.0001, 0.01], "jsd": [0.001, 0.00001], "embedding": [0.005, 0.01]},
    }
    report = {
        "synthetic_only": True,
        "prior_batch_sweep": "Batch 4 delivered 0.89245 variants/sec at 14.66 GB peak allocation versus batch 2 at 0.89759 and 7.32 GB; batch 8 exceeded the A10G memory capacity. Final comparison uses batch 2.",
        "precision_policy": "fp32 parameters, activations, and matrix multiplication; bf16 failed reference parity and TF32 compilation narrowly exceeded the predeclared compiler LLR tolerance, so both are disabled",
        "source_commit": "c0585d0e117075026b4384d552d59b52009d257e",
        "compatibility_fix_commit": "ddc8f9aa85ebd20515e11aee2e432e437dc71741",
        "checkpoint": "gs://marin-us-east5/MarinDNA/exp550_rag_five_regions/checkpoints/dna-exp550-rag46m-five-regions-v1-pilot-mb5/2026.09.10.3/hf/step-20",
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name()},
        "tolerances_rtol_atol": tolerances,
        "fixture_species_counts": [len(row["species_order"]) for row in rows],
        "reference_batch_note": "Use batch 2 because the pinned Trainer unwraps bare singleton-batch tensor outputs; tracked separately in the repository.",
        "timing_unit": "variants/sec including REF/ALT, FWD/RC, on-the-fly tokenization, data loading, and output collection",
        "measurements": {},
        "parity": {},
    }

    def save() -> None:
        output.write_text(json.dumps(report, indent=2) + "\n")

    def score(model, tokenizer, data, *, batch: int, bf16: bool, compiled: bool, name: str):
        kwargs = dict(tf32=False, per_device_eval_batch_size=batch, bf16_full_eval=bf16, torch_compile=compiled, dataloader_num_workers=2, dataloader_pin_memory=True, dataloader_prefetch_factor=2, remove_unused_columns=False, report_to="none", disable_tqdm=True, eval_accumulation_steps=8)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        print(f"START {name}", flush=True)
        arrays = score_rag_dataset(model, tokenizer, data, inference_kwargs=kwargs, return_embeddings=True)
        print({strand: {"shape": array.shape, "nonfinite": int(np.count_nonzero(~np.isfinite(array)))} for strand, array in arrays.items()}, flush=True)
        torch.cuda.synchronize()
        duration = time.monotonic() - start
        assert all(array.shape == (len(data), 1282) and np.isfinite(array).all() for array in arrays.values())
        report["measurements"][name] = {"variants": len(data), "seconds": duration, "variants_per_second": len(data) / duration, "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "batch": batch, "bf16": bf16, "compiled": compiled, "tf32_actual": torch.backends.cuda.matmul.allow_tf32}
        save()
        print(f"DONE {name}: {len(data) / duration:.3f} variants/sec", flush=True)
        return arrays

    def compare(candidate, reference, name: str) -> bool:
        result = {}
        for strand in ("fwd", "rc"):
            for field, column in (("llr", slice(0, 1)), ("jsd", slice(1, 2)), ("embedding", slice(2, None))):
                actual, expected = candidate[strand][:, column], reference[strand][:, column]
                rtol, atol = tolerances[name][field]
                result[f"{strand}_{field}"] = {"max_abs_error": float(np.max(np.abs(actual - expected))), "passed": bool(np.allclose(actual, expected, rtol=rtol, atol=atol))}
        report["parity"][name] = result
        save()
        return all(item["passed"] for item in result.values())

    tokenizer, model = load_hf_causal_lm_and_tokenizer(args.checkpoint)
    model = model.float().eval().cuda()
    reference = score(model, tokenizer, fixture, batch=2, bf16=False, compiled=False, name="fp32_eager")
    torch.backends.cuda.matmul.allow_tf32 = False
    reduced = score(model, tokenizer, fixture, batch=2, bf16=False, compiled=False, name="fp32_repeat")
    reduced_ok = compare(reduced, reference, "fp32_repeat")
    compiled = score(model, tokenizer, fixture, batch=2, bf16=False, compiled=True, name="fp32_compiled_cold")
    compiled_ok = compare(compiled, reduced, "compiled_vs_fp32")
    report["parity_passed"] = reduced_ok and compiled_ok
    save()
    if not report["parity_passed"]:
        raise RuntimeError("Predeclared numerical parity failed; throughput sweep skipped")
    for batch in (2,):
        score(model, tokenizer, fixture, batch=batch, bf16=False, compiled=True, name=f"warmup_batch{batch}")
        score(model, tokenizer, throughput_data, batch=batch, bf16=False, compiled=True, name=f"warmed_batch{batch}")
    best = max((2,), key=lambda batch: report["measurements"][f"warmed_batch{batch}"]["variants_per_second"])
    report["selected_batch"] = best
    report["projected_51623_variant_inference_hours"] = 51623 / report["measurements"][f"warmed_batch{best}"]["variants_per_second"] / 3600
    del model
    gc.collect()
    torch.cuda.empty_cache()
    # Compare equal synthetic work with three checkpoint loads/Trainer calls.
    start = time.monotonic()
    for index in range(3):
        tokenizer, model = load_hf_causal_lm_and_tokenizer(args.checkpoint)
        score(model.eval().cuda(), tokenizer, throughput_data.select(range(index * 32, (index + 1) * 32)), batch=best, bf16=False, compiled=True, name=f"separate_partition{index}")
        del model
        gc.collect()
        torch.cuda.empty_cache()
    separate_seconds = time.monotonic() - start
    start = time.monotonic()
    tokenizer, model = load_hf_causal_lm_and_tokenizer(args.checkpoint)
    score(model.eval().cuda(), tokenizer, throughput_data, batch=best, bf16=False, compiled=True, name="joint_with_reload")
    joint_seconds = time.monotonic() - start
    report["synthetic_partition_comparison"] = {"variants": 96, "separate_three_loads_seconds": separate_seconds, "joint_one_load_seconds": joint_seconds, "saved_seconds": separate_seconds - joint_seconds, "note": "Both use an already warmed compiler cache; includes model loading and Trainer setup, not package installation."}
    report["completed"] = True
    save()
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
