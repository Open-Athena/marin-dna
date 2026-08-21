---
name: run-model-inference
description: Implement, optimize, and validate MarinDNA model inference workloads. Use when building a new or one-off inference loop, choosing an automatic prediction loop, configuring accelerator data loading and compilation, checking numerical parity, or measuring inference throughput.
---

# Run Model Inference

Treat one-off and experimental inference loops as production inference workloads.

## Use An Established Loop

- Prefer an established automatic prediction loop, such as Hugging Face `Trainer.predict`, when it can express the required outputs.
- Record the missing capability when a custom loop is necessary.
- Keep batching, device transfer, output ordering, and failure handling explicit in a custom loop.

## Apply Standard Optimizations

- Batch accelerator inference.
- Use bfloat16 (`bf16`), model compilation, multiple data-loader workers, pinned memory, and prefetching when the model, hardware, and framework support them.
- Record why any applicable optimization is disabled.

## Check Correctness And Throughput

1. Define the output fields and numerical tolerances before a long run.
2. Compare a small sample against an eager, uncompiled fp32 or documented higher-precision reference path.
3. Validate reduced precision and compilation separately so a discrepancy can be attributed to one change.
4. Measure steady-state throughput after warmup.
   State the unit and whether data loading and preprocessing are included.

## Compose Existing Skills

- Use `evaluate-models` for VEP split protection, benchmark filtering, aggregation, and result presentation.
- Use `develop-snakemake-pipelines` when a maintained Snakemake workflow owns inference.
- Use `marin-experiment` when a Marin-launched experiment owns inference.
- Use `manage-research-storage` for durable predictions and performance artifacts.
- Use `wandb-reporting` for dense run metrics and comparisons.
