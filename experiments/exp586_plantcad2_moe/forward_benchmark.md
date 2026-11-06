# Restricted Torch forward benchmark

This follow-up to PR #590 measures only the warmed device forward on one H100.
Tokenization, H2D, D2H, model loading, compilation and warmup are outside the CUDA-event interval.
Prefetch is disabled, and inputs are ordinary complete unique 8,192-base windows.
The source model is `plantcad/marin-dna-exp586-1.5B-60M-216B` at revision `4130f71edcd117a7375af1ac78723d9a9173533f`.
Its configuration matches PR #590's pinned pretrained 1x step-412290 export.
Downloaded LFS weight/tokenizer SHA-256 values are verified before the experiment.

The starting configuration is native grouped MoE, vLLM-packaged FA2 with native GQA, compact ACGTN input and A/C/G/T output weights, compiled direct short convolution, precomputed rotary tables, combined shared-expert projections, and `TORCHINDUCTOR_FREEZING=1`.
`torch.compile` uses `fullgraph=True`, `dynamic=False` and `mode="reduce-overhead"`; `specialized_topk=False` in the baseline.
Weights use BF16; host outputs are converted to FP32 outside timing.
Each trial runs in a fresh process and checks the fixed 32-window native-JAX reference before timing.
The existing probability/NLL/accuracy correctness gate is retained.

`forward_torch.py` measures an individual configuration.
`forward_sweep.py` executes an explicit JSON trial list serially and writes each result and log to durable experiment storage.
Both live in `src/exp586_moe/inference/`.
`examples/restricted_torch_logits.py` demonstrates ordinary model usage with the same baseline configuration.
The constructor's `prepare_vllm_moe=False` avoids allocating unused fused-MoE packing when using the grouped backend.

The initial screen brackets power-of-two capacity, measures the throughput frontier, then refines non-power-of-two batches around the observed peak.
Fresh-process one-factor ablations switch freezing off, specialized top-k on, short convolution to `conv1d`, and combined shared projections off at a common batch.
Controls and consequential differences are repeated before interpretation.
Exact arguments, event samples, versions, provenance, memory, correctness and failures are retained in JSON results.
Batch capacity includes loading, compilation, parity checking and warmup.
Use reserved memory or driver-observed device usage to interpret capacity; the ordinary allocated-memory counter understates private CUDA-graph pool usage.

All Iris submissions use `--priority batch --user eczech`, whole eight-H100 nodes, and `CUDA_VISIBLE_DEVICES=0` for measured processes.
The allocation stays running for user-directed iteration.
User authorization limits communication to this chat; no GitHub updates are made.
The durable owner is `s3://marin-us-east-02a/MarinDNA/exp586_plantcad2_moe/inference-optimization/forward-only/2026-11-06-r1`.


## Results — 2026-11-06

There are 49 successfully timed trials and two expected capacity OOM observations (batch 320 and 512).
Every timed trial passed the same native-JAX probability/NLL/accuracy gate.
All replicated ablations and final batch confirmations used GPU UUID `GPU-b5e2efea-3217-4717-94ab-4a1fe6a348e9` on an H100 80GB.
The table reports the median of three fresh-process medians at batch 64, with eight warmups and 32 timed forwards per process.

| Configuration | Forward ms | DNA tokens/second | Throughput change |
| --- | ---: | ---: | ---: |
| Restricted control | 491.763 | 1,066,139 | +0.00% |
| Freezing disabled | 495.578 | 1,057,933 | -0.77% |
| Separate shared projections | 497.047 | 1,054,806 | -1.06% |
| `sconv_backend="conv1d"` | 615.200 | 852,224 | -20.06% |
| Custom `specialized_topk=True` | 423.533 | 1,237,890 | +16.11% |

The custom top-k case is an explicitly requested ablation and remains outside the restricted baseline.
`direct` short convolution uses ordinary Torch multiply/add/pad operations compiled by Inductor; the separate bespoke Triton short-convolution backend is excluded.
These measurements isolate one flag at a common batch and do not establish identical effect sizes at every batch or on other hardware.

The largest fitting power of two is batch 256.
The highest tested fitting batch is 304; 320 and 512 OOM during the complete compile/validation/warmup protocol.
A coarse screen and 12 additional non-power-of-two sizes identified a flat plateau, with 80–112 nearly tied.
Batch 96 was the highest-throughput screening observation, and a separate interleaved three-process confirmation compared it with batch 256:

| Batch | Median forward ms | DNA tokens/second |
| --- | ---: | ---: |
| 96 | 736.010 | 1,068,507 |
| 256 | 1979.807 | 1,059,271 |

The confirmed throughput gain is 0.87%.
The measured plateau supports a moderate operating batch; 96 is not an exact universal optimum.
The batch-96 process medians span 1,068,068–1,069,335 DNA tokens/second; batch 256 spans 1,059,021–1,059,306.

The throwaway usage example was executed on the GPU against the pinned downloaded HF snapshot and returned `(1, 8192, 4)` A/C/G/T logits.
CPU verification completed with 129 passed and 3 skipped; scoped pre-commit checks passed.
Final summaries, trial CSV, source archive and checksum manifest live under the experiment prefix's `final/` directory.
The retained Iris root is `/eczech/exp586-torch-forward-20261106-a4`; its main task stays alive, while all benchmark workers have exited for user review.
PR #590 and other GitHub artifacts were not updated.
