# Restricted Torch forward benchmark

This follow-up to PR #590 measures only the warmed device forward on one H100.
Tokenization, H2D, D2H, model loading, compilation and warmup are outside the CUDA-event interval.
Prefetch is disabled, and inputs are ordinary complete unique 8,192-base windows.
The source model remains PR #590's pinned pretrained 1x step-412290 checkpoint, rather than the later 1T-token model.

The starting configuration is native grouped MoE, vLLM-packaged FA2 with native GQA, compact ACGTN input and A/C/G/T output weights, compiled direct short convolution, precomputed rotary tables, combined shared-expert projections, and `TORCHINDUCTOR_FREEZING=1`.
Each trial runs in a fresh process and checks the fixed 32-window native-JAX reference before timing.
The existing probability/NLL/accuracy correctness gate is retained.

`forward_torch.py` measures an individual configuration.
`forward_sweep.py` executes an explicit JSON trial list serially and writes each result and log to durable experiment storage.
Both live in `src/exp586_moe/inference/`.
The constructor's `prepare_vllm_moe=False` avoids allocating unused fused-MoE packing when using the grouped backend.

The initial screen brackets power-of-two capacity, measures the throughput frontier, then refines non-power-of-two batches around the observed peak.
Fresh-process one-factor ablations switch freezing off, specialized top-k on, short convolution to `conv1d`, and combined shared projections off at a common batch.
Controls and consequential differences are repeated before interpretation.
Exact arguments, event samples, versions, provenance, memory, correctness and failures are retained in JSON results.

All Iris submissions use `--priority batch --user eczech`, whole eight-H100 nodes, and `CUDA_VISIBLE_DEVICES=0` for measured processes.
The allocation stays running for user-directed iteration.
User authorization limits communication to this chat; no GitHub updates are made.
The durable owner is `s3://marin-us-east-02a/MarinDNA/exp586_plantcad2_moe/inference-optimization/forward-only/2026-11-06-r1`.
