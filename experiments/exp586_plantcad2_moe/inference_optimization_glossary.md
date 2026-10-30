# Plotted inference optimizations

This is the legend for the accepted milestones in the three optimization-progress figures. Each row describes one **backend + optimization** pair. Repeated batch-size milestones are consolidated within their backend. Baselines and confirmation rows are included so every label in the plot tables is accounted for, although they are measurements rather than optimizations.

The plots are cumulative: a milestone is the best configuration observed at that point, including all retained earlier changes. Its throughput increase is therefore not necessarily an isolated estimate of that optimization's effect. Failed candidates marked with `×` are intentionally outside this glossary.

Implementation surfaces mean:

- **Framework:** public framework or tokenizer configuration and ordinary application code.
- **Model-specific:** code specialized to the d768 Grug/Hero architecture or this fixed ACGTN workload.
- **Runtime-specific:** compiler, kernel, scheduler, or engine hooks that may change across releases.
- **Measurement:** a baseline, repeat, or final confirmation rather than an optimization.

## PyTorch

| Plot milestone | What changed | Implementation surface |
|---|---|---|
| Full-vocabulary eager | Deliberately naive baseline: eager PyTorch computes the 128,256-way language head before selecting A/C/G/T logits. | Measurement |
| A/C/G/T output head | Selects the four A/C/G/T rows of the language head before the matrix multiplication, avoiding the full-vocabulary projection while preserving the requested logits. | Framework |
| Grouped MoE kernels | Replaces the Python expert loop with three `torch._grouped_mm` calls over token/expert assignments sorted by expert: gate, up, and down projections. | Model-specific |
| Compile model | Wraps the fixed-shape forward in `torch.compile(fullgraph=True, dynamic=False, mode="reduce-overhead")`. | Runtime-specific |
| Direct HF tokenizer | Calls the Hugging Face Rust backend tokenizer directly instead of going through `PreTrainedTokenizerFast`. | Framework |
| Reduced HF tokenizer | Replaces the BPE model with an ACGTN character tokenizer that preserves the checkpoint's trained token IDs and lowercase normalization. | Framework |
| Local FlexAttention | Runs local-window layers through compiled block-masked FlexAttention instead of the expanded SDPA path. | Runtime-specific |
| Global FlashAttention | Forces the full-context layers onto the FlashAttention SDPA backend instead of the prior global-attention path. | Framework |
| Increase batch / Retune batch | Repeats batch sweeps after graph changes; plotted accepted sizes progress through 4, 8, 224, 128, 224, 16, 128, 192, and finally 240. The memory limit is not assumed to be the throughput optimum. | Framework |
| Fused MoE | Replaces native grouped GEMMs with vLLM's packed Triton `fused_experts` kernel for the routed experts. | Runtime-specific |
| Tokenizer pipeline | Uses batched Hugging Face tokenization with one-batch prefetch so CPU tokenization overlaps the preceding GPU batch. This cumulative milestone also uses inference-only top-8 routing. | Model-specific |
| Specialized router top-k | Uses a Triton top-k implementation specialized to 384 experts and top-8 routing, while preserving stable expert selection and the trained combine weights. | Model-specific |
| Direct short convolution | Replaces generic depthwise `conv1d` bookkeeping with the model's exact four-tap causal recurrence expressed as shifts, multiplies, and adds inside the compiled graph. | Model-specific |
| Batched HF tokenizer | Switches from per-sequence calls to the tokenizer's native `encode_batch` path, allowing the Rust/Rayon implementation to process the batch together. | Framework |
| FlashAttention 3 + native GQA | Uses vLLM's FlashAttention 3 kernel and passes grouped-query attention directly rather than expanding KV heads. | Runtime-specific |
| Precompute rotary tables | Computes the fixed 8,192-position RoPE sine/cosine tables once and reuses them in every layer and batch. | Model-specific |
| Combine shared-expert gate/up | Concatenates each shared expert's gate and up weights so one input GEMM produces both projections. | Model-specific |
| Shared GEMMs (short run) | Evaluates both fixed shared experts with one combined gate/up projection and one combined down projection instead of separate GEMMs per expert. | Model-specific |
| Freeze graph (short run) / Freeze inference graph | Enables TorchInductor freezing for the static inference graph; the later row measures the complete configuration for longer at batch 240. | Runtime-specific |
| Long-run confirmation | Median of three fresh, longer processes using the selected graph and batch 240; this supplies the 1,584,801 tok/s headline. | Measurement |

## vLLM

| Plot milestone | What changed | Implementation surface |
|---|---|---|
| Naive eager, full vocabulary | vLLM baseline with its fused serving engine already active, but the capture hook computes the full language head before selecting A/C/G/T. | Measurement |
| Project only A/C/G/T | Projects captured prompt states directly through the four requested output rows instead of the 128,256-way head. | Model-specific |
| + compile engine | Enables vLLM's compiled execution path for the fixed model and prefill workload. | Runtime-specific |
| + reduced HF tokenizer | Uses the character-level Hugging Face tokenizer while preserving the checkpoint's trained ACGTN IDs and normalization. | Framework |
| + tuned Triton MoE kernel | Selects the Triton fused-MoE backend and a better initial tile for this model's expert shapes. | Runtime-specific |
| Increase batch | Enlarges the caller's request batch as the engine path improves; the plotted local batches move from 1 to 4, 32, 192, 224, 236, and 240. Later queue sweeps are described separately. | Framework |
| FlashAttention 2 + batch | Selects the FlashAttention 2 prefill backend and raises the batch to 32 in the same cumulative milestone. | Runtime-specific |
| Right-size KV cache + batch | Gives the engine an explicit approximately 8 GiB KV-cache allocation and raises the request batch to 256, avoiding an oversized default allocation for a one-token completion workload. | Runtime-specific |
| Batch + overlap HF tokenization | Uses `encode_batch` and prepares the next request batch while the engine processes the current one. | Model-specific |
| Specialized router top-k | Replaces the general router path with the same fixed 384-expert, top-8 Triton selection used by the Torch path and omits training-only routing work. | Model-specific |
| FlashAttention 3 | Moves long-prefill attention from FlashAttention 2 to the Hopper-optimized FlashAttention 3 path. | Runtime-specific |
| Throughput peak | Records the best local-batch/KV-cache boundary before the later scheduler and transfer work: batch 240 with an explicitly sized cache. | Measurement |
| Direct sconv + explicit MoE tile | Installs the fixed-length direct short-convolution kernel and explicitly sets the fused-MoE block sizes, warp count, and pipeline stages for this architecture. | Runtime-specific |
| Long-repeat confirmation | Repeats the direct-sconv/MoE-tile candidate for longer before further engine changes. | Measurement |
| Chunked prefill + direct sconv | Enables vLLM chunked prefill while retaining the specialized direct short convolution, allowing the scheduler to divide the large 8,192-token requests into bounded work units. | Runtime-specific |
| Limit scheduler to 262k tokens | Caps one scheduling step at 262,144 input tokens, preventing a very large queued request batch from becoming one oversized device step. | Runtime-specific |
| Increase request batch | Grows the queued request set from 240 through 264, 416, 512, and 768 while the scheduler admits only the work allowed by its token budget. | Framework |
| Transfer BF16 logits | Tests half-width device-to-host transfer for the dense output. It was an accepted plotted milestone, but the final path returns FP32 and supersedes it. | Model-specific |
| Tune scheduler to 196k tokens | Tests a smaller 196,608-token scheduling budget with a 512-request queue. | Runtime-specific |
| 245k-token scheduler (short run) | Raises the scheduler budget to 245,760 tokens at queue size 768; this became the selected final budget. | Runtime-specific |
| Overlap output transfers | Copies dense prompt logits into pinned host memory asynchronously so transfer can overlap later GPU work. | Runtime-specific |
| Increase request queue | Expands the caller queue from 768 to 1,536 without changing the active scheduler limit, keeping the engine fed. | Framework |
| Reach request-queue plateau | Expands the queue to 2,048; the small gain confirms that the fixed active-request and token budgets are nearly saturated. | Measurement |
| Reuse pinned output buffer | Reuses one preallocated pinned host array for prompt logits instead of allocating a destination for every capture; the engine keeps at most 64 requests active. | Runtime-specific |
| Long-run confirmation | Median of three fresh, longer processes with queue 2,048, 64 active requests, and the 245,760-token budget; this supplies the 1,240,392 tok/s headline. | Measurement |

## JAX

| Plot milestone | What changed | Implementation surface |
|---|---|---|
| Full-vocabulary head | Deliberately naive baseline: XLA computes the 128,256-way output projection before selecting the A/C/G/T columns. | Measurement |
| A/C/G/T output head | Selects the four output-projection columns before the GEMM, removing almost all full-vocabulary compute and producing the largest single JAX gain. | Framework |
| Increase batch / Retune batch | Sweeps fixed compiled batches as the graph changes: 1→2→4→8→16→32→64→256, then 128→224, and later 128→352. | Framework |
| Overlap HF tokenization | Prefetches the next Hugging Face tokenized batch while the current JAX executable runs. This cumulative milestone also uses compact ACGTN embedding/head parameters; the measured gain came mainly from the pipeline once the head was already narrow. | Model-specific |
| Specialized router top-k | Uses an inference top-k specialized to the fixed 384-expert, top-8 router while preserving routing decisions required by the correctness gate. | Model-specific |
| Inference-only routing | Removes training-only router statistics and related bookkeeping from the compiled forward. | Model-specific |
| Hopper FA4 64×128 tile | Uses the CuTe FlashAttention 4 Hopper kernel with a 64×128 forward tile and 128 threads, selected for this sequence/head geometry. | Runtime-specific |
| Combine shared-expert GEMMs | Rewrites the two fixed shared experts into combined input and output projections. This was an accepted milestone but was later superseded by the faster GEMM-fusion graph. | Model-specific |
| GEMM fusion v2 (short) | Enables XLA's experimental GEMM-fusion-v2 pass; the winning graph leaves shared experts separate and lets the compiler choose the fusion. | Runtime-specific |
| Long-run confirmation | Median of three fresh, longer processes using batch 352, Hopper FA4, Sonic MoE, and GEMM fusion v2; this supplies the 1,146,750 tok/s headline. | Measurement |
