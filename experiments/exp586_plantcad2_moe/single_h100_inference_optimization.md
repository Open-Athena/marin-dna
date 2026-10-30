# Single-H100 inference optimization

This document records the exp586 inference study for `exp586-plantcad2-d768-pretrained-lrm1-seed0-v1`. The model has 1,591,848,448 total parameters and 60,555,264 active parameters per token. The study is an experimental estimate of the throughput available from this architecture on one H100, not a deployment recommendation. PR #590 is intentionally left unmerged so future work can choose a smaller, maintainable subset of the optimizations.

## Workload and measurement

Every throughput trial scores unique 8,192-base ACGTN windows. Each input character is one token, and every completed sequence returns an FP32 NumPy array shaped `[8192, 4]` containing A/C/G/T logits at every position. There is no sequence packing, shared-prefix reuse, tensor parallelism, expert parallelism, or multi-GPU execution.

CoreWeave Iris can allocate H100s only as complete eight-GPU nodes for this workload. Every job therefore requested `H100x8` at `--priority batch --user eczech`, then exposed only GPU 0 to the benchmark with `CUDA_VISIBLE_DEVICES=0`. The other seven GPUs were not used by the process.

The headline rate is a steady-state pipeline measurement after model loading, compilation, and one warmup batch. A timed batch includes:

1. Hugging Face tokenization, overlapped with the preceding device batch when enabled;
2. host-to-device input transfer;
3. the complete model forward pass on one H100;
4. materialization of all A/C/G/T logits as FP32 host arrays.

It excludes model download, model loading, compilation, warmup, reading sequence files, and writing output files. Throughput is batch size divided by the median interval between adjacent completed batches, multiplied by 8,192 input tokens per sequence. This removes one-time pipeline fill while retaining the sustained tokenizer, device, and host-output costs. It is more representative than a kernel-only forward benchmark, but it is not the wall-clock time of an entire command-line program.

vLLM needs `max_tokens=1` to execute a normal prefill request. The capture hook retains the 8,192 prompt positions, and the timed engine call includes vLLM's scheduler and the small one-token request completion overhead. This is still a prompt-logit workload rather than a decode-throughput benchmark.

The discovery ledgers contain 544 completed trials over 10.59 productive H100-hours: 187 PyTorch trials, 147 JAX trials, and 210 vLLM trials. Productive time counts benchmark and profiling work while an accelerator was available; it excludes queueing, environment installation, and other downtime. The final fresh-process confirmations were run afterward.

## Correctness contract

Every accepted candidate is checked on the same 32 unique, real PlantCAD windows against a fixed native-JAX reference. Both arrays have shape `[32, 8192, 4]`. Softmax is restricted to A/C/G/T, next-base labels are shifted by one position, and positions whose target is `N` are excluded from loss and accuracy.

A candidate must satisfy all four limits:

| Metric | Limit |
|---|---:|
| Mean absolute probability error | ≤ 0.003 |
| 99th-percentile absolute probability error | ≤ 0.03 |
| Absolute pooled next-base NLL delta | ≤ 0.001 |
| Absolute pooled next-base accuracy delta | ≤ 0.001 |

Raw-logit error and argmax agreement are diagnostic only. This allows numerically different kernel and compiler paths when they preserve the probabilities and task-level behavior used by PlantCAD evaluation.

## Confirmed result

The final table and replication ranges are generated from `results/inference_optimization/final-confirmation-summary.json`. Each headline is the median of three fresh processes using the exact leading configuration. PyTorch and JAX use 64 timed batches per process; vLLM uses 16 large queued-request batches per process. All nine runs passed the correctness gate.

| Backend | Confirmed median | Sequences/s | Naive baseline | Speedup | Three-process range |
|---|---:|---:|---:|---:|---:|
| PyTorch | **1,584,801 tok/s** | 193.46 | 18,051 tok/s | **87.8×** | 1,584,032–1,585,404 tok/s |
| vLLM | **1,240,392 tok/s** | 151.42 | 245,367 tok/s | **5.1×** | 1,231,782–1,240,629 tok/s |
| JAX | **1,146,750 tok/s** | 139.98 | 7,677 tok/s | **149.4×** | 1,146,699–1,147,459 tok/s |

The baseline ratios describe progress within each backend, not the intrinsic efficiency of the frameworks: the vLLM baseline already used a fused serving engine, while the Torch and JAX baselines deliberately began with naive full-vocabulary paths.

## What mattered

### Restricting the output

The workload consumes four logits per position, while the pretrained model has a 128,256-token language vocabulary. Avoiding the full output projection was essential: it moved JAX from roughly 7.7k to 391.6k tok/s in one step and also improved the other backends. Physically shrinking the five input embedding rows and four output columns did not provide a stable additional gain in Torch or JAX after the narrow projection was already in place. The benefit comes from avoiding the full-vocabulary computation and transfer, not from making the stored checkpoint smaller.

### Tokenization and batching

The original Hugging Face BPE model is inappropriate for this restricted workload unless it is forced to emit one token per base. The optimized tokenizers retain the Hugging Face `tokenizers` runtime and lowercase normalizer while replacing the segmentation model with an ACGTN character unigram. Batched encoding and a one-thread prefetch pipeline hide most tokenizer time behind GPU work. This was one of the largest improvements in all three paths; in JAX it raised the frontier from about 582k to 958k tok/s.

Batch size was retuned after every material graph change rather than inherited from an earlier path. The largest batch that fit was not always the fastest. vLLM additionally separates the caller's request queue from the number of requests active in the scheduler.

### Model-specific kernels and graph changes

- **Router top-k specialization:** the general training router sorted and maintained statistics that inference does not need. The specialized path computes stable top-k expert assignments for the fixed 384-expert, top-8 configuration and omits training-only statistics. It improved all three backends but requires model-specific code.
- **MoE execution:** Torch gained most from replacing an eager expert loop with grouped/fused MoE kernels. JAX's existing Sonic MoE path was already strong; alternate Sonic tile sizes, scatter MoE, and a CUTLASS-DSL attempt did not improve it. vLLM started with a good fused Triton MoE and gained little from additional tile sweeps.
- **Short convolution:** a direct inference-only implementation removed generic bookkeeping and gave Torch another large gain. Equivalent Triton/tile experiments did not improve JAX, and vLLM's direct implementation only helped after interactions with the rest of its engine were controlled.
- **Shared experts:** combining gate and up projections, and later combining both shared experts, provided small reproducible Torch gains. The analogous JAX graph helped, but the final XLA GEMM-fusion configuration was slightly faster without that rewrite.
- **Attention and position handling:** FlashAttention 3 with native GQA was the best Torch and vLLM choice. JAX used the Hopper FA4 CuTe kernel with a 64×128 forward tile and 128 threads. Precomputing rotary tables helped Torch.
- **Compilation:** `torch.compile(..., fullgraph=True, dynamic=False, mode="reduce-overhead")` plus `TORCHINDUCTOR_FREEZING=1` gave the best Torch graph. JAX's final small gain came from `--xla_gpu_experimental_gemm_fusion_v2=true`. These compiler settings are less portable than the output/tokenizer changes.

## Selected backend paths

### PyTorch

The final path uses a compact ACGTN input embedding and four-column output head, the batched character tokenizer, one-batch tokenization prefetch, compiled grouped MoE, FlashAttention 3, native GQA, specialized router top-k, direct short convolution, precomputed rotary tables, combined shared-expert projections, graph freezing, and batch 240. Dense FP32 host logits are part of the timed path.

The biggest gains were grouped MoE execution, tokenization overlap, global FlashAttention, and the direct short convolution. The final graph/compiler changes were smaller but reproducible. FP8 MoE, explicit CUDA graph capture, alternative `torch.compile` modes, output BF16, custom MoE tile sweeps, and physical vocabulary compaction by itself were neutral, slower, or failed correctness.

The runnable interface is [`src/exp586_moe/inference/optimized_torch.py`](src/exp586_moe/inference/optimized_torch.py), with [`examples/optimized_logits.py`](examples/optimized_logits.py) as the command-line example.

### JAX

The final path directly restores the original checkpoint into the Hero JAX architecture, remaps the input embedding to ACGTN and the output head to A/C/G/T, uses batched character tokenization with prefetch, Sonic MoE, inference-only router top-k, Hopper FA4 with the selected tile, batch 352, a 95% XLA memory pool, and XLA GEMM fusion v2.

JAX's apparent vocabulary-pruning gain is almost entirely the four-column output projection. Physical five-row/four-column parameter remapping was neutral after that. The tested scan-unroll settings, rematerialization changes, input donation, output BF16, command-buffer modes, PGLE, latency-hiding flags, Triton warp specialization, Sonic tile sweeps, short-convolution tile sweeps, alternate FA4 tiles, and scatter MoE did not beat the final path.

The runnable interface is [`src/exp586_moe/inference/optimized_jax.py`](src/exp586_moe/inference/optimized_jax.py), with [`examples/optimized_jax_logits.py`](examples/optimized_jax_logits.py) as the command-line example.

### vLLM

vLLM began much faster than the naive framework baselines because it already supplies compiled attention, fused MoE, memory management, and a high-throughput request engine. Its final ceiling was lower because the workload is one long, unique prefill per request with no generation and almost no reusable prefixes: the scheduler and worker/RPC abstractions remain in the hot path without providing their usual serving benefits.

The selected configuration uses the A/C/G/T capture head, reduced character tokenizer, batched tokenization and prefetch, compiled engine, fused Triton MoE, specialized top-k, FlashAttention 3, direct short convolution, chunked prefill, a 245,760-token scheduler budget, a queue of 2,048 requests, 64 active requests, an explicitly sized KV cache, and one reused pinned output buffer. The largest late gain came from overlapping dense host-output transfer with later GPU work.

Alternative FlashInfer MoE backends did not initialize for this model. In-process execution, OMP tuning, CUDA graph modes, BF16 output transfer, alternate attention backends, MoE tile sweeps, smaller/larger scheduler budgets, and separate/deferred copy variants were neutral or slower. Very large request queues plateaued once the active scheduler and token budget were saturated.

The runnable interface is [`src/exp586_moe/inference/optimized_vllm.py`](src/exp586_moe/inference/optimized_vllm.py), with [`examples/optimized_vllm_logits.py`](examples/optimized_vllm_logits.py) as the command-line example.

## Portability and deployment cost

The optimizations fall into three practical groups:

| Group | Examples | Deployment cost |
|---|---|---|
| Framework-compatible | A/C/G/T projection, character tokenizer, larger batches, tokenization prefetch, standard FlashAttention selection | Low to moderate; these fit normal application code and public framework APIs. |
| Model-specific code | Specialized router top-k, direct short convolution, combined shared-expert projections, compact embedding/head remapping | Moderate to high; these must be maintained against the exact architecture and validated on each new checkpoint. |
| Runtime-specific hooks | Torch graph freezing, experimental XLA GEMM fusion, vLLM prompt-logit capture and pinned-buffer reuse, scheduler budget/queue tuning | High; these depend on compiler internals or custom engine hooks and are the least likely to survive framework upgrades unchanged. |

The experiment demonstrates an upper bound. A production implementation should start with the first group, measure how much throughput remains, and add custom code only when the operational cost is justified.

## Reproducing on an identical model

1. Create the backend environments with [`tools/create_inference_env.sh`](tools/create_inference_env.sh). Torch and vLLM share one environment; JAX uses the `jax` argument and a separate environment.
2. Export the final checkpoint to the validated BF16 Hugging Face form for Torch/vLLM. JAX restores the original checkpoint. Preserve the architecture, tokenizer IDs, and weight hashes.
3. Materialize unique 8,192-base fixtures with [`fixtures.py`](src/exp586_moe/inference/fixtures.py). Use the fixed 32-sequence reference fixture for every correctness decision, regardless of throughput batch size.
4. Run [`benchmark_torch.py`](src/exp586_moe/inference/benchmark_torch.py), [`benchmark_jax.py`](src/exp586_moe/inference/benchmark_jax.py), or [`benchmark_vllm.py`](src/exp586_moe/inference/benchmark_vllm.py) with the selected settings described above. Expose only one GPU with `CUDA_VISIBLE_DEVICES=0`.
5. Repeat the final configuration in fresh processes. Use the median run-level throughput as the headline and keep the range; do not select a single favorable short run.

The exact Python package set is encoded in [`tools/create_inference_env.sh`](tools/create_inference_env.sh). The three simple interfaces are the clearest executable specifications of the selected graphs. The benchmark scripts retain all timing and correctness instrumentation used here.

<details>
<summary>Exact selected benchmark commands</summary>

The commands below assume `ROOT` contains `hf-bf16/`, the 512- and 8,192-sequence throughput fixtures, the 32-sequence validation fixture, and `jax-reference32-latest.reference.npz`. `RESULTS` must already exist.

```bash
export ROOT=/path/to/inference-artifacts
export RESULTS=/path/to/results
export PYTHONPATH=$PWD/experiments/exp586_plantcad2_moe/src
export CUDA_VISIBLE_DEVICES=0
```

PyTorch:

```bash
TORCHINDUCTOR_FREEZING=1 .inference-venv/bin/python -m exp586_moe.inference.benchmark_torch \
  --model_dir "$ROOT/hf-bf16" \
  --fixture "$ROOT/benchmark-fixture-512.json" \
  --validation_fixture "$ROOT/benchmark-fixture-32.json" \
  --reference "$ROOT/jax-reference32-latest.reference.npz" \
  --ledger "$RESULTS/torch.jsonl" --output "$RESULTS/torch-final.json" \
  --name Torch-final --variant acgt_vllm_moe \
  --compile_model True --compile_mode reduce-overhead \
  --attention_backend vllm_fa3 --global_attention_backend flash \
  --flash_attention_version 3 --batch_size 240 --iterations 64 \
  --prefetch_tokenization True --specialized_topk True \
  --sconv_backend direct --tokenizer_batch True --precomputed_rope True \
  --combined_shared_expert_projections True \
  --tokenizer_variant compact_unigram --compact_output_head True
```

JAX:

```bash
XLA_CLIENT_MEM_FRACTION=.95 \
XLA_PYTHON_CLIENT_PREALLOCATE=true \
XLA_FLAGS=--xla_gpu_experimental_gemm_fusion_v2=true \
  .jax-inference-venv/bin/python -m exp586_moe.inference.benchmark_jax \
  --fixture "$ROOT/benchmark-fixture-512.json" \
  --validation_fixture "$ROOT/benchmark-fixture-32.json" \
  --reference "$ROOT/jax-reference32-latest.reference.npz" \
  --ledger "$RESULTS/jax.jsonl" --output "$RESULTS/jax-final.json" \
  --name JAX-final --variant acgt_head --tokenizer_variant reduced_unigram \
  --batch_size 352 --iterations 64 --compact_vocab True \
  --moe_implementation sonic --attention_implementation gpu_fa4_cute \
  --output_dtype float32 --prefetch_tokenization True \
  --layer_scan_unroll 1 --remat_layers True \
  --fa4_forward_tile_m 64 --fa4_forward_tile_n 128 --fa4_num_threads 128
```

vLLM:

```bash
.inference-venv/bin/python -m exp586_moe.inference.benchmark_vllm \
  --model-dir "$ROOT/hf-bf16" \
  --fixture "$ROOT/benchmark-fixture-8192.json" \
  --validation-fixture "$ROOT/benchmark-fixture-32.json" \
  --reference "$ROOT/jax-reference32-latest.reference.npz" \
  --ledger "$RESULTS/vllm.jsonl" --output "$RESULTS/vllm-final.json" \
  --name vLLM-final --variant acgt_head --tokenizer-variant reduced_unigram \
  --compile-engine --gpu-memory-utilization 0.2 \
  --kv-cache-memory-bytes 8267812045 --moe-backend triton \
  --attention-backend FLASH_ATTN --flash-attention-version 3 \
  --batch-size 2048 --iterations 16 --tokenizer-batch --prefetch-tokenization \
  --specialized-topk --direct-sconv --chunked-prefill \
  --max-num-batched-tokens 245760 --max-num-seqs 64 \
  --preallocated-output-transfer
```

</details>

## Evidence

- [PyTorch raw trial ledger](results/inference_optimization/torch.jsonl)
- [JAX raw trial ledger](results/inference_optimization/jax.jsonl)
- [vLLM raw trial ledger](results/inference_optimization/vllm.jsonl)
- [Final confirmation summary](results/inference_optimization/final-confirmation-summary.json)
- [Progress-figure renderer](src/exp586_moe/inference/plot_progress.py)
- [Correctness implementation](src/exp586_moe/inference/correctness.py)
- [Immutable workload contract](src/exp586_moe/inference/contract.py)

The ledgers are the detailed record of every completed candidate, including slower accepted trials and candidates rejected by the correctness gate. Startup failures and out-of-memory boundaries that produced no valid measurement are summarized above rather than represented as zero-throughput results.
