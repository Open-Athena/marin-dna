# Issue 430 inference investigation

The benchmark consumes the pinned, pre-materialized `marin-dna/evals_mendelian_traits_harness_255` dataset. It does not read a reference FASTA during scoring.

Each inference stack has an independent minimal uv project under `environments/`. Local MarinDNA scoring code is imported through `PYTHONPATH`; installing the repository package (and its unrelated pipeline, plotting, and training dependencies) is intentionally unnecessary.

## Hugging Face / TorchAO

The `hf_torchao` environment covers eager and compiled BF16 plus TorchAO PTQ. `mslk-cuda` is installed only on x86_64, where the INT4 kernel is available.

The LLR benchmark exposes four exact execution layouts:

- `prefix-cache` forwards the shared prefix once, duplicates its dynamic KV cache, and forwards the REF/ALT suffixes together.
- `sequential-branches` forwards the shared prefix once, keeps one read-only KV cache, and forwards REF then ALT separately. Its lower peak memory is intended to support roughly twice the variant batch size without changing the score contract.
- `branch-packed` forwards `prefix + ref_suffix + alt_suffix` once with duplicated suffix position IDs and an isolation mask, removing dynamic-cache allocation while retaining shared-prefix compute.
- `full-pair` forwards complete REF and ALT sequences together. It recomputes the prefix but is a useful single-call, regular-causal performance control.

The `fp8-rowwise` quantization choice is distinct from the original `fp8-dynamic` per-tensor recipe. It uses TorchAO's rowwise activation and weight scaling and should be screened at the production batch size, where quantization overhead can be amortized.

```bash
uv sync --project scripts/issue430/environments/hf_torchao --frozen
PYTHONPATH="$PWD/src" uv run --project scripts/issue430/environments/hf_torchao --no-sync \
  python scripts/issue430/benchmark_llr.py \
  --execution-layout branch-packed \
  --torch-compile --compile-mode default \
  --out-dir scratch/issue430/branch-packed-compile
```

## Transformer Engine

The `transformer_engine` environment is intentionally separate from the Hugging Face/TorchAO project. It runs inside the pinned NVIDIA PyTorch `25.03-py3` container, which supplies a mutually compatible PyTorch, CUDA 12.8, cuDNN, and Transformer Engine build; its own pyproject installs only the pinned Hugging Face and analysis dependencies into a venv that can see those container packages.

The `transformer_engine_modern` environment instead installs PyTorch 2.10 from the CUDA 12.8 wheel index and Transformer Engine 2.13, the newest release whose PyTorch integration and aarch64 core wheel both support CUDA 12. The stage installs the wheel-backed core first and then builds only `transformer-engine-torch` without build isolation, as required by NVIDIA's installation guide.

`te-bf16` replaces the same 133 non-LM-head linear layers with Transformer Engine `Linear` modules but disables FP8, providing a same-container/backend control. `te-fp8-delayed` enables E4M3 delayed scaling with a fixed-size amax history. Delayed scaling chooses each activation scale from historical maxima, avoiding the separate current-amax tensor read performed by dynamic scaling. Add `--te-fp8-model-init` to construct the replaced layers with FP8-only parameter storage, Transformer Engine's experimental inference-oriented pre-materialization path. The LM head remains BF16 because the seven-token output dimension is not an FP8 GEMM shape.

Add `--te-fused-mlp` to replace each Qwen3 post-attention RMSNorm plus gate/up/down MLP with one Transformer Engine `LayerNormMLP`. Add `--te-fused-qkv` to concatenate each layer's Q/K/V weights into one Transformer Engine projection while retaining Hugging Face attention, RoPE, and KV-cache behavior. Both flags record pre/post-conversion LLR parity before timing and are intentionally incompatible with the separate experimental `--te-fp8-model-init` combination.

The CUDA wheel libraries must precede the base NVIDIA container's cuDNN and cuBLAS libraries on fresh Lambda shells:

```bash
ISSUE430_TE_SITE_PACKAGES="$PWD/scripts/issue430/environments/transformer_engine_modern/.venv/lib/python3.12/site-packages"
export LD_LIBRARY_PATH="$ISSUE430_TE_SITE_PACKAGES/nvidia/cudnn/lib:$ISSUE430_TE_SITE_PACKAGES/nvidia/cublas/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$PWD/src"
```

The promoted LLR-only frontier command is:

```bash
scripts/issue430/environments/transformer_engine_modern/.venv/bin/python \
  scripts/issue430/benchmark_llr.py \
  --checkpoint "$HOME/ckpt" --subset all \
  --quantization te-fp8-delayed \
  --te-amax-history-len 1 --te-amax-compute-algo most_recent \
  --te-fused-mlp --te-fused-qkv \
  --torch-compile --compile-mode default --dynamo-recompile-limit 64 \
  --batching fused --execution-layout prefix-cache \
  --batch-size 512 --num-workers 4 --prefetch-factor 2 --repetitions 3 \
  --out-dir scratch/issue430/te-fp8-fused-mlp-qkv-b512-full
```

The terminal embeddings-on confirmation uses the same converted model with a separate output contract:

```bash
scripts/issue430/environments/transformer_engine_modern/.venv/bin/python \
  scripts/issue430/benchmark_embeddings.py \
  --checkpoint "$HOME/ckpt" --subset missense_variant \
  --quantization te-fp8-delayed \
  --te-amax-history-len 1 --te-amax-compute-algo most_recent \
  --te-fused-mlp --te-fused-qkv \
  --torch-compile --compile-mode default --dynamo-recompile-limit 64 \
  --batch-size 128 --num-workers 4 --prefetch-factor 2 --repetitions 3 \
  --out-dir scratch/issue430/te-fp8-fused-mlp-qkv-embeddings-b128-r3
```

For the terminal frozen-probe compatibility gate, export the complete aligned embedding bundle once using the same production f16 storage cast, then apply the canonical BF16 classifiers unchanged to both arms:

```bash
scripts/issue430/environments/transformer_engine_modern/.venv/bin/python \
  scripts/issue430/benchmark_embeddings.py \
  --checkpoint "$HOME/ckpt" --subset all --save-scores \
  --quantization te-fp8-delayed \
  --te-amax-history-len 1 --te-amax-compute-algo most_recent \
  --te-fused-mlp --te-fused-qkv \
  --torch-compile --compile-mode default --dynamo-recompile-limit 64 \
  --batch-size 128 --num-workers 4 --prefetch-factor 2 --repetitions 1 \
  --out-dir scratch/issue430/probe-fp8-embeddings

PYTHONPATH="$PWD/src" uv run python scripts/issue430/evaluate_probe_compatibility.py \
  --bf16-scores scratch/issue430/probe_compat/bf16_scores.parquet \
  --fp8-scores scratch/issue430/probe_compat/fp8_scores.parquet \
  --classifiers scratch/issue430/probe_compat/bf16_classifiers.joblib \
  --out-dir scratch/issue430/probe_compat/results
```

The published joblib contains the production all-data classifier for each consequence subset. This is therefore a frozen-classifier perturbation test: absolute AUPRC is in-sample, while the BF16-to-FP8 prediction and AUPRC deltas isolate whether quantized embeddings remain compatible without retraining.

## vLLM

The `vllm` environment is x86_64-only and pins vLLM 0.18.0 with PyTorch 2.10 / CUDA 12.8 compatibility. During the H100 probe, vLLM 0.26.0 selected PyTorch 2.11 / CUDA 13, which was incompatible with the host's CUDA 12.8 driver.

```bash
uv sync --project scripts/issue430/environments/vllm --frozen
PYTHONPATH="$PWD/src" uv run --project scripts/issue430/environments/vllm --no-sync \
  python scripts/issue430/benchmark_vllm.py --help
```
