# Issue 430 inference investigation

The benchmark consumes the pinned, pre-materialized `marin-dna/evals_mendelian_traits_harness_255` dataset. It does not read a reference FASTA during scoring.

Each inference stack has an independent minimal uv project under `environments/`. Local MarinDNA scoring code is imported through `PYTHONPATH`; installing the repository package (and its unrelated pipeline, plotting, and training dependencies) is intentionally unnecessary.

## Hugging Face / TorchAO

The `hf_torchao` environment covers eager and compiled BF16 plus TorchAO PTQ. `mslk-cuda` is installed only on x86_64, where the INT4 kernel is available.

The LLR benchmark exposes three exact execution layouts:

- `prefix-cache` forwards the shared prefix once, duplicates its dynamic KV cache, and forwards the REF/ALT suffixes together.
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

## vLLM

The `vllm` environment is x86_64-only and pins vLLM 0.18.0 with PyTorch 2.10 / CUDA 12.8 compatibility. During the H100 probe, vLLM 0.26.0 selected PyTorch 2.11 / CUDA 13, which was incompatible with the host's CUDA 12.8 driver.

```bash
uv sync --project scripts/issue430/environments/vllm --frozen
PYTHONPATH="$PWD/src" uv run --project scripts/issue430/environments/vllm --no-sync \
  python scripts/issue430/benchmark_vllm.py --help
```
