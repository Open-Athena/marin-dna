# Tested inference environment

The examples require this project plus the selected attention provider.
The reference environment uses Python 3.12.13, Torch 2.14.1+cu132, CUDA 13.2, Triton 3.8.0, NumPy 2.3.5, Safetensors 0.6.2, Tokenizers 0.23.0rc0, and Hugging Face Hub 1.33.0 on an H100 80GB.
Install the project independently from the training environment:

```bash
uv sync --locked --extra cuda --group dev
```

## FlashAttention providers

The tested FA2 2.8.4 and FA3 3.0.1 builds come from upstream revision `7c53542eab6a1c4fe760d1101b8d57d6307d1dbe`.
The exact executed build recipe and provider provenance are archived alongside the benchmark artifacts under `.agents/artifacts/exp586-deployment/`.
With a working CUDA 13.2 compiler toolkit, the corresponding source installation is:

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
git -C flash-attention checkout 7c53542eab6a1c4fe760d1101b8d57d6307d1dbe
uv pip install packaging setuptools wheel ninja einops psutil
export MAX_JOBS=1 NVCC_THREADS=1 FLASH_ATTN_CUDA_ARCHS=90 FLASH_ATTENTION_FORCE_BUILD=TRUE
uv pip install --no-build-isolation --no-deps ./flash-attention
export FLASH_ATTENTION_DISABLE_BACKWARD=TRUE FLASH_ATTENTION_DISABLE_FP16=TRUE FLASH_ATTENTION_DISABLE_FP8=TRUE
export FLASH_ATTENTION_DISABLE_HDIM64=TRUE FLASH_ATTENTION_DISABLE_HDIM96=TRUE FLASH_ATTENTION_DISABLE_HDIM192=TRUE FLASH_ATTENTION_DISABLE_HDIM256=TRUE
export FLASH_ATTENTION_DISABLE_HDIMDIFF64=TRUE FLASH_ATTENTION_DISABLE_HDIMDIFF192=TRUE FLASH_ATTENTION_DISABLE_SM80=TRUE
export FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE FLASH_ATTENTION_DISABLE_APPENDKV=TRUE FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE
uv pip install --no-build-isolation --no-deps ./flash-attention/hopper
```

FA3 imports as `flash_attn_interface`; its extension is `flash_attn_3._C`.
The restricted binary supports the tested Hopper/BF16/head-128 workload, including causal local/global attention, variable-length attention and GQA.
These build switches omit unused features; the inference code does not set kernel tiles.
The modular NVIDIA toolkit used by the executed recipe required an unversioned `libcudart.so` link to its `libcudart.so.13` for linking.
FP16 model weights are an API option, but this particular restricted FA3 binary only supports BF16.

## vLLM providers

The tested dependency is the Marin vLLM build at `ac3b5ea5e851`, not a verified stock PyPI release:

```bash
uv pip install 'https://github.com/marin-community/vllm/releases/download/marin-vllm-gpu-20261013-ac3b5ea5e851/vllm-0.0.0.dev20261013%2Bmarin.ac3b5ea5e851.cu132-cp38-abi3-manylinux_2_28_x86_64.whl'
uv pip install --no-deps --reinstall --extra-index-url https://download.pytorch.org/whl/cu132 'torchaudio==2.11.1+cu132'
```

The model calls its attention and BF16 fused-expert library functions directly, with no serving engine, quantization override or model-specific tile override.
Selecting an unavailable provider raises rather than falling back.
Fast additionally compiles the repository's custom Triton router kernel.

## Full benchmark reproduction

Inputs are `benchmark-fixture-512.json`, `benchmark-fixture-32.json`, `jax-reference32-latest.reference.npz`, and the pinned Hugging Face model snapshot.
Their hashes and exact native storage owner are recorded in the benchmark artifacts.
The lightweight examples use random data; the benchmark below validates the unchanged 32-window native-JAX gate before timing all 512 real windows equally.

```bash
env -u TORCHINDUCTOR_FREEZING CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
  python -m exp586_deployment.benchmark \
  --model-dir /path/to/hf-bf16 \
  --fixture /path/to/benchmark-fixture-512.json \
  --validation-fixture /path/to/benchmark-fixture-32.json \
  --reference /path/to/jax-reference32-latest.reference.npz \
  --output torch-fa2.json --name torch-fa2 --source-revision "$(git rev-parse HEAD)" \
  --batch-size 64 --iterations 32 --warmup 8 --fa-version 2 --require-no-vllm
```

Repeat in three fresh processes with distinct output names and aggregate the median of process medians.
Use `--fa-version 3` for torch-fa3; the vLLM variants add `--use-vllm` and omit `--require-no-vllm`.
Fast adds `--model-class fast --use-vllm --fa-version 3 --compile-mode reduce-overhead --dynamic static --freezing`.
Every case keeps `fullgraph=True`; compiler or parity failures stop the run.
Add `--input-source random --random-seed 0` to measure the examples' input distribution while retaining the real native-JAX validation gate.
Use separate compiler-cache directories per source/configuration; compilation is excluded from timing.
