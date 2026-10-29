#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
environment_dir="${1:-${project_dir}/.inference-venv}"
backend="${2:-torch-vllm}"
uv_command=(uvx uv@0.11.31)
vllm_wheel="https://github.com/marin-community/vllm/releases/download/marin-vllm-gpu-20261013-ac3b5ea5e851/vllm-0.0.0.dev20261013%2Bmarin.ac3b5ea5e851.cu132-cp38-abi3-manylinux_2_28_x86_64.whl"

if [[ "${backend}" != "jax" && "${backend}" != "torch-vllm" ]]; then
  echo "backend must be jax or torch-vllm" >&2
  exit 2
fi

cd "${project_dir}"
"${uv_command[@]}" venv "${environment_dir}" --python 3.12
"${uv_command[@]}" pip install \
  --python "${environment_dir}/bin/python" \
  --prerelease allow \
  -e "${project_dir}"
if [[ "${backend}" == "jax" ]]; then
  # Marin's current Sonic MoE and FA4 kernels require this tested JAX/Triton/
  # CuTe set. Install the JAX-only dependencies explicitly: the published GPU
  # extra also pulls Torch/NCCL constraints that conflict with each other and
  # are unused by native JAX inference.
  "${uv_command[@]}" pip install \
    --python "${environment_dir}/bin/python" \
    --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu132 \
    "jax[cuda13]==0.11.1" \
    "jax-triton==0.3.1" \
    "nvidia-cutlass-dsl[cu13]==4.6.2" \
    "triton>=3.6.0,<3.7"
  "${environment_dir}/bin/python" - <<'PY'
import jax
from exp586_moe.inference import benchmark_jax

print(f"jax={jax.__version__}")
print(f"devices={jax.devices()}")
print(f"benchmark={benchmark_jax.__name__}")
PY
else
  "${uv_command[@]}" pip install \
    --python "${environment_dir}/bin/python" \
    --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu132 \
    "torch==2.14.1+cu132" \
    "${vllm_wheel}"
  # The released vLLM wheel pins the PyPI torchaudio 2.11.0 build, which was
  # compiled for CUDA 13.0 and crashes when Transformers imports it beside the
  # wheel's CUDA 13.2 Torch. Replace it with PyTorch's ABI-compatible cu132
  # build; neither this text-only model nor vLLM uses TorchAudio.
  "${uv_command[@]}" pip install \
    --python "${environment_dir}/bin/python" \
    --no-deps \
    --reinstall \
    --extra-index-url https://download.pytorch.org/whl/cu132 \
    "torchaudio==2.11.1+cu132"
  "${environment_dir}/bin/python" - <<'PY'
import torch
import torchaudio
import vllm
from exp586_moe.inference import benchmark_torch, benchmark_vllm

print(f"torch={torch.__version__}")
print(f"torchaudio={torchaudio.__version__}")
print(f"vllm={vllm.__version__}")
print(f"benchmarks={benchmark_torch.__name__},{benchmark_vllm.__name__}")
PY
fi
