#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
environment_dir="${1:-${project_dir}/.inference-venv}"
uv_command=(uvx uv@0.11.31)
vllm_wheel="https://github.com/marin-community/vllm/releases/download/marin-vllm-gpu-20261013-ac3b5ea5e851/vllm-0.0.0.dev20261013%2Bmarin.ac3b5ea5e851.cu132-cp38-abi3-manylinux_2_28_x86_64.whl"

cd "${project_dir}"
"${uv_command[@]}" venv "${environment_dir}" --python 3.12
"${uv_command[@]}" pip install \
  --python "${environment_dir}/bin/python" \
  --prerelease allow \
  -e "${project_dir}"
"${uv_command[@]}" pip install \
  --python "${environment_dir}/bin/python" \
  --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu132 \
  "torch==2.14.1+cu132" \
  "${vllm_wheel}"
# The released vLLM wheel pins the PyPI torchaudio 2.11.0 build, which was
# compiled for CUDA 13.0 and crashes when Transformers imports it beside the
# wheel's CUDA 13.2 Torch. Replace it with PyTorch's ABI-compatible cu132 build;
# neither this text-only model nor vLLM's execution path uses TorchAudio.
"${uv_command[@]}" pip install \
  --python "${environment_dir}/bin/python" \
  --no-deps \
  --reinstall \
  --extra-index-url https://download.pytorch.org/whl/cu132 \
  "torchaudio==2.11.1+cu132"

"${environment_dir}/bin/python" - <<'PY'
import torch
import vllm

print(f"torch={torch.__version__}")
print(f"vllm={vllm.__version__}")
PY
