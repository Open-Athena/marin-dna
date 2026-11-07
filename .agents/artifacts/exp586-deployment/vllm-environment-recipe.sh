set -euo pipefail
uv venv --python /app/inference-venv/bin/python /app/deploy002/vllm
uv pip install --python /app/deploy002/vllm/bin/python --prerelease allow --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu132 'torch==2.14.1+cu132' 'numpy==2.3.5' 'safetensors==0.6.2' 'tokenizers==0.23.0rc0' 'huggingface-hub==1.33.0' 'einops==0.8.1' 'https://github.com/marin-community/vllm/releases/download/marin-vllm-gpu-20261013-ac3b5ea5e851/vllm-0.0.0.dev20261013%2Bmarin.ac3b5ea5e851.cu132-cp38-abi3-manylinux_2_28_x86_64.whl'
uv pip install --python /app/deploy002/vllm/bin/python --no-deps --reinstall --extra-index-url https://download.pytorch.org/whl/cu132 'torchaudio==2.11.1+cu132'
/app/deploy002/vllm/bin/python - <<'CHECK'
import torch,vllm,torchaudio
from vllm.vllm_flash_attn import flash_attn_varlen_func
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
print('VLLM_ENV_READY',torch.__version__,vllm.__version__,flush=True)
CHECK
