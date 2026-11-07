set -euo pipefail
mkdir -p /app/deploy002
uv venv --python /app/inference-venv/bin/python /app/deploy002/no-vllm
uv pip install --python /app/deploy002/no-vllm/bin/python --prerelease allow --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu132 'torch==2.14.1+cu132' 'numpy==2.3.5' 'safetensors==0.6.2' 'tokenizers==0.23.0rc0' 'huggingface-hub==1.33.0' packaging setuptools wheel ninja einops psutil
uv pip install --target /app/deploy002/toolchain --no-deps 'nvidia-cuda-nvcc==13.2.86' 'nvidia-cuda-crt==13.2.86' 'nvidia-cuda-runtime==13.2.86' 'nvidia-cuda-cccl==13.2.86' 'nvidia-nvvm==13.2.86'
/app/deploy002/no-vllm/bin/python - <<'CONFIG'
from pathlib import Path
import json,subprocess
root=Path('/app/deploy002/toolchain')
nvcc=next(p for p in root.rglob('nvcc') if p.is_file())
home=nvcc.parent.parent
assert (home/'include/cuda.h').exists(),list(home.iterdir())
assert (home/'include/cuda_runtime.h').exists()
Path('/app/deploy002/cuda-home').write_text(str(home))
print('CUDA_HOME',home,subprocess.check_output([str(nvcc),'--version'],text=True),flush=True)
import torch,importlib.util
assert torch.__version__=='2.14.1+cu132'
assert importlib.util.find_spec('vllm') is None
print('ENVIRONMENT_READY',torch.__version__,flush=True)
CONFIG
/app/deploy002/no-vllm/bin/python - <<'CUDART'
from pathlib import Path
lib=Path(Path('/app/deploy002/cuda-home').read_text().strip())/'lib'
assert (lib/'libcudart.so.13').is_file()
link=lib/'libcudart.so'
if not link.exists():link.symlink_to('libcudart.so.13')
assert link.samefile(lib/'libcudart.so.13')
CUDART
export CUDA_HOME="$(cat /app/deploy002/cuda-home)"
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=1 NVCC_THREADS=1 FLASH_ATTN_CUDA_ARCHS=90 FLASH_ATTENTION_FORCE_BUILD=TRUE
unset TORCHINDUCTOR_FREEZING
if [ ! -d /app/deploy002/flash-attention/.git ]; then
 git clone --filter=blob:none --no-checkout https://github.com/Dao-AILab/flash-attention.git /app/deploy002/flash-attention
fi
git -C /app/deploy002/flash-attention checkout 7c53542eab6a1c4fe760d1101b8d57d6307d1dbe
uv pip install --python /app/deploy002/no-vllm/bin/python --no-build-isolation --no-deps /app/deploy002/flash-attention
printf 'FA2_BUILD_DONE\n'
export FLASH_ATTENTION_DISABLE_BACKWARD=TRUE FLASH_ATTENTION_DISABLE_FP16=TRUE FLASH_ATTENTION_DISABLE_FP8=TRUE FLASH_ATTENTION_DISABLE_HDIM64=TRUE FLASH_ATTENTION_DISABLE_HDIM96=TRUE FLASH_ATTENTION_DISABLE_HDIM192=TRUE FLASH_ATTENTION_DISABLE_HDIM256=TRUE FLASH_ATTENTION_DISABLE_HDIMDIFF64=TRUE FLASH_ATTENTION_DISABLE_HDIMDIFF192=TRUE FLASH_ATTENTION_DISABLE_SM80=TRUE FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE FLASH_ATTENTION_DISABLE_APPENDKV=TRUE FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE
uv pip install --python /app/deploy002/no-vllm/bin/python --no-build-isolation --no-deps /app/deploy002/flash-attention/hopper
/app/deploy002/no-vllm/bin/python - <<'CHECK'
import flash_attn
import flash_attn_interface
import importlib.util
assert importlib.util.find_spec('vllm') is None
print('STANDALONE_READY',flash_attn.__version__,flash_attn_interface.__file__,flush=True)
CHECK
printf 'DEPLOY002_BOOTSTRAP_DONE\n'
