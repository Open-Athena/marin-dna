#!/bin/bash
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:$PATH
export UV_CONCURRENT_DOWNLOADS=4 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
trap 'status=$?; echo "$status" > /opt/issue550/prepare-exit.txt; if [ "$status" -ne 0 ]; then sudo shutdown -h now; fi' EXIT
: "${ISSUE550_SOURCE_COMMIT:?Pass the full reviewed source commit}"
test "${#ISSUE550_SOURCE_COMMIT}" -eq 40
curl -fLsS https://astral.sh/uv/0.11.31/install.sh | sh
git clone --filter=blob:none --no-checkout https://github.com/Open-Athena/marin-dna.git /opt/issue550/repo
git -C /opt/issue550/repo checkout "$ISSUE550_SOURCE_COMMIT"
cd /opt/issue550/repo/snakemake/analysis/evals_v2
uv sync --locked --group dev --group genome-s3
mkdir -p /opt/issue550/checkpoint /opt/issue550/batch-sweep
aws s3 cp --recursive --only-show-errors \
    s3://oa-bolinas/snakemake/analysis/evals_v2/results/checkpoints/dna-exp550-rag46m-five-regions-v1-step-20000/ \
    /opt/issue550/checkpoint/
CUDA_VISIBLE_DEVICES="" uv run --locked pytest > /opt/issue550/all-tests.log 2>&1
cat /opt/issue550/all-tests.log
uv run --locked python /opt/issue550/repo/.agents/artifacts/issue-550/evaluation/sweep-cached-bf16.py \
    --checkpoint /opt/issue550/checkpoint --batch 8 --output /opt/issue550/batch-sweep/batch-8.json
uv run --locked python - <<'PY'
import json
import subprocess
from pathlib import Path

root = Path('/opt/issue550/batch-sweep')
measurement = json.loads((root / 'batch-8.json').read_text())
assert measurement['batch'] == 8 and measurement['finite_outputs']
assert measurement['peak_allocated_bytes'] <= .90 * measurement['device_total_bytes']
report = {
    'synthetic_only': True,
    'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'selection_basis': 'Full A10G BF16 batch sweep at 10k; fresh batch-8 timing and finite-output check at 20k',
    'measurements': [measurement],
    'selected_batch': 8,
    'projected_51623_variant_hours': 51623 / measurement['variants_per_second'] / 3600,
    'completed': True,
}
(root / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
print('BATCH_CHECK_COMPLETE ' + json.dumps(report), flush=True)
PY
printf 'READY_FOR_INSPECTED_DRY_RUN\n'
