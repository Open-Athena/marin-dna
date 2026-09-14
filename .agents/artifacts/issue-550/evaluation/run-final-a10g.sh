#!/bin/bash
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:$PATH
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
: "${ISSUE550_SHUTDOWN_EPOCH:?Pass the sixteen-hour worker deadline}"
finish() {
    status=$?
    echo "$status" > /opt/issue550/vep-exit.txt
    echo "VEP_WORKER_EXIT $status"
    sync
    # Final records travel over the existing log stream; allow it to flush.
    sleep 5
    sudo shutdown -h now
}
trap finish EXIT
cd /opt/issue550/repo/snakemake/analysis/evals_v2
uv run --locked python /opt/issue550/repo/.agents/artifacts/issue-550/evaluation/run-10k-a10g.py \
    --model dna-exp550-rag46m-five-regions-v1-step-100000 \
    --shutdown-epoch "$ISSUE550_SHUTDOWN_EPOCH" --cores 4 --with-probes --execute
uv run --locked python /opt/issue550/repo/.agents/artifacts/issue-550/evaluation/collect-final-a10g.py
