#!/bin/bash
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:$PATH
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
: "${ISSUE550_SHUTDOWN_EPOCH:?Pass the four-hour worker deadline}"
finish() {
    status=$?
    echo "$status" > /opt/issue550/vep-exit.txt
    echo "VEP_WORKER_EXIT $status"
    sync
    # Allow the live log follower to collect the final receipt before stopping.
    if [ "$(( $(date +%s) + 60 ))" -lt "$ISSUE550_SHUTDOWN_EPOCH" ]; then
        sudo shutdown -h +1
    else
        sudo shutdown -h now
    fi
}
trap finish EXIT
cd /opt/issue550/repo/snakemake/analysis/evals_v2
uv run --locked python /opt/issue550/repo/.agents/artifacts/issue-550/evaluation/run-10k-a10g.py \
    --model dna-exp550-rag46m-five-regions-v1-step-20000 \
    --shutdown-epoch "$ISSUE550_SHUTDOWN_EPOCH" --execute
uv run --locked python /opt/issue550/repo/.agents/artifacts/issue-550/evaluation/collect-20k-a10g.py
