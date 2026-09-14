#!/bin/bash
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:$PATH
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
: "${ISSUE550_SHUTDOWN_EPOCH:?Pass the budgeted worker deadline}"
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
export ISSUE550_MODEL=${ISSUE550_MODEL:-dna-exp550-rag46m-five-regions-v1-step-100000}
case "$ISSUE550_MODEL" in
    dna-exp550-rag46m-five-regions-v1-step-50000|dna-exp550-rag46m-five-regions-v1-step-100000) ;;
    *) echo "Unsupported checkpoint model" >&2; exit 2 ;;
esac
cd /opt/issue550/repo/snakemake/analysis/evals_v2
uv run --locked python /opt/issue550/repo/.agents/artifacts/issue-550/evaluation/run-10k-a10g.py \
    --model "$ISSUE550_MODEL" \
    --shutdown-epoch "$ISSUE550_SHUTDOWN_EPOCH" --cores 4 --with-probes --execute
uv run --locked python /opt/issue550/repo/.agents/artifacts/issue-550/evaluation/collect-final-a10g.py --model "$ISSUE550_MODEL"
