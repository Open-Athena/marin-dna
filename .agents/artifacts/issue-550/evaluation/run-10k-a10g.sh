#!/usr/bin/env bash
# Run on the bounded A10G worker; preserve logs and terminate on success or failure.
set -euo pipefail
cd /opt/issue550/repo/snakemake/analysis/evals_v2
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
artifact_prefix=s3://oa-bolinas/marin/MarinDNA/exp550_rag_five_regions/evaluation/a10g-10k-20260911
finish() {
  run_status=$?
  trap - EXIT
  printf '%s\n' "$run_status" > /opt/issue550/a10g-exit-status.txt
  # Standard Snakemake publishes each finished rule's outputs to canonical S3.
  # Preserve any local results after a failure in a separate recovery prefix.
  for attempt in 1 2 3; do
    if aws s3 sync /opt/issue550/storage/s3/oa-bolinas/snakemake/analysis/evals_v2/results/ "$artifact_prefix/recovery-results/" --exclude '*' --include '*.parquet' --only-show-errors &&
       aws s3 sync /opt/issue550/batch-sweep/ "$artifact_prefix/batch-sweep/" --only-show-errors &&
       aws s3 sync /opt/issue550/ "$artifact_prefix/runtime/" --exclude '*' --include '*.log' --include 'a10g-*.json' --include 'a10g-*.yaml' --include 'a10g-exit-status.txt' --only-show-errors; then
      break
    fi
    sleep 10
  done
  sudo shutdown -h now
  exit "$run_status"
}
trap finish EXIT
/home/ubuntu/.local/bin/uv run --locked --group genome-s3 python ../../../.agents/artifacts/issue-550/evaluation/run-10k-a10g.py --execute
