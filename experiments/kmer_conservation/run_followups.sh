#!/bin/bash
set -euo pipefail
cd /data/issue568/experiments/kmer_conservation
BENCH_ROOT=${BENCH_ROOT:-/data/issue568/v2}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=2
uv run --locked kmer-screen --root "$BENCH_ROOT/synthetic" --split dev
for setting in '255 9' '1024 13'; do
  read -r width k <<< "$setting"
  for divisor in 1 4; do
    uv run --locked kmer-screen --root "$BENCH_ROOT" --split dev --width "$width" --k "$k" --divisor "$divisor"
  done
  uv run --locked kmer-screen --root "$BENCH_ROOT" --split dev --width "$width" --k "$k" --mask
  uv run --locked kmer-screen --root "$BENCH_ROOT" --split dev --width 4096 --k "$k"
done
