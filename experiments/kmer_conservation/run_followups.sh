#!/bin/bash
set -euo pipefail
cd /data/issue568/experiments/kmer_conservation
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=2
uv run --locked kmer-screen --root /data/issue568/synthetic --split dev
for setting in '255 9' '1024 13'; do
  read -r width k <<< "$setting"
  for divisor in 1 4; do
    uv run --locked kmer-screen --root /data/issue568 --split dev --width "$width" --k "$k" --divisor "$divisor"
  done
  uv run --locked kmer-screen --root /data/issue568 --split dev --width "$width" --k "$k" --mask
  uv run --locked kmer-screen --root /data/issue568 --split dev --width 4096 --k "$k"
done
