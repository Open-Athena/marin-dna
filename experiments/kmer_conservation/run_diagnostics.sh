#!/bin/bash
set -euo pipefail
cd /data/issue568/experiments/kmer_conservation
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=2
for k in 9 13; do
  uv run --locked kmer-screen --root /data/issue568/v2/synthetic --split dev --width 4096 --k "$k"
done
uv run --locked python -m kmer_conservation.diagnostics --root /data/issue568/v2 --split dev --mode union
for setting in '255 9' '1024 13'; do
  read -r width k <<< "$setting"
  uv run --locked python -m kmer_conservation.diagnostics --root /data/issue568/v2 --split dev --mode verify --width "$width" --k "$k"
done
uv run --locked python -m kmer_conservation.report --root /data/issue568/v2 --out /data/issue568/v2/report --figures
uv run --locked python -m kmer_conservation.report --root /data/issue568/v2/synthetic --out /data/issue568/v2/synthetic/report
uv run --locked python -m kmer_conservation.report --root /data/issue568/v2/synthetic-paralogs10 --out /data/issue568/v2/synthetic-paralogs10/report
