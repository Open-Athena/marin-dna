#!/bin/bash
set -euo pipefail
cd /data/issue568/experiments/kmer_conservation
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=2
for setting in '255 9' '1024 13'; do
  read -r width k <<< "$setting"
  uv run --locked python -m kmer_conservation.linclust --root /data/issue568 --split dev --width "$width" --k "$k" --mmseqs /data/issue568/tools/mmseqs/bin/mmseqs
  for hashes in 32 128 512; do
    uv run --locked python -m kmer_conservation.sketch --root /data/issue568 --split dev --width "$width" --k "$k" --hashes "$hashes" --method scan
    for rows in 1 2 4; do
      uv run --locked python -m kmer_conservation.sketch --root /data/issue568 --split dev --width "$width" --k "$k" --hashes "$hashes" --method lsh --rows "$rows"
    done
  done
done
